//! Spike proof: in-process Gemma-4-E4B MTP (NextN draft head) speculative
//! decoding + grammar, fully from Rust via the forked `llama-cpp-2`.
//!
//! Proves the fork (with the submodule bumped to a gemma4-MTP-capable
//! llama.cpp) gives us everything the witness `LlmBackend` llamacpp path
//! needs: load target + MTP-head, drive `MtpSpeculative`, verify drafts,
//! roll back KV, and optionally constrain with a GBNF grammar.
//!
//! Usage:
//!   cargo run -p mtp_grammar --features metal --release -- \
//!     <target.gguf> <mtp-head.gguf> [--grammar] [--max N] [--prompt P]

#![allow(clippy::cast_possible_wrap, clippy::cast_sign_loss, clippy::cast_possible_truncation)]

use std::num::NonZeroU32;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::speculative::{MtpSpeculative, MtpSpeculativeParams};
use llama_cpp_2::token::LlamaToken;

// JSON schema for the candidate-extraction output. Converted to GBNF in-process
// via the crate's `json_schema_to_grammar` (the same json_schema -> GBNF path
// production uses) so the grammar terminates correctly.
const CANDIDATE_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "candidates": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "kind": {"type": "string", "enum": ["fact","decision","action_item","open_question"]},
          "text": {"type": "string"},
          "evidence_segment_ids": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["kind","text","evidence_segment_ids"]
      }
    }
  },
  "required": ["candidates"]
}"#;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        bail!("usage: mtp_grammar <target.gguf> <mtp-head.gguf> [--grammar] [--max N] [--prompt P]");
    }
    let target_path = &args[1];
    let draft_path = &args[2];
    let use_grammar = args.iter().any(|a| a == "--grammar");
    let max_new: usize = args
        .iter()
        .position(|a| a == "--max")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let prompt_file = args
        .iter()
        .position(|a| a == "--prompt-file")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let prompt = if let Some(pf) = prompt_file {
        std::fs::read_to_string(&pf).with_context(|| format!("read prompt file {pf}"))?
    } else {
        args
        .iter()
        .position(|a| a == "--prompt")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| {
            "<start_of_turn>user\nList three primary colors as JSON \
             {\"candidates\":[{\"kind\":\"fact\",\"text\":\"...\"}]}.\
             <end_of_turn>\n<start_of_turn>model\n"
                .to_string()
        })
    };

    let backend = LlamaBackend::init()?;
    let model_params = LlamaModelParams::default().with_n_gpu_layers(999);

    eprintln!("loading target: {target_path}");
    let target_model = LlamaModel::load_from_file(&backend, target_path, &model_params)
        .context("load target model")?;
    eprintln!("loading MTP draft head: {draft_path}");
    let draft_model = LlamaModel::load_from_file(&backend, draft_path, &model_params)
        .context("load MTP draft head")?;

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(16384).unwrap()))
        .with_n_batch(16384);
    let target_ctx = target_model
        .new_context(&backend, ctx_params.clone())
        .context("target ctx")?;
    // Gemma4Assistant (MTP head) cannot init a standalone context — it must be
    // bound to the target context via ctx_other.
    let draft_ctx = draft_model
        .new_context(&backend, ctx_params.with_ctx_other(target_ctx.as_ptr()))
        .context("draft ctx")?;

    let mut spec = MtpSpeculative::new(
        target_ctx,
        draft_ctx,
        MtpSpeculativeParams { n_max: 4, n_min: 0, p_min: 0.0 },
    )
    .context("MtpSpeculative::new (init draft-mtp)")?;
    eprintln!("MtpSpeculative initialized OK (draft-mtp, n_max=4)");

    let prompt_tokens = target_model.str_to_token(&prompt, AddBos::Always)?;
    let n_prompt = prompt_tokens.len();
    eprintln!("prompt tokens: {n_prompt}");

    // --- prime target with the prompt ---
    let mut batch = LlamaBatch::new(16384, 1);
    for (i, &t) in prompt_tokens.iter().enumerate() {
        batch.add(t, i as i32, &[0], i == n_prompt - 1)?;
    }
    spec.target_context_mut().decode(&mut batch).context("prompt decode")?;
    spec.process(&batch).context("process(prompt)")?;
    spec.begin(&prompt_tokens).context("begin")?;

    // sampler: grammar+greedy, or plain greedy
    // NOTE: grammar mode here builds a raw `chain_simple([grammar, greedy])` and
    // applies the grammar mask on every token. That hits llama.cpp's
    // `GGML_ASSERT(!stacks.empty())` once a multi-char token advances the grammar
    // past a rule boundary — the server avoids this with `common_sampler`'s
    // optimistic path (sample, then validate/resample). The correct fix is to
    // wrap `common_sampler` / `common_sampler_sample_and_accept_n` in the fork
    // (tracked separately); greedy mode below is the validated MTP path.
    let mut sampler = if use_grammar {
        let gbnf = llama_cpp_2::json_schema_to_grammar(CANDIDATE_SCHEMA)
            .context("json_schema_to_grammar")?;
        eprintln!("json_schema -> GBNF: {} bytes", gbnf.len());
        LlamaSampler::chain_simple([
            LlamaSampler::grammar(&target_model, &gbnf, "root").context("grammar")?,
            LlamaSampler::greedy(),
        ])
    } else {
        LlamaSampler::greedy()
    };
    eprintln!("sampler: {}", if use_grammar { "grammar(JSON)+greedy" } else { "greedy" });

    // first token off the prompt's last logit
    let mut id_last = sampler.sample(spec.target_context(), batch.n_tokens() - 1);
    sampler.accept(id_last);

    let mut output: Vec<LlamaToken> = vec![id_last];
    let mut context = prompt_tokens.clone();
    context.push(id_last);
    let mut n_past: i32 = n_prompt as i32; // position where id_last will be decoded

    let mut total_draft = 0usize;
    let mut total_accepted = 0usize;
    let mut rounds = 0usize;
    let mut stream_dec = encoding_rs::UTF_8.new_decoder();
    let t0 = Instant::now();

    'gen: while output.len() < max_new {
        rounds += 1;
        let draft = if std::env::var("NO_DRAFT").is_ok() {
            Vec::new()
        } else {
            spec.draft(n_past, id_last, &context).context("draft")?
        };
        total_draft += draft.len();

        // verify batch: id_last @ n_past, then each draft token, all with logits
        batch.clear();
        batch.add(id_last, n_past, &[0], true)?;
        for (k, &d) in draft.iter().enumerate() {
            batch.add(d, n_past + 1 + k as i32, &[0], true)?;
        }
        spec.target_context_mut().decode(&mut batch).context("verify decode")?;
        spec.process(&batch).context("process(verify)")?;

        // greedy/grammar verify against the draft
        let mut n_acc = 0usize;
        let next_id;
        let mut done = false;
        loop {
            let pred = sampler.sample(spec.target_context(), n_acc as i32);
            sampler.accept(pred);
            output.push(pred);
            context.push(pred);
            if let Ok(p) = target_model.token_to_piece(pred, &mut stream_dec, true, None) {
                use std::io::Write;
                eprint!("{p}");
                std::io::stderr().flush().ok();
            }
            // Grammar-aware termination: once the constrained output is a
            // complete JSON value, the grammar's stacks empty and the next
            // sample() would abort — stop here (what common_sampler does).
            if use_grammar && json_complete(&target_model, &output) {
                next_id = pred;
                done = true;
                break;
            }
            if n_acc < draft.len() && pred == draft[n_acc] {
                n_acc += 1;
                if output.len() >= max_new {
                    next_id = pred;
                    break;
                }
            } else {
                next_id = pred;
                break;
            }
        }
        total_accepted += n_acc;
        if done {
            break 'gen;
        }

        // commit: id_last + accepted drafts; drop rejected drafts from KV
        let committed_end = (n_past + 1 + n_acc as i32) as u32;
        spec.target_context_mut()
            .clear_kv_cache_seq(Some(0), Some(committed_end), None)
            .context("kv rollback")?;
        spec.accept(n_acc as u16).context("spec.accept")?;

        n_past = committed_end as i32;
        id_last = next_id;

        if target_model.is_eog_token(next_id) {
            break 'gen;
        }
    }

    let secs = t0.elapsed().as_secs_f64();
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut text = String::new();
    for &t in &output {
        text.push_str(&target_model.token_to_piece(t, &mut decoder, true, None)?);
    }

    println!("\n=========== RUST-NATIVE MTP{} ===========", if use_grammar { " + GRAMMAR" } else { "" });
    println!("generated {} tok in {secs:.2}s = {:.1} tok/s", output.len(), output.len() as f64 / secs);
    println!(
        "rounds {rounds} | drafted {total_draft} accepted {total_accepted} = {:.3} acceptance | {:.2} tok/forward",
        if total_draft > 0 { total_accepted as f64 / total_draft as f64 } else { 0.0 },
        output.len() as f64 / rounds.max(1) as f64
    );
    println!("\noutput:\n{}", text.replace('\n', " "));
    Ok(())
}

/// True once the decoded tokens form a complete, parseable JSON value.
fn json_complete(model: &LlamaModel, tokens: &[LlamaToken]) -> bool {
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut s = String::new();
    for &t in tokens {
        if let Ok(p) = model.token_to_piece(t, &mut decoder, true, None) {
            s.push_str(&p);
        }
    }
    let trimmed = s.trim();
    !trimmed.is_empty() && serde_json::from_str::<serde_json::Value>(trimmed).is_ok()
}
