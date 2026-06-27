//! Safe wrapper around llama.cpp's `common_sampler` (the grammar-aware sampler
//! the server uses).
//!
//! Unlike a raw `grammar` + `greedy` [`crate::sampling::LlamaSampler`] chain —
//! which trips `GGML_ASSERT(!stacks.empty())` when a multi-character token
//! advances the grammar past a rule boundary — `common_sampler` samples then
//! validates/resamples against the grammar, so it is safe to drive
//! grammar-constrained (speculative) decoding token-by-token.

use std::ffi::CString;
use std::ptr::NonNull;

use crate::context::LlamaContext;
use crate::model::LlamaModel;
use crate::token::LlamaToken;

/// Errors returned by [`CommonSampler`].
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum CommonSamplerError {
    /// The grammar string contained an interior NUL byte.
    #[error("grammar string contained a NUL byte")]
    InvalidGrammar,
    /// llama.cpp returned a null sampler handle.
    #[error("llama.cpp failed to initialize common_sampler")]
    InitFailed,
}

/// RAII owner of a grammar-constrained `common_sampler`.
#[derive(Debug)]
pub struct CommonSampler {
    raw: NonNull<llama_cpp_sys_2::common_sampler>,
}

impl CommonSampler {
    /// Initialize a grammar-constrained sampler. `temp <= 0.0` decodes greedily
    /// (the right choice for structured/JSON extraction); `seed` only matters
    /// when `temp > 0.0`.
    ///
    /// # Errors
    /// Returns an error if the grammar has an interior NUL or llama.cpp cannot
    /// build the sampler.
    pub fn new_grammar(
        model: &LlamaModel,
        grammar: &str,
        temp: f32,
        seed: u32,
    ) -> Result<Self, CommonSamplerError> {
        let c_grammar = CString::new(grammar).map_err(|_| CommonSamplerError::InvalidGrammar)?;
        let raw = unsafe {
            llama_cpp_sys_2::llama_rs_common_sampler_init_grammar(
                model.model.as_ptr(),
                c_grammar.as_ptr(),
                temp,
                seed,
            )
        };
        let raw = NonNull::new(raw).ok_or(CommonSamplerError::InitFailed)?;
        Ok(Self { raw })
    }

    /// Sample one token from the logits at position `idx`. With
    /// `grammar_first = true` the grammar constraint is applied before sampling.
    pub fn sample(&mut self, ctx: &LlamaContext, idx: i32, grammar_first: bool) -> LlamaToken {
        let id = unsafe {
            llama_cpp_sys_2::llama_rs_common_sampler_sample(
                self.raw.as_ptr(),
                ctx.context.as_ptr(),
                idx,
                grammar_first,
            )
        };
        LlamaToken(id)
    }

    /// Advance the sampler + grammar state with an emitted token.
    /// `is_generated` marks it as model-generated (vs. a forced/prompt token).
    pub fn accept(&mut self, token: LlamaToken, is_generated: bool) {
        unsafe {
            llama_cpp_sys_2::llama_rs_common_sampler_accept(
                self.raw.as_ptr(),
                token.0,
                is_generated,
            );
        }
    }
}

impl Drop for CommonSampler {
    fn drop(&mut self) {
        unsafe {
            llama_cpp_sys_2::llama_rs_common_sampler_free(self.raw.as_ptr());
        }
    }
}
