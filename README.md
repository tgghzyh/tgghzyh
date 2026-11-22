## Open source contributions

### vLLM

Fixed a pipeline-parallel inference bug in the Transformers backend. Later stages now pass received hidden states to their first decoder layer without reapplying embedding scaling or positional embeddings. This fixes a GPTBigCode shape mismatch and incorrect outputs in models such as Granite. [PR #67648](https://github.com/vllm-project/vllm/pull/67648) · [Upstream commit](https://github.com/vllm-project/vllm/commit/1c0b139293a22cce96c41c9fb32cbf2c31d30e7a)

Reduced host memory use in online beam search by retaining at most `beam_width` finished hypotheses between decoding steps. In a Dolly workload with beam width 4, the peak number retained fell from 36 to 4. [PR #66841](https://github.com/vllm-project/vllm/pull/66841) · [Upstream commit](https://github.com/vllm-project/vllm/commit/0c80c3a6aa0201d1b4f71915f90c03fd64837cec)

### SGLang

Fixed intermediate state indexing in SGLang's Triton Mamba kernel during speculative verification. When the buffer has room for more steps than a request uses, parent-state reads and snapshot writes now use the buffer's actual batch stride. [PR #51164](https://github.com/sgl-project/sglang/pull/51164) · [Upstream commit](https://github.com/sgl-project/sglang/commit/c0bdbb306b2c60557e2516b0d419eb293544a494)

### PyTorch

Fixed a shape mismatch in NCCL2 `gather_single` when gathering multidimensional tensors. Added regression tests for 2D and channels-last inputs. [PR #205316](https://github.com/pytorch/pytorch/pull/205316) · [Upstream commit](https://github.com/pytorch/pytorch/commit/e47f90e068a41e5fc8d253e7bcf8c84b56b23e79)

### NVIDIA NCCL

Two NCCL4Py optimizations I contributed to [NVIDIA NCCL](https://github.com/NVIDIA/nccl) landed on the `dev` branch.

- **Small reductions.** Reused buffer metadata for in-place reductions and cached PyTorch dtype lookups to reduce Python overhead. [PR #2609](https://github.com/NVIDIA/nccl/pull/2609) · [Upstream commit](https://github.com/NVIDIA/nccl/commit/f84202b0fd8876e6ec06dd570f27044c5bac364b)
- **Framework imports.** Avoided importing PyTorch or CuPy when resolving buffers that don't need either framework. [PR #2610](https://github.com/NVIDIA/nccl/pull/2610) · [Upstream commit](https://github.com/NVIDIA/nccl/commit/27a903ff2b21a12c615f385408d66cbb86b7ceed)
