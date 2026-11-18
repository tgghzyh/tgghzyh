## Open source contributions

### vLLM

Reduced host memory use in online beam search by retaining at most `beam_width` finished hypotheses between decoding steps. In a Dolly workload with beam width 4, the peak number retained fell from 36 to 4. [PR #66841](https://github.com/vllm-project/vllm/pull/66841) · [Upstream commit](https://github.com/vllm-project/vllm/commit/0c80c3a6aa0201d1b4f71915f90c03fd64837cec)

### PyTorch

Fixed a shape mismatch in NCCL2 `gather_single` when gathering multidimensional tensors. Added regression tests for 2D and channels-last inputs. [PR #205316](https://github.com/pytorch/pytorch/pull/205316) · [Upstream commit](https://github.com/pytorch/pytorch/commit/e47f90e068a41e5fc8d253e7bcf8c84b56b23e79)

### NVIDIA NCCL

Two NCCL4Py optimizations I contributed to [NVIDIA NCCL](https://github.com/NVIDIA/nccl) landed on the `dev` branch.

- **Small reductions.** Reused buffer metadata for in-place reductions and cached PyTorch dtype lookups to reduce Python overhead. [PR #2609](https://github.com/NVIDIA/nccl/pull/2609) · [Upstream commit](https://github.com/NVIDIA/nccl/commit/f84202b0fd8876e6ec06dd570f27044c5bac364b)
- **Framework imports.** Avoided importing PyTorch or CuPy when resolving buffers that don't need either framework. [PR #2610](https://github.com/NVIDIA/nccl/pull/2610) · [Upstream commit](https://github.com/NVIDIA/nccl/commit/27a903ff2b21a12c615f385408d66cbb86b7ceed)
