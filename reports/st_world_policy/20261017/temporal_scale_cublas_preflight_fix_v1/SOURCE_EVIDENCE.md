# Deterministic cuBLAS evidence

The installed interpreter is `/root/miniconda3/envs/ddp/bin/python`, PyTorch
`2.5.1+cu124`; installed source checksums are in `INSTALLED_SOURCE_PINS.json`.

The installed `torch/__init__.py` documentation for
`use_deterministic_algorithms` (lines 1295–1304) says CUDA matrix operations
including mm/mv/bmm reject deterministic execution without either supported
workspace setting. Its behavior matches the actual saved real4 exception.

[PyTorch v2.5.1 Context.cpp](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/Context.cpp)
implements the environment check and caches its result in
`alertCuBLASConfigNotDeterministic`. Therefore this repair sets the child process
environment before starting Python, rather than trying to recover by changing
the environment after an operation fails. Checking `torch.cuda.is_initialized()`
is non-initializing; the installed `torch/cuda/__init__.py` source is also pinned.

[NVIDIA cuBLAS 12.4.1 reproducibility documentation](https://docs.nvidia.com/cuda/archive/12.4.1/cublas/index.html#results-reproducibility)
describes workspace-related algorithm changes with concurrent streams and both
supported environment choices. The chosen `:4096:8` uses approximately 24 MiB of
additional library GPU workspace; no GPU memory measurement is claimed here.

This fixes a required cuBLAS startup precondition. It does not establish that all
FA2/model operations are bitwise deterministic or that the four real gradients
have been successfully measured. Those remain properties of the separate
bounded actual execution, not of these CPU tests.
