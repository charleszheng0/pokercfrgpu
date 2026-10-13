# Preserved CUDA experiments

These original `.cu` files are kept together for later audit: `vector_add.cu`, `reduction.cu`, `regretmatch.cu`, and `kuhn_flat_cuda.cu`. They were moved without changing their kernels. `regretmatch.cu` visibly contains unfinished syntax and scope errors, so it should be treated as a draft. None of these files is part of the validated CPU Hold'em path, and none has been compiled or benchmarked on this Apple M2 host. CUDA integration remains gated on CPU correctness and profiling.
