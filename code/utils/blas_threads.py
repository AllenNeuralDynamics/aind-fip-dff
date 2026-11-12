"""Pin BLAS/OpenMP to one thread per process. Import before numpy.

The matrix-vector products in `nonlinear_fit`'s gradient are too small to
benefit from threaded BLAS (16 OpenBLAS threads are ~40x slower than one),
so parallelism comes from joblib across assets instead.
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
