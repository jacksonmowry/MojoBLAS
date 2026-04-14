from gpu import thread_idx, block_idx, block_dim, grid_dim
from gpu.host import DeviceContext
from math import ceildiv

comptime TBsize = 512

# level2.spmv
# Performs symmetric packed matrix-vector multiplication
#    y := alpha*A*x + beta*y,
# where A is an n by n symmetric matrix stored in packed format.
# AP uses column-major (Fortran-order) packed storage:
#   Upper triangular: AP[j*(j+1)/2 + i] = A[i,j]  for 0 <= i <= j < n
#   Lower triangular: AP[j*(2*n-j-1)/2 + i] = A[i,j]  for 0 <= j <= i < n
#
# Each output element y[i] is independent, so the routine is fully
# parallelizable with no write-after-read data hazards.

fn sspmv_device(
    uplo: Int,
    n: Int,
    alpha: Float32,
    AP: UnsafePointer[Float32, ImmutAnyOrigin],
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int,
    beta: Float32,
    y: UnsafePointer[Float32, MutAnyOrigin],
    incy: Int,
):
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var n_threads = grid_dim.x * block_dim.x

    for i in range(global_i, n, n_threads):
        var sum = Scalar[DType.float32](0)
        for j in range(n):
            var idx: Int
            if uplo:  # upper triangular
                if j >= i:
                    idx = j * (j + 1) // 2 + i
                else:
                    idx = i * (i + 1) // 2 + j
            else:  # lower triangular
                if j <= i:
                    idx = j * (2 * n - j - 1) // 2 + i
                else:
                    idx = i * (2 * n - i - 1) // 2 + j
            sum += AP[idx] * x[j * incx]
        y[i * incy] = alpha * sum + beta * y[i * incy]


fn dspmv_device(
    uplo: Int,
    n: Int,
    alpha: Float64,
    AP: UnsafePointer[Float64, ImmutAnyOrigin],
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int,
    beta: Float64,
    y: UnsafePointer[Float64, MutAnyOrigin],
    incy: Int,
):
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var n_threads = grid_dim.x * block_dim.x

    for i in range(global_i, n, n_threads):
        var sum = Scalar[DType.float64](0)
        for j in range(n):
            var idx: Int
            if uplo:  # upper triangular
                if j >= i:
                    idx = j * (j + 1) // 2 + i
                else:
                    idx = i * (i + 1) // 2 + j
            else:  # lower triangular
                if j <= i:
                    idx = j * (2 * n - j - 1) // 2 + i
                else:
                    idx = i * (2 * n - i - 1) // 2 + j
            sum += AP[idx] * x[j * incx]
        y[i * incy] = alpha * sum + beta * y[i * incy]


fn blas_spmv[dtype: DType](
    uplo: Bool,
    n: Int,
    alpha: Scalar[dtype],
    d_AP: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    d_x: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incx: Int,
    beta: Scalar[dtype],
    d_y: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    incy: Int,
    ctx: DeviceContext,
) raises:
    var uplo_i = 1 if uplo else 0

    @parameter
    if dtype == DType.float32:
        ctx.enqueue_function[sspmv_device, sspmv_device](
            uplo_i, n,
            alpha, d_AP,
            d_x, incx,
            beta, d_y, incy,
            grid_dim=ceildiv(n, TBsize),
            block_dim=TBsize,
        )
    elif dtype == DType.float64:
        ctx.enqueue_function[dspmv_device, dspmv_device](
            uplo_i, n,
            alpha, d_AP,
            d_x, incx,
            beta, d_y, incy,
            grid_dim=ceildiv(n, TBsize),
            block_dim=TBsize,
        )
    else:
        raise Error("blas_spmv: Unsupported type")

    ctx.synchronize()
