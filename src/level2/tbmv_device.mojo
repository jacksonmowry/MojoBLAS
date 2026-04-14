from gpu import thread_idx, block_idx, block_dim, grid_dim
from gpu.host import DeviceContext

# level2.tbmv
# Performs triangular band matrix-vector multiplication
#    x := A*x,   or   x := A**T*x,
# where x is an n element vector and A is an n by n unit, or non-unit,
# upper or lower triangular band matrix with k super/sub-diagonals.
#
# Row-major band storage:
#   Upper: A[i,j] stored at A_band[i*lda + (j-i)], for j in [i, min(n-1, i+k)]
#   Lower: A[i,j] stored at A_band[i*lda + (i-j)], for j in [max(0, i-k), i]
#
# uplo:  0 = lower triangular, 1 = upper triangular
# trans: 0 = no transpose,     1 = transpose
# diag:  0 = non-unit diagonal, 1 = unit diagonal (diagonal not accessed)

fn stbmv_device(
    uplo: Int,
    trans: Int,
    diag: Int,
    n: Int,
    k: Int,
    A: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int,
    x: UnsafePointer[Float32, MutAnyOrigin],
    incx: Int,
):
    var tid = block_dim.x * block_idx.x + thread_idx.x
    if tid != 0:
        return

    if not trans:
        if uplo:
            # Upper triangular, no-transpose: x[i] = sum_{j=i}^{min(n-1,i+k)} A[i,j]*x[j]
            # Process forward so original x[j] (j>i) is still valid when computing x[i]
            for i in range(n):
                var temp = Float32(0)
                var j_end = min(n - 1, i + k)
                for j in range(i, j_end + 1):
                    if j == i and diag:
                        temp += x[j * incx]
                    else:
                        temp += A[i * lda + (j - i)] * x[j * incx]
                x[i * incx] = temp
        else:
            # Lower triangular, no-transpose: x[i] = sum_{j=max(0,i-k)}^{i} A[i,j]*x[j]
            # Process backward so original x[j] (j<i) is still valid when computing x[i]
            for idx in range(n):
                var i = n - 1 - idx
                var temp = Float32(0)
                var j_start = max(0, i - k)
                for j in range(j_start, i + 1):
                    if j == i and diag:
                        temp += x[j * incx]
                    else:
                        temp += A[i * lda + (i - j)] * x[j * incx]
                x[i * incx] = temp
    else:
        if uplo:
            # Upper triangular, transpose: x[j] = sum_{i=max(0,j-k)}^{j} A[i,j]*x[i]
            # Process backward so original x[i] (i<j) is still valid when computing x[j]
            for idx in range(n):
                var j = n - 1 - idx
                var temp = Float32(0)
                var i_start = max(0, j - k)
                for i in range(i_start, j + 1):
                    if i == j and diag:
                        temp += x[i * incx]
                    else:
                        temp += A[i * lda + (j - i)] * x[i * incx]
                x[j * incx] = temp
        else:
            # Lower triangular, transpose: x[j] = sum_{i=j}^{min(n-1,j+k)} A[i,j]*x[i]
            # Process forward so original x[i] (i>j) is still valid when computing x[j]
            for j in range(n):
                var temp = Float32(0)
                var i_end = min(n - 1, j + k)
                for i in range(j, i_end + 1):
                    if i == j and diag:
                        temp += x[i * incx]
                    else:
                        temp += A[i * lda + (i - j)] * x[i * incx]
                x[j * incx] = temp


fn dtbmv_device(
    uplo: Int,
    trans: Int,
    diag: Int,
    n: Int,
    k: Int,
    A: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int,
    x: UnsafePointer[Float64, MutAnyOrigin],
    incx: Int,
):
    var tid = block_dim.x * block_idx.x + thread_idx.x
    if tid != 0:
        return

    if not trans:
        if uplo:
            # Upper triangular, no-transpose: x[i] = sum_{j=i}^{min(n-1,i+k)} A[i,j]*x[j]
            # Process forward so original x[j] (j>i) is still valid when computing x[i]
            for i in range(n):
                var temp = Float64(0)
                var j_end = min(n - 1, i + k)
                for j in range(i, j_end + 1):
                    if j == i and diag:
                        temp += x[j * incx]
                    else:
                        temp += A[i * lda + (j - i)] * x[j * incx]
                x[i * incx] = temp
        else:
            # Lower triangular, no-transpose: x[i] = sum_{j=max(0,i-k)}^{i} A[i,j]*x[j]
            # Process backward so original x[j] (j<i) is still valid when computing x[i]
            for idx in range(n):
                var i = n - 1 - idx
                var temp = Float64(0)
                var j_start = max(0, i - k)
                for j in range(j_start, i + 1):
                    if j == i and diag:
                        temp += x[j * incx]
                    else:
                        temp += A[i * lda + (i - j)] * x[j * incx]
                x[i * incx] = temp
    else:
        if uplo:
            # Upper triangular, transpose: x[j] = sum_{i=max(0,j-k)}^{j} A[i,j]*x[i]
            # Process backward so original x[i] (i<j) is still valid when computing x[j]
            for idx in range(n):
                var j = n - 1 - idx
                var temp = Float64(0)
                var i_start = max(0, j - k)
                for i in range(i_start, j + 1):
                    if i == j and diag:
                        temp += x[i * incx]
                    else:
                        temp += A[i * lda + (j - i)] * x[i * incx]
                x[j * incx] = temp
        else:
            # Lower triangular, transpose: x[j] = sum_{i=j}^{min(n-1,j+k)} A[i,j]*x[i]
            # Process forward so original x[i] (i>j) is still valid when computing x[j]
            for j in range(n):
                var temp = Float64(0)
                var i_end = min(n - 1, j + k)
                for i in range(j, i_end + 1):
                    if i == j and diag:
                        temp += x[i * incx]
                    else:
                        temp += A[i * lda + (i - j)] * x[i * incx]
                x[j * incx] = temp


fn blas_tbmv[dtype: DType](
    uplo: Int,
    trans: Bool,
    diag: Int,
    n: Int,
    k: Int,
    d_A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    lda: Int,
    d_x: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    incx: Int,
    ctx: DeviceContext,
) raises:

    # NOTE: add error checking here?
    # check n > 0
    # check k >= 0
    # check lda >= k + 1
    # check incx > 0

    var trans_i = 1 if trans else 0

    @parameter
    if dtype == DType.float32:
        ctx.enqueue_function[stbmv_device, stbmv_device](
            uplo, trans_i, diag,
            n, k, d_A, lda,
            d_x, incx,
            grid_dim=1,
            block_dim=1,
        )
    elif dtype == DType.float64:
        ctx.enqueue_function[dtbmv_device, dtbmv_device](
            uplo, trans_i, diag,
            n, k, d_A, lda,
            d_x, incx,
            grid_dim=1,
            block_dim=1,
        )
    else:
        raise Error("blas_tbmv: Unsupported type")

    ctx.synchronize()
