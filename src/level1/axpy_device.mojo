from gpu import grid_dim, block_dim, global_idx

fn axpy_device[dtype: DType](
    n: Int, 
    a: SIMD[dtype, 1],
    x: UnsafePointer[SIMD[dtype, 1], ImmutAnyOrigin],
    incx: Int,
    y: UnsafePointer[SIMD[dtype, 1], MutAnyOrigin],
    incy: Int
):
    if (n <= 0):
        return
    if (a == 0):
        return
    if (incx == 0 or incy == 0):
        return

    var global_i = global_idx.x
    var n_threads = grid_dim.x * block_dim.x

    if (n_threads <= global_i):
        # Standard case: each thread gets 1 cell
        if (global_i < n):
            y[global_i*incy] += a * x[global_i*incx]
    
    else:
        # Multiple cells per thread
        for i in range(global_i, n, n_threads):
            y[i*incy] += a * x[i*incx]

