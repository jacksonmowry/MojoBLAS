from gpu import grid_dim, block_dim, global_idx
from gpu.host import DeviceContext
from math import ceildiv

comptime TBsize = 512

fn swap_device[dtype: DType](
    n: Int,
    x: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    incx: Int,
    y: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    incy: Int
):
    if (n <= 0):
        return
    if (incx == 0 or incy == 0):
        return

    var global_i = global_idx.x
    var n_threads = Int(grid_dim.x * block_dim.x)

    # Multiple cells per thread
    for i in range(global_i, n, n_threads):
        var tmp = x[i * incx]
        x[i * incx] = y[i * incy]
        y[i * incy] = tmp


fn blas_swap[dtype: DType](
    n: Int,
    d_x: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    incx: Int,
    d_y: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    incy: Int,
    ctx: DeviceContext
) raises:
    kernel = ctx.compile_function[swap_device[dtype], swap_device[dtype]]()
    ctx.enqueue_function(
        kernel,
        n,
        d_x, incx,
        d_y, incy,
        grid_dim=ceildiv(n, TBsize),
        block_dim=TBsize,
    )
    ctx.synchronize()
