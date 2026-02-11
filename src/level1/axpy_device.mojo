from gpu import grid_dim, block_dim, global_idx
from gpu.host import DeviceContext
from math import ceildiv

comptime TBsize = 512

fn axpy_device[dtype: DType](
    n: Int,
    a: Scalar[dtype],
    x: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incx: Int,
    y: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    incy: Int
):
    if (n <= 0):
        return
    if (a == 0):
        return
    if (incx == 0 or incy == 0):
        return

    var global_i = global_idx.x
    var n_threads = Int(grid_dim.x * block_dim.x)

    # Multiple cells per thread
    for i in range(global_i, n, n_threads):
        y[i*incy] += a * x[i*incx]


fn blas_axpy[dtype: DType](
    n: Int,
    a: Scalar[dtype],
    d_x: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incx: Int,
    d_y: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    incy: Int,
    ctx: DeviceContext
) raises:
    axpy_kernel = ctx.compile_function[axpy_device[dtype], axpy_device[dtype]]()
    ctx.enqueue_function(
        axpy_kernel,
        n, a,
        d_x, incx,
        d_y, incy,
        grid_dim=ceildiv(n, TBsize),
        block_dim=TBsize,
    )
    ctx.synchronize()
