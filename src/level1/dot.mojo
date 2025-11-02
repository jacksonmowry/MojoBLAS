from gpu import thread_idx, block_idx, block_dim, lane_id
from gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from layout import Layout, LayoutTensor
from gpu.warp import sum as warp_sum
from math import ceildiv
from buffer import NDBuffer, DimList
from algorithm import sum
from gpu.warp import sum as warp_sum, WARP_SIZE
from layout import Layout, LayoutTensor

alias BLOCKS_PER_GRID = (1, 1)
alias THREADS_PER_BLOCK = (WARP_SIZE, 1)

fn dot_product[
    dtype: DType,
    size: Int
](
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]]
) raises -> Scalar[dtype]:
    alias n_warps = (size + WARP_SIZE - 1) // WARP_SIZE
    alias in_layout = Layout.row_major(size)
    alias out_layout = Layout.row_major(n_warps)
    alias n_blocks = (ceildiv(size, WARP_SIZE), 1)

    ctx = DeviceContext()

    out = ctx.enqueue_create_buffer[dtype](n_warps).enqueue_fill(0)
    a_device = ctx.enqueue_create_buffer[dtype](size).enqueue_fill(0)
    b_device = ctx.enqueue_create_buffer[dtype](size).enqueue_fill(0)

    with a_device.map_to_host() as a_host, b_device.map_to_host() as b_host:
        for i in range(size):
            a_host[i] = a[i]
            b_host[i] = b[i]


    a_tensor = LayoutTensor[mut=False, dtype, in_layout](a_device.unsafe_ptr())
    b_tensor = LayoutTensor[mut=False, dtype, in_layout](b_device.unsafe_ptr())
    out_tensor = LayoutTensor[mut=True, dtype, out_layout](out.unsafe_ptr())

    ctx.enqueue_function[
        dot_product_kernel[in_layout, out_layout, size, dtype, WARP_SIZE]
    ](
        out_tensor,
        a_tensor,
        b_tensor,
        grid_dim=n_blocks,
        block_dim=THREADS_PER_BLOCK,
    )

    ctx.synchronize()

    with out.map_to_host() as out_host:
        var device_ptr = out_host.unsafe_ptr()
        var actual_ndbuf = NDBuffer[dtype, 1](device_ptr, DimList(n_warps))
        return sum(actual_ndbuf)


fn dot_product_kernel[
    in_layout: Layout, out_layout: Layout, size: Int, dtype: DType, warp_size: Int
](
    output: LayoutTensor[mut=True, dtype, out_layout],
    a: LayoutTensor[mut=False, dtype, in_layout],
    b: LayoutTensor[mut=False, dtype, in_layout],
):
    global_i = block_dim.x * block_idx.x + thread_idx.x

    # Each thread computes one partial product using vectorized approach as values in Mojo are SIMD based
    var partial_product: Scalar[dtype] = 0
    if global_i < UInt(size):
        partial_product = (a[global_i] * b[global_i]).reduce_add()

    # warp_sum() replaces all the shared memory + barriers + tree reduction
    total = warp_sum(partial_product)

    # Only lane 0 writes the result (all lanes have the same total)
    if lane_id() == 0:
        output[global_i // UInt(warp_size)] = total
