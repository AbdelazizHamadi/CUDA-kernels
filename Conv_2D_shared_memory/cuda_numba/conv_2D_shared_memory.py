import numpy as np
from numba import cuda, float32, int32
import cupy as cp
import math
import torch
import torch.nn as nn


# PyTorch Convolution Function
def pytorch_conv2d(input_matrix, kernel):
    # Add batch and channel dimensions
    input_tensor = torch.tensor(input_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    kernel_tensor = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # create conv 2D layer
    conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel.shape, padding=MASK_RADIUS, bias=False)
    conv_layer.weight.data = kernel_tensor

    # Apply the convolution
    with torch.no_grad():
        output_tensor = conv_layer(input_tensor)

    return output_tensor.squeeze().numpy().astype(np.int32)


def get_best_tile_size(mask_width=3):
    # GPU shared memory limit (typically 48 KB for modern GPUs)
    # Select the GPU device
    device = cuda.get_current_device()

    mask_radius = mask_width // 2
    # get the max tile output considering the applied kernel width
    tile_width = int(math.sqrt(device.MAX_THREADS_PER_BLOCK) - mask_width + 1)
    # size of the shared memory
    w = tile_width + mask_width - 1
    return int(tile_width), int(mask_width), int(mask_radius), int(w)


@cuda.jit
def conv_2D_shared_memory(input_matrix, kernel, flatten_o_matrix, width, height):
    # Define shared memory
    shared_mem = cuda.shared.array(shape=(SHARED_MEM_SIZE, SHARED_MEM_SIZE), dtype=int32)
    # Calculate global thread coordinates
    tx: int = cuda.threadIdx.x
    ty: int = cuda.threadIdx.y

    row_o = cuda.blockIdx.y * TILE_WIDTH + ty
    col_o = cuda.blockIdx.x * TILE_WIDTH + tx

    row_i = row_o - MASK_RADIUS
    col_i = col_o - MASK_RADIUS

    if (row_i >= 0) and (row_i < height) and (col_i >= 0) and (col_i < width):
        # shared_mem[ty, tx] = input_matrix[row_i, col_i]
        shared_mem[ty, tx] = input_matrix[row_i * width + col_i]
    else:
        shared_mem[ty, tx] = 0.0
    cuda.syncthreads()

    # Perform convolution (simplified for example)
    if tx < TILE_WIDTH and ty < TILE_WIDTH:
        accum = 0.0
        for i in range(MASK_DIM):
            for j in range(MASK_DIM):
                accum += shared_mem[ty + i, tx + j] * kernel[i, j]
        if row_o < height and col_o < width:
            flatten_o_matrix[row_o * width + col_o] = accum


# get the best params given a kernel size depending on the GPU used
TILE_WIDTH, MASK_DIM, MASK_RADIUS, SHARED_MEM_SIZE = get_best_tile_size(mask_width=15)

# Example input matrix
matrix = cp.array([
    [35, 80, 91, 89, 55, 30, 36, 14],
    [11, 3, 35, 75, 71, 83, 6, 25],
    [72, 53, 99, 55, 90, 30, 13, 54],
    [16, 91, 27, 83, 94, 13, 58, 52],
    [17, 75, 29, 84, 18, 63, 94, 51],
    [68, 83, 21, 30, 23, 80, 58, 16],
    [52, 65, 97, 66, 82, 55, 60, 86],
    [59, 31, 35, 84, 56, 55, 50, 73]
]).astype(cp.int32)

N = 1 << 20  # 1048576
# square
width = height = int(math.sqrt(N))
# define matrix
matrix = cp.ones((width, height), dtype=cp.int32)

# define kernel
kernel = cp.ones((MASK_DIM, MASK_DIM)).astype(cp.int32)

# output matrix
o_matrix = cp.zeros(matrix.shape).astype(cp.int32)
o_matrix_flatten = o_matrix.flatten()

# kernel launch params
ThreadPerBlock = SHARED_MEM_SIZE
blocks_per_grid = (int((matrix.shape[0] + TILE_WIDTH - 1) / TILE_WIDTH))

conv_2D_shared_memory[(blocks_per_grid, blocks_per_grid), (ThreadPerBlock, ThreadPerBlock)](matrix.flatten(), kernel,
                                                                                     o_matrix_flatten, width, height)
cuda.synchronize()

# Convert CUDA output to NumPy array
custom_output = cp.asnumpy(o_matrix_flatten.reshape(matrix.shape))

print("Input Matrix:")
print(matrix)
print("kernel :")
print(kernel)
print("Output Matrix 'custom kernel':")
print(custom_output)
# PyTorch Convolution
torch_output = pytorch_conv2d(np.array(matrix.get()), np.array(kernel.get()))
print("Output Matrix Pytorch:")
print(torch_output)

# Compare outputs
if np.allclose(custom_output, torch_output, atol=1e-5):
    print("The results match")
else:
    print("The results differ.")
