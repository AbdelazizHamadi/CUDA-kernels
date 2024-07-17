from numba import cuda, types, float32, int32
import cupy as cp
import time


# Define block size
BLOCK_SIZE = 32  # max 1024 threads per block

# exact same implementation as C++ with 1D array matrix
# example matrix (2, 2) = [[0, 1], [3, 2]] --> [0, 1, 3, 2]

@cuda.jit
def matmul_shared_memory_1D(A, B, C, X, Y, Z):

    # Define shared memory tiles
    tile_A = cuda.shared.array((BLOCK_SIZE, BLOCK_SIZE), dtype=float32)
    tile_B = cuda.shared.array((BLOCK_SIZE, BLOCK_SIZE), dtype=float32)

    # Calculate thread indices
    tx: int = cuda.threadIdx.x
    ty: int = cuda.threadIdx.y

    # global thread indices
    col = cuda.blockIdx.x * BLOCK_SIZE + tx
    row = cuda.blockIdx.y * BLOCK_SIZE + ty

    tmp = 0.0

    blocks_required = (Y + BLOCK_SIZE - 1) / BLOCK_SIZE

    for i in range(int(blocks_required)):

        if row < X and (i * BLOCK_SIZE + tx) < Y:
            tile_A[ty][tx] = A[row * Y + i * BLOCK_SIZE + tx]
        else:
            tile_A[ty][tx] = 0.0

        if col < Z and (i * BLOCK_SIZE + ty) < Y:
            tile_B[ty][tx] = B[(i * BLOCK_SIZE + ty) * Z + col]
        else:
            tile_B[ty][tx] = 0.0

        cuda.syncthreads()

        for j in range(BLOCK_SIZE):
            tmp += tile_A[ty][j] * tile_B[j][tx]

        cuda.syncthreads()

    if row < X and col < Z:
        C[row * Z + col] = tmp

@cuda.jit
def matmul_shared_memory_2D(A, B, C):

    # Define shared memory tiles
    tile_A = cuda.shared.array((BLOCK_SIZE, BLOCK_SIZE), dtype=float32)
    tile_B = cuda.shared.array((BLOCK_SIZE, BLOCK_SIZE), dtype=float32)

    # print("created tile ")
    # Calculate thread indices
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # global thread indices
    bx = cuda.blockIdx.x * BLOCK_SIZE + tx
    by = cuda.blockIdx.y * BLOCK_SIZE + ty

    block_x = cuda.blockIdx.y
    block_y = cuda.blockIdx.y
    # same as above
    global_x, global_y = cuda.grid(2)


    # Initialize the output value for this thread
    tmp = 0.0
    # Loop over the matrix depending on the tile size : A[0..9] -> tile[0, 3], tile[3, 6] tile[6, 9]
    # how many tile steps there are in the original matrix

    for i in range(0, A.shape[1], BLOCK_SIZE):

        # Load data into shared memory
        if by < A.shape[0] and (i + tx) < A.shape[1]:

            # let's say it's a (4 x 4) matrix and tile is (2 x 2) matrix

            # row is fixed, so we get the global indices (by)
            # columns change each iteration ->
            # each tile in x axis start at - > i : 0, 2, 4 .. n
            # and each beginning of tile we take i + tx : 0, 1, 2...n

            tile_A[ty, tx] = A[by, i + tx]

        else:
            tile_A[ty, tx] = 0.0

        if bx < B.shape[1] and (i + ty) < B.shape[0]:

            # column is fixed, so we get the global indices (bx)
            # rows change each iteration ->
            # each tile in y axis start at - > i : 0, 2, 4 .. n
            # and each beginning of tile we take i + tx : 0, 1, 2...n
            tile_B[ty, tx] = B[i + ty, bx]

        else:
            tile_B[ty, tx] = 0.0

        # Synchronize to make sure the tiles are loaded
        cuda.syncthreads()
        # Compute partial product
        for j in range(BLOCK_SIZE):
            tmp += tile_A[ty, j] * tile_B[j, tx]

        # Synchronize to make sure computation is done before loading new tiles
        cuda.syncthreads()

    # Write the result to the output matrix
    if by < C.shape[0] and bx < C.shape[1]:
        C[by, bx] = tmp


sizes = [2048, 4096, 8192, 12288, 128*128]
times = []

for i in range(len(sizes)):
    print(f"executing cuda Kernel ({i})...")
    X = sizes[i]
    Y = sizes[i]
    Z = sizes[i]
    # Initialize matrices A and B (1D)
    A_device = cp.ones((X * Y)).astype(cp.float32)
    B_device = cp.ones((X * Y)).astype(cp.float32)

    # Allocate memory for the result matrix C
    C_device = cp.zeros((X * Z)).astype(cp.float32)

    # Calculate grid and block dimensions
    threads_per_block = (BLOCK_SIZE, BLOCK_SIZE)
    blocks_per_grid = (int((Z + BLOCK_SIZE - 1) / BLOCK_SIZE), int((X + BLOCK_SIZE - 1) / BLOCK_SIZE))

    print("matrix A shape : ", A_device.reshape(X, Y).shape)
    print("matrix B shape : ", A_device.reshape(Y, Z).shape)
    print("thread size : ", threads_per_block)
    print("blocks per grid : ", blocks_per_grid)

    times_per_size = []
    for i in range(5):
        # Start the timer
        start_time = time.time()
        # Launch the kernel
        matmul_shared_memory_1D[blocks_per_grid, threads_per_block](A_device, B_device, C_device, X, Y, Z)
        cuda.synchronize()
        # Stop the timer
        end_time = time.time()
        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        times_per_size.append(elapsed_time)

    elapsed_time_per_size = times_per_size

    times.append(sum(elapsed_time_per_size) / len(elapsed_time_per_size))
    print(f"({i}) CUDA kernels execution complete in : {sum(elapsed_time_per_size) / len(elapsed_time_per_size)}")


