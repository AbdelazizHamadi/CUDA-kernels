#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>
#include <numeric>

#define BLOCK_SIZE 32

// Define ANSI color codes
#define RED "\x1b[31m"
#define RESET "\x1b[0m"

// CUDA kernel for matrix multiplication using shared memory
__global__ void matmul_shared_memory(float *A, float *B, float *C, int X, int Y, int Z) {
    __shared__ float tile_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tile_B[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    float tmp = 0.0f;
    
    /*float previous_temp = 0.0f;*/

    // a block size of (N, N) with matrix A (X, Y) we need Y/N blocks to cover A
    // example : A @ B = C --> (X, Y) @ (Y, Z) = (X, Z)
    // X = 2, Y = 4, Z = 8
    // threadsPerBlock: (2, 2) || blocksPerGrid: (4, 1)
    int blocks_required = (Y + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // iterate over A : if A (4, 2) with block size (2, 2) we need two iterations (i = 0 | 1)
    for (int i = 0; i < blocks_required; ++i) {
        if (row < X && (i * BLOCK_SIZE + tx) < Y) {
            // possible cases 
            // blockIdx.y = 0  blockIdx.x = 0 | 1 | 2 | 3 
            // ty = 0 | 1, tx = 0 | 1 -->  row = 0 | 1, Y = 4, i = 0 | 1, blocksize = 2, tx = 0 | 1
            
            // example : block (2, 0) - thread (1, 0) in second block iteration (i = 1) 
            // for tile_A[0][1] : tx=1, ty=0, i=1, row = 0 * 2 + 0 = 0 --> A[0 * 4 + 1 * 2 + 1] -> Tile_A[0][1] = A[3]
            tile_A[ty][tx] = A[row * Y + i * BLOCK_SIZE + tx];  
        } else {
            tile_A[ty][tx] = 0.0f;
        }

        if (col < Z && (i * BLOCK_SIZE + ty) < Y) {
            // possible cases
            // blockIdx.y = 0  blockIdx.x = 0 | 1 | 2 | 3
            // ty = 0 | 1, tx = 0 | 1 -->  col = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7, Z = 8, i = 1, blocksize = 2, tx = 0 | 1
            // example : block (2, 0) - thread (1, 0) in second block iteration (i = 1) 
            // for tile_B[0][1] : tx=1, ty=0, i=1, col = 2 * 2 + 1 = 5 --> B[(1 * 2 + 0) * 8 + 5] -> Tile_B[0][1] = B[21]

            tile_B[ty][tx] = B[(i * BLOCK_SIZE + ty) * Z + col];
        } else {
            tile_B[ty][tx] = 0.0f;
        }

        /*

        if (blockIdx.x == 2 && blockIdx.y == 0){
            printf(RED "Block(%d, %d) Thread (%d, %d) - Loaded tile_A[%d, %d] = A[%d] = %f | tile_B[%d, %d] = B[%d] = %f \n" RESET, blockIdx.x, blockIdx.y, tx, ty, ty, tx,
        (row * Y + i * BLOCK_SIZE + tx), tile_A[ty][tx], tx, ty, ((i * BLOCK_SIZE + ty) * Z + col), tile_B[tx][ty]);
        }else{
            printf("Block(%d, %d) Thread (%d, %d) - Loaded tile_A[%d, %d] = A[%d] = %f | tile_B[%d, %d] = B[%d] = %f \n", blockIdx.x, blockIdx.y, tx, ty, ty, tx,
        (row * Y + i * BLOCK_SIZE + tx), tile_A[ty][tx], tx, ty, ((i * BLOCK_SIZE + ty) * Z + col), tile_B[tx][ty]);

        }

        */

        
        

        __syncthreads();
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            
            tmp += tile_A[ty][j] * tile_B[j][tx];

            /*
            previous_temp = tmp - tile_A[ty][j] * tile_B[j][tx];

            if (blockIdx.x == 2 && blockIdx.y == 0){
                printf(RED "(iteration=%d) Block(%d, %d) Thread (%d, %d) | tile_A[%d][%d] * tile_B[%d][%d] = %f * %f = %f (+ %f)\n" RESET, i, blockIdx.x, blockIdx.y, tx, ty, ty, j, j, tx, tile_A[ty][j], tile_B[j][tx], tmp, previous_temp); 

            }
            
            */
            
           
            

        }

        __syncthreads();
    }

    if (row < X && col < Z) {
        C[row * Z + col] = tmp;

        
        /*

        if (blockIdx.x == 2 && blockIdx.y == 0){
            printf(RED "C[%d] = %f \n" RESET, (row * Z + col) ,  C[row * Z + col]);
        }else{

            printf("C[%d] = %f \n", (row * Z + col) ,  C[row * Z + col]);
        }
        
        */       
        
       
    }
}

extern "C" void MatMul(float *A, float *B, float *C, int X, int Y, int Z, float *time, bool print) {

    float *d_A, *d_B, *d_C;

    cudaMalloc((void **)&d_A, X * Y * sizeof(float));
    cudaMalloc((void **)&d_B, Y * Z * sizeof(float));
    cudaMalloc((void **)&d_C, X * Z * sizeof(float));

    cudaMemcpy(d_A, A, X * Y * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, Y * Z * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((Z + BLOCK_SIZE - 1) / BLOCK_SIZE, (X + BLOCK_SIZE - 1) / BLOCK_SIZE);

    if (print == true){
        std::cout << "threadsPerBlock: (" << threadsPerBlock.x << ", " << threadsPerBlock.y << ")" << std::endl;
        std::cout << "blocksPerGrid: (" << blocksPerGrid.x << ", " << blocksPerGrid.y << ")" << std::endl;
    }

    
    // Get the start time
    auto start = std::chrono::high_resolution_clock::now();
    // Call the CUDA function
    matmul_shared_memory<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, X, Y, Z);
    // Get the end time
    auto end = std::chrono::high_resolution_clock::now();
    // Calculate the duration
    std::chrono::duration<double> duration = end - start;
    *time  = duration.count();
    // std::cout << "CUDA kernel execution complete in : " << duration.count() << "s" << std::endl;


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
