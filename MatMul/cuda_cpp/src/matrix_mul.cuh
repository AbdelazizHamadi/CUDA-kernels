#ifndef MATRIX_MUL_CUH
#define MATRIX_MUL_CUH

extern "C" void MatMul(float *A, float *B, float *C, int X, int Y, int Z, float *time, bool print);

#endif // CUDA_CUH
