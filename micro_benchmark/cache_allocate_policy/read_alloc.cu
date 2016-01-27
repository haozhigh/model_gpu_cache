#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>

#define N 1024*1024              //array size

__global__ void read_alloc_kernel1(int *A, int *B, int *time){
    int x1, x2, x3, x4, x5, x6, x7, x8, x9;
    int t0, t1, t2, t3, t4, t5;

    t0 = clock();
    x1 = A[64];
    x2 = A[1088];
    x3 = A[2144];
    x4 = A[3168];
    x1 ++;
    x2 ++;
    x3 ++;
    x4 ++;
    t1 = clock();

    t2 = clock();
    x5 = A[4096];
    t3 = clock();
    x7 = A[1088];
    x8 = A[2144];
    x9 = A[3168];
    t4 = clock();
    x6 = A[64];
    t5 = clock();

    B[0] = x1;
    B[1] = x2;
    B[2] = x3;
    B[3] = x4;
    B[4] = x5;
    B[5] = x6;
    B[6] = x7;
    B[7] = x8;
    B[8] = x9;

    time[0] = t1 - t0;
    time[1] = t3 - t2;
    time[2] = t5 - t4;
}

int main(int argc, char **argv) {
    int *A, *B, *A_gpu, *B_gpu;
    int *time, *time_gpu;
    int i;

    A = (int *)malloc(sizeof(int) * N);
    B = (int *)malloc(sizeof(int) * N);
    time = (int *)malloc(sizeof(int) * N);
    cudaMalloc((void **)&A_gpu, sizeof(int) * N);
    cudaMalloc((void **)&B_gpu, sizeof(int) * N);
    cudaMalloc((void **)&time_gpu, sizeof(int) * N);

    for (i = 0; i < N; i++)
        B[i] = 0;
    cudaMemcpy(B_gpu, B, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaThreadSynchronize();

    dim3 block(1);
    dim3 grid(1);
    read_alloc_kernel1<<< grid, block >>>(A_gpu, B_gpu, time_gpu);
    cudaThreadSynchronize();

    cudaMemcpy(time, time_gpu, sizeof(int) * N, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    free(A);
    free(B);
    free(time);
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(time_gpu);

    return 0;
}
