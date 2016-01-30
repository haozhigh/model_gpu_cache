#include <stdio.h>
#include <cuda_runtime.h>

#define DATA_TYPE int
#define ARRAY_SIZE 32

__global__ void copy_kernel(DATA_TYPE *d_a, DATA_TYPE *d_b) {
	int i;

	i = threadIdx.x;
	d_b[i] = d_a[i];
}

int main(int argc, char* argv[]) {
    cudaError_t err = cudaSuccess;
    DATA_TYPE* h_a = NULL;
    DATA_TYPE* h_b = NULL;
    DATA_TYPE* d_a = NULL;
    DATA_TYPE* d_b = NULL;

    size_t size;
    int i;

	//  Calculate array memory in bytes
    size = sizeof(DATA_TYPE) * ARRAY_SIZE;

	//  Assign memory for h_a, h_b on host side
    h_a = (DATA_TYPE*)malloc(size);
    if (h_a == NULL) {
        printf("Failed to malloc h_a!\n");
        return -1;
    }
    h_b = (DATA_TYPE*)malloc(size);
    if (h_b == NULL) {
        printf("Failed to malloc h_b!\n");
		free(h_a);
        return -1;
    }

	//  Assign memory for d_a, d_b on device side
    err = cudaMalloc((void**)&d_a, size);
    if (err != cudaSuccess) {
        printf("Failed to cudaMalloc d_a!\n");
		free(h_a);
		free(h_b);
        return -1;
    }
    err = cudaMalloc((void**)&d_b, size);
    if (err != cudaSuccess) {
        printf("Failed to cudaMalloc d_b!\n");
		free(h_a);
		free(h_b);
		cudaFree(d_a);
        return -1;
    }

	//  Init h_a values
    for (i = 0; i < ARRAY_SIZE; i++) {
		h_a[i] = i;
	}

	//  Copy h_a to d_a
    err = cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Failed to cudaMemcpy from h_a to d_a!\n");
		free(h_a);
		free(h_b);
		cudaFree(d_a);
		cudaFree(d_b);
        return -1;
    }

	//  Call the kernel
    copy_kernel<<<1, ARRAY_SIZE>>>(d_a, d_b);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Failed to invoke kernel!\n");
		free(h_a);
		free(h_b);
		cudaFree(d_a);
		cudaFree(d_b);
        return -1;
    }


	//  Release memory
	free(h_a);
	free(h_b);
	cudaFree(d_a);
	cudaFree(d_b);
    return 0;
}
