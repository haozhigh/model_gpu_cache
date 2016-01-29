#include <stdio.h>
#include <cuda_runtime.h>

#define REP1(x)       x
#define REP2(x)       REP1(x)       REP1(x)      
#define REP4(x)       REP2(x)       REP2(x)      
#define REP8(x)       REP4(x)       REP4(x)      
#define REP16(x)      REP8(x)       REP8(x)      
#define REP32(x)      REP16(x)      REP16(x)     
#define REP64(x)      REP32(x)      REP32(x)     
#define REP128(x)     REP64(x)      REP64(x)     
#define REP256(x)     REP128(x)     REP128(x)    
#define REP512(x)     REP256(x)     REP256(x)    
#define REP1024(x)    REP512(x)     REP512(x)    
#define REP2048(x)    REP1024(x)    REP1024(x)   
#define REP4096(x)    REP2048(x)    REP2048(x)   
#define REP8192(x)    REP4096(x)    REP4096(x)   
#define REP16384(x)   REP8192(x)    REP8192(x)   
#define REP32768(x)   REP16384(x)   REP16384(x)

//#define ARRAY_SIZE 17000
//#define STRIDE 11

#define DATA_TYPE long long

__global__ void read_cache(DATA_TYPE* device_array) {
    DATA_TYPE* j = &device_array[0];
    REP4096(j=*(DATA_TYPE**)j;)
    device_array[0] = (DATA_TYPE)j;
}

int main(int argc, char* argv[]) {
    cudaError_t err = cudaSuccess;
    DATA_TYPE* host_array = NULL;
    DATA_TYPE* device_array = NULL;
    size_t size;
    int i;

    if (argc < 3) {
        printf("Not enough parameters! Exitting...\n");
        return -1;
    }
    int ARRAY_SIZE = atoi(argv[1]);
    int STRIDE = atoi(argv[2]);

    size = sizeof(DATA_TYPE) * ARRAY_SIZE;
    host_array = (DATA_TYPE*)malloc(size);
    if (host_array == NULL) {
        printf("Failed to malloc!\n");
        return -1;
    }

    err = cudaMalloc((void**)&device_array, size);
    if (err != cudaSuccess) {
        printf("Failed to cudaMalloc!\n");
        free(host_array);
        return -1;
    }

    for (i = 0; i < ARRAY_SIZE; i++) {
        DATA_TYPE t = i + STRIDE;
        if (t >= ARRAY_SIZE) t %= STRIDE;
        host_array[i] = (DATA_TYPE)device_array + (DATA_TYPE)sizeof(DATA_TYPE) * t;
    }

    err = cudaMemcpy(device_array, host_array, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Failed to cudaMemcpy!\n");
        free(host_array);
        cudaFree(device_array);
        return -1;
    }

    read_cache<<<1, 1>>>(device_array);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Failed to invoke kernel!\n");
        free(host_array);
        cudaFree(device_array);
        return -1;
    }

    free(host_array);
    cudaFree(device_array);
    return 0;
}
