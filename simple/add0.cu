//: nvcc add0.cu -o add0

#include <stdlib.h>
#include <stdio.h>

__global__ void cuda_add(int a, int b, int *c)
{
    *c = a + b;
}

int main(int argc, char **argv)
{
    int c;
    int *dev_c;
    cudaMalloc((void**)&dev_c, sizeof(int));
    cuda_add<<<1,1>>>(2, 2, dev_c);
    /*
     * Arguments pour cudaMemcpy
     * 1 : destination
     * 2 : memoire sur le device
     * 3 : taille du bloc
     * 4 : direction de la copie
     */
    cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Almighty CUDA's answer: 2 + 2 = %d.\n", c);
    cudaFree(dev_c);
    return EXIT_SUCCESS;
}

