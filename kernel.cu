
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <vector>

#include "cuda_helper.h"

#include "cyk_table.h"
#include "cyk_rules_table.h"

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void cykKernel(int *c, cyk_table<4, 10>* table)
{
	int i = threadIdx.x;
	c[i] = table->size() + table->max_num_of_symbols();
}

int test_cyk_second_row_filling()
{
	return 1;
}

int test_cuda_basic()
{
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	std::cout << deviceProp.warpSize << std::endl;

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

int main()
{
	auto result = 
		test_cuda_basic() ||
		test_cyk_second_row_filling();

	std::cout << "enter to exit" << std::endl;
	std::cin.ignore();

	return result;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

	cyk_table<4, 10>* dev_table = 0;

	cyk_table<4, 10> table;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

	device_malloc<cyk_table<4, 10>>(&dev_table, AT);
	device_memcpy<cyk_table<4, 10>>(dev_table, &table, AT);

	device_malloc<int>(&dev_a, AT, size * sizeof(int));
	device_malloc<int>(&dev_b, AT, size * sizeof(int));
	device_malloc<int>(&dev_c, AT, size * sizeof(int));


    // Copy input vectors from host memory to GPU buffers.

	device_memcpy<int>(dev_a, a, AT, size * sizeof(int));
	device_memcpy<int>(dev_b, b, AT, size * sizeof(int));

	cykKernel << < 1, size >> >(dev_c, dev_table);

	//INNER_LAUNCH_KERNEL(cykKernel, 1, size, dev_c, dev_table);

    // Check for any errors launching the kernel
	check_for_errors_after_launch(AT);
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
	device_synchronize(AT);

    // Copy output vector from GPU buffer to host memory.
	host_memcpy(c, dev_c, AT, size * sizeof(int));

	std::cout << c[0] << std::endl;

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
