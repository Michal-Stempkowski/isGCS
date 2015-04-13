
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <vector>

#include <cassert>

#include "cuda_helper.h"

#include "cyk_table.h"
#include "cyk_rules_table.h"
#include "device_memory.h"

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

template <int sentence_length, int max_symbol_length>
__global__ void cyk2Kernel(cyk_table<sentence_length, max_symbol_length>* table)
{
	int index = threadIdx.x;

	/*if (index < sentence_length)
	{
		table->fill_cell(index, index, )
	}*/
}

int test_cyk_fill_cell()
{
	std::cout << "test_cyk_fill_cell" << std::endl;

	const int max_symbol_count = 5;

	cyk_table<4, max_symbol_count> cyk;

	const auto NM = constants::NO_MATCHING_RULE;

	int rules_table[max_symbol_count][max_symbol_count] =
	{
		//	0	1	2	3	4
		{
			NM,	NM,	NM,	NM,	NM	//0
		},
		{
			NM, NM, 3, NM, NM	//1
		},
		{
			NM, NM,	NM, NM, NM	//2
		},
		{
			NM, NM, NM, NM, 0	//3
		},
		{
			NM, NM, NM,	NM,	NM	//4
		}
	};

	cyk_rules_table<max_symbol_count> rules(rules_table);

	const int sentence_length = 4;

	int sentence[sentence_length] = { 1, 2, 3, 4 };
	cyk.fill_first_row(sentence);

	for (int i = 0; i < sentence_length - 1; ++i)
	{
		cyk.fill_cell(1, i, rules);
	}

	expect_eq(cyk.get_cell_rule(0, 0, 0), 1);
	expect_eq(cyk.get_cell_rule(0, 1, 0), 2);
	expect_eq(cyk.get_cell_rule(1, 0, 0), 3);
	expect_eq(cyk.get_cell_rule(1, 2, 0), 0);

	return 0;
}

int test_cuda_status()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	std::cout << deviceProp.major << "major" << std::endl <<
		deviceProp.minor << "minor" << std::endl <<
		deviceProp.sharedMemPerBlock << "sharedMemPerBlock" << std::endl <<
		deviceProp.warpSize << "warpSize" << std::endl;

	return 1;
}

int test_cyk_second_row_filling()
{
	return 0;
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
		test_cyk_fill_cell() ||
		test_cyk_second_row_filling() ||
		test_cuda_status();

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

	cyk_table<4, 10>* dev_table = 0;

	cyk_table<4, 10> table;

    // Choose which GPU to run on, change this on a multi-GPU system.
    /*cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }*/

	try
	{
		cuda_helper.copy_to<cyk_table<4, 10>>(&dev_table, &table);

		cuda_helper.copy_to<int>(&dev_a, a, size * sizeof(int));
		cuda_helper.copy_to<int>(&dev_b, b, size * sizeof(int));

		cuda_helper.device_malloc<int>(&dev_c, size * sizeof(int));

		cykKernel << < 1, size >> >(dev_c, dev_table);

		cuda_helper.check_for_errors_after_launch();

		cuda_helper.device_synchronize();

		cuda_helper.copy_from(c, dev_c, size * sizeof(int));

		std::cout << c[0] << std::endl;
	}
	catch (std::runtime_error &error)
	{
		std::cout << error.what() << std::endl;
	}
	
	cuda_helper.free(dev_a);
	cuda_helper.free(dev_a);
	cuda_helper.free(dev_b);
	cuda_helper.free(dev_table);
    
	return cudaSuccess;
}
