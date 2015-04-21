#include <cassert>

#include "cuda_helper.h"

#if !defined(TEST_CUDA_MISC_H)
#define TEST_CUDA_MISC_H

void test_cuda_status()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	//std::cout << deviceProp.major << "major" << std::endl <<
	//	deviceProp.minor << "minor" << std::endl <<
	//	deviceProp.sharedMemPerBlock << "sharedMemPerBlock" << std::endl <<
	//	deviceProp.warpSize << "warpSize" << std::endl;

	expect_eq(true, true, AT);
}

void test_cuda_index_calculation()
{
	expect_eq(index_out_of_bounds, generate_absolute_index(0, 0), AT);
	expect_eq(5, generate_absolute_index(5, 8), AT);
	expect_eq(4, generate_absolute_index(0, 2, 4, 5), AT);
	expect_eq(14, generate_absolute_index(2, 7, 4, 5), AT);
	expect_eq(15, generate_absolute_index(
		1, 2, 
		1, 2, 
		1, 2, 
		1, 2), AT);
	expect_eq(index_out_of_bounds, generate_absolute_index(
		2, 0,
		2, 2,
		2, 2,
		2, 2), AT);
}

#endif