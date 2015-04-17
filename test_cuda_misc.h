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

#endif