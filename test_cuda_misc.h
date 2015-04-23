#include <cassert>

#include "cuda_all_helpers.h"

#if !defined(TEST_CUDA_MISC_H)
#define TEST_CUDA_MISC_H

const char margin[] = "\n-----------------------------------\n";
const char margin_lined[] = "\n===================================\n";
const char error[] = "+++++++++++++++++++++++++++++++++++";

void test_header(const char* name)
{
	std::cout << "+++" << name << std::endl;
}

template<class A, class B>
void expect_eq(A a, B b, const char *at)
{
	if (a != b)
	{
		std::cout << std::endl << error << std::endl <<
			"EXPECTATION FAILURE!" << std::endl <<
			"left = " << a << ";" << std::endl <<
			"right = " << b << ";" << std::endl <<
			"left != right." << std::endl <<
			"at: " << at << std::endl <<
			error << std::endl << std::endl;
	}
}

template<class A, class B>
void expect_table_eq(A a, B b, int size, const char *at)
{
	for (int i = 0; i < size; ++i)
	{
		if (a[i] != b[i])
		{
			std::cout << std::endl << error << std::endl <<
				"EXPECTATION FAILURE!" << std::endl <<
				"tables differ at: " << i << std::endl <<
				"left = " << a[i] << ";" << std::endl <<
				"right = " << b[i] << ";" << std::endl <<
				"left != right." << std::endl <<
				"at: " << at << std::endl <<
				error << std::endl << std::endl;
		}
	}
}

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