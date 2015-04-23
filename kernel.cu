#include "test_cuda_misc.h"
#include "test_copy_kernel.h"
#include "test_cyk_kernel.h"
#include "cuda_all_helpers.h"



int main()
{
	try
	{
		test_header("test_cuda_status"), test_cuda_status();
		test_header("test_cuda_copy"), test_copy_kernel();
		test_header("test_cyk_kernel"), test_cyk_kernel();
		test_header("test_cuda_index_calculation"), test_cuda_index_calculation();
	}
	catch (std::runtime_error &error)
	{
		std::cout << error.what() << std::endl;
	}

	std::cout <<
		margin <<
		"TESTS FINISHED!" <<
		margin <<
		std::endl;
	std::cin.ignore();

	return 0;
}