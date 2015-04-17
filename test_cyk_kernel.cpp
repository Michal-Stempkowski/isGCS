#include "test_cyk_kernel.h"
#include "preferences.h"

//__global__ void cyk_kernel(int* prefs)
//{
//
//}

void test_cyk_kernel()
{
	const int sentence_length = 16;
	const int max_alphabet_size = 32;
	const int max_symbols_in_cell = 16;
	const int num_of_blocks = 3;
	const int num_of_threads = 32;

	const int prefs_size = num_of_blocks * preferences::enum_size;
	int prefs[prefs_size] =
	{
		sentence_length, max_alphabet_size, max_symbols_in_cell, num_of_blocks, num_of_threads,
		sentence_length, max_alphabet_size, max_symbols_in_cell, num_of_blocks, num_of_threads,
		sentence_length, max_alphabet_size, max_symbols_in_cell, num_of_blocks, num_of_threads
	};
	int * dev_prefs = nullptr;

	

	try
	{
		cuda_helper(AT).copy_to(&dev_prefs, prefs, prefs_size);
	}
	catch (std::runtime_error &error)
	{
		std::cout << error.what() << std::endl;
	}

	cuda_helper(AT).free(dev_prefs);

	expect_eq(true, false, AT);
}