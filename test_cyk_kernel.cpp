#include "test_cyk_kernel.h"
#include "preferences.h"

__global__ void cyk_kernel(int* prefs, int* sentence, int* cyk_table)
{

}

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

	const int sentence_block_size = num_of_blocks * sentence_length;
	int sentence[sentence_block_size] = 
	{
		1, 2, 2, 2, 2, 1, 2, 2, 1, 3, 4, 5, 1, 3, 4, 5,
		1, 2, 2, 2, 2, 1, 2, 2, 1, 3, 4, 5, 1, 3, 4, 5,
		1, 2, 2, 2, 2, 1, 2, 2, 1, 3, 4, 5, 1, 3, 4, 5
	};
	int *dev_sentence = nullptr;

	const int cyk_block_size = num_of_blocks * sentence_length * sentence_length * max_symbols_in_cell;
	int cyk_table[cyk_block_size];
	int *dev_cyk_table = nullptr;
	

	try
	{
		cuda_helper(AT).copy_to(&dev_prefs, prefs, prefs_size);
		cuda_helper(AT).copy_to(&dev_sentence, sentence, sentence_block_size);
		cuda_helper(AT).copy_to(&dev_cyk_table, cyk_table, cyk_block_size);
	}
	catch (std::runtime_error &error)
	{
		std::cout << error.what() << std::endl;
	}

	cuda_helper(AT).free(dev_prefs);
	cuda_helper(AT).free(dev_sentence);
	cuda_helper(AT).free(dev_cyk_table);

	//expect_eq(true, false, AT);
}