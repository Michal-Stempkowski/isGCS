#include <cassert>

#include "cuda_helper.h"
#include "preferences.h"
#include "cyk_table.h"

#if !defined(TEST_COPY_KERNEL_H)
#define TEST_COPY_KERNEL_H

#define sentence_length 16
#define max_alphabet_size 32
#define max_symbols_in_cell 16
#define num_of_blocks 3
#define num_of_threads 32

#define prefs_size (num_of_blocks * preferences::enum_size)
#define sentence_block_size  (num_of_blocks * sentence_length)
#define cyk_block_header_size (num_of_blocks * sentence_length * sentence_length)
#define cyk_block_size (cyk_block_header_size * max_symbols_in_cell)

__global__ void copy_kernel(int* prefs, int* sentence, int* table, int* table_header)
{
	const int block_id = blockIdx.x;
	const int thread_id = threadIdx.x;

	int mine_id = block_id * num_of_threads + thread_id;

	if (mine_id < prefs_size)
	{
		prefs[mine_id] += thread_id;
	}

	if (mine_id < sentence_block_size)
	{
		sentence[mine_id] += thread_id;
	}

}

void test_copy_kernel()
{
	int prefs[prefs_size] =
	{
		sentence_length, max_alphabet_size, max_symbols_in_cell, num_of_blocks, num_of_threads,
		sentence_length, max_alphabet_size, max_symbols_in_cell, num_of_blocks, num_of_threads,
		sentence_length, max_alphabet_size, max_symbols_in_cell, num_of_blocks, num_of_threads
	};
	int * dev_prefs = nullptr;
	int expected_dest_prefs[prefs_size] =
	{
		sentence_length,      max_alphabet_size + 1,  max_symbols_in_cell + 2,  num_of_blocks + 3,  num_of_threads + 4,
		sentence_length + 5,  max_alphabet_size + 6,  max_symbols_in_cell + 7,  num_of_blocks + 8,  num_of_threads + 9,
		sentence_length + 10, max_alphabet_size + 11, max_symbols_in_cell + 12, num_of_blocks + 13, num_of_threads + 14
	};

	int sentence[sentence_block_size] =
	{
		1, 2, 2, 2, 2, 1, 2, 2, 1, 3, 4, 5, 1, 3, 4, 5,
		1, 2, 2, 2, 2, 1, 2, 2, 1, 3, 4, 5, 1, 3, 4, 5,
		1, 2, 2, 2, 2, 1, 2, 2, 1, 3, 4, 5, 1, 3, 4, 5
	};
	int *dev_sentence = nullptr;
	int expected_dest_sentence[sentence_block_size] =
	{
		1,       2 + 1,   2 + 2,   2 + 3,   2 + 4,   1 + 5,   2 + 6,   2 + 7,   1 + 8,   3 + 9,   4 + 10,  5 + 11,  1 + 12,  3 + 13,  4 + 14,  5 + 15,
		1 + 16,  2 + 17,  2 + 18,  2 + 19,  2 + 20,  1 + 21,  2 + 22,  2 + 23,  1 + 24,  3 + 25,  4 + 26,  5 + 27,  1 + 28,  3 + 29,  4 + 30,  5 + 31,
		1,       2 + 1,   2 + 2,   2 + 3,   2 + 4,   1 + 5,   2 + 6,   2 + 7,   1 + 8,   3 + 9,   4 + 10,  5 + 11,  1 + 12,  3 + 13,  4 + 14,  5 + 15
	};

	int cyk_table_header[cyk_block_header_size];
	int *dev_cyk_table_header = nullptr;

	int cyk[cyk_block_size];
	int *dev_cyk = nullptr;


	try
	{
		cuda_helper(AT).from_host_to_device(&dev_prefs, prefs, prefs_size);
		cuda_helper(AT).from_host_to_device(&dev_sentence, sentence, sentence_block_size);
		cuda_helper(AT).from_host_to_device(&dev_cyk, cyk, cyk_block_size);
		cuda_helper(AT).from_host_to_device(&dev_cyk_table_header, cyk_table_header, cyk_block_header_size);

		copy_kernel << <num_of_blocks, num_of_threads >> > (
			dev_prefs, dev_sentence, dev_cyk, dev_cyk_table_header);

		cuda_helper(AT).check_for_errors_after_launch();
		cuda_helper(AT).device_synchronize();

		cuda_helper(AT).from_device_to_host(prefs, dev_prefs, prefs_size);

		cuda_helper(AT).from_device_to_host(sentence, dev_sentence, sentence_block_size);
		cuda_helper(AT).from_device_to_host(cyk_table_header, dev_cyk_table_header, cyk_block_header_size);
		cuda_helper(AT).from_device_to_host(cyk, dev_cyk, cyk_block_size);
	}
	catch (std::runtime_error &error)
	{
		std::cout << error.what() << std::endl;
	}

	cuda_helper(AT).free(dev_prefs);
	cuda_helper(AT).free(dev_sentence);
	cuda_helper(AT).free(dev_cyk);

	expect_table_eq(prefs, expected_dest_prefs, prefs_size, AT);
	expect_table_eq(sentence, expected_dest_sentence, sentence_block_size, AT);

	//expect_eq(true, false, AT);
}

#undef sentence_length
#undef max_alphabet_size
#undef max_symbols_in_cell
#undef num_of_blocks
#undef num_of_threads

#undef prefs_size
#undef sentence_block_size 
#undef cyk_block_header_size
#undef cyk_block_size

#endif