#include <cassert>

#include "cuda_all_helpers.h"
#include "preferences.h"
#include "cyk_table.h"

#if !defined(TEST_CYK_KERNEL_H)
#define TEST_CYK_KERNEL_H

__global__ void cyk_kernel(int* prefs, int* sentence, int* table, int* table_header)
{
	const int block_id = blockIdx.x;
	const int number_of_blocks = preferences(block_id, AT).get(prefs, preferences::number_of_blocks);
	const int thread_id = threadIdx.x;
	const int number_of_threads = preferences(block_id, AT).get(prefs, preferences::number_of_threads);

	cyk_table cyk(block_id, AT, prefs, table, table_header);

	int row = 0;
	int col = cyk.get_starting_col_coord_for_thread(thread_id);

	for (int i = 0; i < 1/*cyk.size()*/; ++i)
	{
		//log_debug("block_id=%d, thread_id=%d, i=%d, row=%d, col=%d occured!\n",
		//			block_id, thread_id, i, row, col);
		if (row < 0 || col < 0)
		{

		}
		else if (row == 0)
		{
			auto symbol = table_get(sentence, generate_absolute_index(
				block_id, number_of_blocks,
				col, cyk.size()));

			auto result = cyk.add_symbol_to_cell(row, col, symbol);

			//log_debug("result=%d\n", result);

			//log_debug("block_id=%d, thread_id=%d, i=%d, row=%d, col=%d, symbol=%d occured!\n",
			//	block_id, thread_id, i, row, col, symbol);
		}

		
	//	if (row < 0 || col < 0)
	//	{
	//		
	//	}
	//	else if (row == 0)
	//	{
	//		auto symbol = table_get(sentence, generate_absolute_index(block_id, number_of_blocks, col, cyk.size()));
	//		//log_debug("block_id=%d, thread_id=%d, i=%d, row=%d, col=%d, symbol=%d, error=%d occured!\n",
	//		//	block_id, thread_id, i, row, col, symbol, 0);
	//		//auto result = cyk.add_symbol_to_cell(row, col, symbol);

	////		if (symbol < error::no_errors_occured || result != error::no_errors_occured)
	////		{
	////			log_debug("i=%d, row=%d,, col=%d, symbol=%d, error=%d occured!\n", i, row, col, sentence[col], result);
	////		}
	//	}
	////	else
	////	{

	////	}
	}
	
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

	const int cyk_block_header_size = num_of_blocks * sentence_length * sentence_length;
	int cyk_table_header[cyk_block_header_size] = { 0 };
	int *dev_cyk_table_header = nullptr;

	const int cyk_block_size = cyk_block_header_size * max_symbols_in_cell;
	int cyk_tab[cyk_block_size];
	int *dev_cyk_tab = nullptr;


	try
	{
		cuda_helper(AT).from_host_to_device(&dev_prefs, prefs, prefs_size);
		cuda_helper(AT).from_host_to_device(&dev_sentence, sentence, sentence_block_size);
		cuda_helper(AT).from_host_to_device(&dev_cyk_tab, cyk_tab, cyk_block_size);
		cuda_helper(AT).from_host_to_device(&dev_cyk_table_header, cyk_table_header, cyk_block_header_size);

		cyk_kernel <<<num_of_blocks, num_of_threads >>> (
			dev_prefs, dev_sentence, dev_cyk_tab, dev_cyk_table_header);

		cuda_helper(AT).check_for_errors_after_launch();
		cuda_helper(AT).device_synchronize();

		cuda_helper(AT).from_device_to_host(cyk_table_header, dev_cyk_table_header, cyk_block_header_size);
		cuda_helper(AT).from_device_to_host(cyk_tab, dev_cyk_tab, cyk_block_size);
	}
	catch (std::runtime_error &error)
	{
		std::cout << error.what() << std::endl;
	}

	cuda_helper(AT).free(dev_prefs);
	cuda_helper(AT).free(dev_sentence);
	cuda_helper(AT).free(dev_cyk_tab);

	for (int i = 0; i < sentence_length; ++i)
	{
		expect_eq(
			table_get(cyk_tab, generate_absolute_index(
				0, num_of_blocks,
				0, sentence_length,
				i, sentence_length,
				0, max_symbols_in_cell)),
			table_get(sentence, generate_absolute_index(
				0, num_of_blocks,
				i, sentence_length)),
			AT);
	}
	//expect_eq(true, false, AT);
}

#endif