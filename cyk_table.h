#include "cuda_all_helpers.h"

#include "preferences.h"

#if !defined(CYK_TABLE_H)
#define CYK_TABLE_H

class cyk_table
{
public:
	enum info : int
	{
		replicated = 1,
		not_enough_space = 2
	};

	CCM cyk_table(const int block_id_, const char* source_code_localization_, int* prefs_, int* table_, int* table_header_);
	CCM ~cyk_table();

	CCM int size() const;
	CCM int depth() const;

	CCM int get_number_of_symbols_in_cell(int row, int col);
	CCM int get_symbol_at(int row, int col, int pos);

	CCM int add_symbol_to_cell(int row, int col, int symbol);

	CCM int get_row_coord_for_thread(int thread_id, int current_row, int current_col);
	CCM int get_starting_col_coord_for_thread(int thread_id);
	CCM int get_coord_col_for_thread(int thread_id, int current_row, int current_col);

private:
	const int block_id;
	const int number_of_blocks;
	const char* source_code_localization;
	const int size_;
	const int depth_;

	int* prefs;
	int* table;
	int* table_header;

	CCM int set_symbol_at(int row, int col, int pos, int val);
};


CCM cyk_table::cyk_table(const int block_id_, const char* source_code_localization_, int* prefs_, int* table_, int* table_header_) :
block_id(block_id_),
number_of_blocks(preferences(block_id_, source_code_localization_).get(prefs_, preferences::number_of_blocks)),
source_code_localization(source_code_localization_),
prefs(prefs_),
table(table_),
table_header(table_header_),
size_(preferences(block_id_, source_code_localization_).get(prefs_, preferences::sentence_length)),
depth_(preferences(block_id_, source_code_localization_).get(prefs_, preferences::max_symbols_in_cell))
{
}

CCM cyk_table::~cyk_table()
{
}

CCM int cyk_table::size() const
{
	return size_;
}

CCM int cyk_table::depth() const
{
	return depth_;
}

CCM int cyk_table::get_number_of_symbols_in_cell(int row, int col)
{
	return table_get(table_header, generate_absolute_index(
		block_id, number_of_blocks, 
		row, size(), 
		col, size()));
}

CCM int cyk_table::get_symbol_at(int row, int col, int pos)
{
	return table_get(table, generate_absolute_index(
		block_id, number_of_blocks,
		row, size(),
		col, size(),
		pos, get_number_of_symbols_in_cell(row, col)));
}

CCM int cyk_table::set_symbol_at(int row, int col, int pos, int val)
{
	//log_debug("block_id=%d, row=%d, col=%d, pos=%d, val=%d\n", block_id, row, col, pos, val);
	return table_set(table, generate_absolute_index(
		block_id, number_of_blocks,
		row, size(),
		col, size(),
		pos, depth()), val);
}

CCM int cyk_table::add_symbol_to_cell(int row, int col, int symbol)
{
	int symbols_in_cell = get_number_of_symbols_in_cell(row, col);
	for (int i = 0; i < symbols_in_cell; ++i)
	{
		if (get_symbol_at(row, col, i) == symbol)
		{
			return info::replicated;
		}
	}

	if (symbols_in_cell >= depth())
	{
		return info::not_enough_space;
	}

	int result = set_symbol_at(row, col, symbols_in_cell, symbol);

	if (result == error::no_errors_occured)
	{
		result = table_set(table_header, generate_absolute_index(
			block_id, number_of_blocks,
			row, size(),
			col, size()), symbols_in_cell + 1);

		return result;
	}

	return result;
}

CCM int cyk_table::get_row_coord_for_thread(int thread_id, int current_row, int current_col)
{
	int number_of_threads = preferences(block_id, AT).get(prefs, preferences::number_of_threads);
	int margin = size() - current_row;
	int active_threads = min(number_of_threads, margin);

	if (thread_id >= active_threads)
	{
		return error::index_out_of_bounds;
	}

	return current_row + (current_col + active_threads) / margin;
}

CCM int cyk_table::get_starting_col_coord_for_thread(int thread_id)
{
	return thread_id < size() ? thread_id : error::index_out_of_bounds;
}

CCM int cyk_table::get_coord_col_for_thread(int thread_id, int current_row, int current_col)
{
	int number_of_threads = preferences(block_id, AT).get(prefs, preferences::number_of_threads);
	int margin = size() - current_row;
	int active_threads = min(number_of_threads, margin);

	//log_debug("thread_id=%d, current_row=%d, current_col=%d, number_of_threads=%d, size()=%d, margin=%d, active_threads=%d occured!\n",
	//	thread_id, current_row, current_col, number_of_threads, size(), margin, active_threads);

	if (thread_id >= active_threads)
	{
		return error::index_out_of_bounds;
	}

	return (current_col + active_threads) % margin;
}

#endif