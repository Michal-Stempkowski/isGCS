#include "cyk_table.h"

CCM cyk_table::cyk_table(const int block_id_, const char* source_code_localization_, int* prefs_, int* table_, int* table_header_) :
block_id(block_id_), 
source_code_localization(source_code_localization_), 
prefs(prefs_), 
table(table_),
table_header(table_header_),
size_(preferences(block_id, source_code_localization).get(prefs, preferences::sentence_length)),
depth_(preferences(block_id, source_code_localization).get(prefs, preferences::max_symbols_in_cell))
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
	return table_get(table_header, generate_absolute_index(row, size(), col, size()));
}

CCM int cyk_table::get_symbol_at(int row, int col, int pos)
{
	return table_get(table, generate_absolute_index(
		row, size(), 
		col, size(), 
		pos, get_number_of_symbols_in_cell(row, col)));
}

CCM int cyk_table::set_symbol_at(int row, int col, int pos, int val)
{
	return table_set(table, generate_absolute_index(
		row, size(),
		col, size(),
		pos, get_number_of_symbols_in_cell(row, col)), val);
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
			row, size(),
			col, size()), symbols_in_cell + 1);
		
		return result;
	}

	return result;
}