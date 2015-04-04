#include "cuda_helper.h"

#ifndef CYK_TABLE_H
#define CYK_TABLE_H

template <int sentence_length, int max_symbol_length>
class cyk_table
{
public:
	CCM cyk_table();
	CCM ~cyk_table();

	CCM int size() const;
	CCM int* first_symbol(int row, int col);
	CCM int* last_symbol(int row, int col);

	CCM int max_num_of_symbols() const;

	CCM void fill_cell(int row, int col, int** rules_table);

private:
	CCM void assign_rules_for_two_cell_combination(int offset, int current_row, int current_col, int** rules_table);
	CCM void assign_rule_if_possible(int** rules_table, int left_symbol, int right_symbol, int current_row, int current_col);
	CCM void assign_rule(int row, int col, int rule);

	enum special_field : int
	{
		symbol_count,
		enum_size	// DO NOT use it as enum, it represents size of this enum
	};

	int table[sentence_length][sentence_length][max_symbol_length + special_field::enum_size];
};

template <int sentence_length, int max_symbol_length>
CCM void cyk_table<sentence_length, max_symbol_length>::
assign_rule(int row, int col, int rule)
{
	*last_symbol(row, col) = rule;
	++table[row][col][special_field::symbol_count];
}

template <int sentence_length, int max_symbol_length>
CCM void cyk_table<sentence_length, max_symbol_length>::
assign_rule_if_possible(int** rules_table, int left_symbol, int right_symbol, int current_row, int current_col)
{
	auto rule = rules_table[left_symbol][rightSymbol];

	if (rule != constants::NO_MATCHING_RULE)
	{
		assign_rule(current_row, current_col, rule);
	}
}

template <int sentence_length, int max_symbol_length>
CCM void cyk_table<sentence_length, max_symbol_length>::
assign_rules_for_two_cell_combination(int offset, int current_row, int current_col, int** rules_table)
{
	for (
		int* left_symbol = first_symbol(offset, current_col);
		left_symbol != last_symbol(offset, current_col);
	++left_symbol)
	{
		int right_symbol_row = current_row - (offset + 1);
		int right_symbol_col = current_col + (offset + 1);

		for (
			int* rightSymbol = first_symbol(right_symbol_row, right_symbol_col);
			rightSymbol != last_symbol(right_symbol_row, right_symbol_col);
		++rightSymbol)
		{
			assign_rule_if_possible(rules_table, *left_symbol, *rightSymbol, current_row, current_col);
		}
	}
}

template <int sentence_length, int max_symbol_length>
CCM cyk_table<sentence_length, max_symbol_length>::
cyk_table()
{
	static_assert(sentence_length > 0 && max_symbol_length > 0,
		"sentence_length and max_symbol_length must be greater than 0");

	for (int row = 0; row < size(); ++row)
	{
		for (int col = 0; col < size(); ++col)
		{
			table[row][col][special_field::symbol_count] = 0;
		}
	}
}

template <int sentence_length, int max_symbol_length>
CCM cyk_table<sentence_length, max_symbol_length>::
~cyk_table()
{
	
}

template <int sentence_length, int max_symbol_length>
CCM int cyk_table<sentence_length, max_symbol_length>::
size() const
{
	return sentence_length;
}

template <int sentence_length, int max_symbol_length>
CCM int* cyk_table<sentence_length, max_symbol_length>::
first_symbol(int row, int col)
{
	return table[row][col] + special_field::enum_size;
}

template <int sentence_length, int max_symbol_length>
CCM int* cyk_table<sentence_length, max_symbol_length>::
last_symbol(int row, int col)
{
	return table[row][col] + special_field::enum_size + table[row][col][special_field::symbol_count];
}

template <int sentence_length, int max_symbol_length>
CCM int cyk_table<sentence_length, max_symbol_length>::
max_num_of_symbols() const
{
	return max_symbol_length;
}

template <int sentence_length, int max_symbol_length>
CCM void cyk_table<sentence_length, max_symbol_length>::
fill_cell(int row, int col, int** rules_table)
{
	for (int i = 0; i < row; ++i)
	{
		assign_rules_for_two_cell_combination(i, row, col, rules_table);
	}
}

#endif