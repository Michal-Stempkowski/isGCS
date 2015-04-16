#include "cuda_helper.h"
#include "cyk_common.h"

#include "cyk_rules_table.h"

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

	CCM int get_symbol_at(int row, int col, int pos);
	CCM void set_symbol_at(int row, int col, int pos, int value);
	CCM int get_symbol_count(int row, int col);

	CCM int max_num_of_symbols() const;

	void fill_first_row(int* sentence)
	{
		for (int i = 0; i < sentence_length; ++i)
		{
			assign_rule(0, i, sentence[i]);
		}
	}

	CCM void fill_cell(int row, int col, cyk_rules_table<max_symbol_length> *rules_table);

	CCM int get_cell_rule(int row, int col, int rule_number);

private:
	CCM void assign_rules_for_two_cell_combination(int offset, int current_row, int current_col, 
		cyk_rules_table<max_symbol_length> *rules_table);
	CCM void assign_rule_if_possible(cyk_rules_table<max_symbol_length> *rules_table, 
		int left_symbol, int right_symbol, int current_row, int current_col);
	CCM void assign_rule(int row, int col, int rule);

	enum special_field : int
	{
		symbol_count,
		enum_size	// DO NOT use it as enum, it represents size of this enum
	};

	int table[sentence_length][sentence_length][max_symbol_length + special_field::enum_size];
};

#define CYK_TABLE(type) template <int sentence_length, int max_symbol_length> CCM type cyk_table<sentence_length, max_symbol_length>::

CYK_TABLE(void) assign_rule(int row, int col, int rule)
{
	//get_symbol_at(1, 1, 1);
	//set_symbol_at(row, col, get_symbol_count(row, col), rule);
	//set_symbol_at(0, 0, 0, 0);
	table[0][0][1] = 0;
	//++table[row][col][special_field::symbol_count];
}

CYK_TABLE(void) assign_rule_if_possible(cyk_rules_table<max_symbol_length> *rules_table, 
	int left_symbol, int right_symbol, int current_row, int current_col)
{
	int rule = rules_table->get_rule_by_right_side(left_symbol, right_symbol);

	if (rule != constants::NO_MATCHING_RULE)
	{
		assign_rule(current_row, current_col, rule);
	}
}

CYK_TABLE(void) assign_rules_for_two_cell_combination(int offset, int current_row, int current_col, 
	cyk_rules_table<max_symbol_length> *rules_table)
{
	for (int i = 0; i < get_symbol_count(offset, current_col); ++i)
	{
		int left_symbol = get_symbol_at(offset, current_col, i);
		int right_symbol_row = current_row - (offset + 1);
		int right_symbol_col = current_col + (offset + 1);

		for (int j = 0; j < get_symbol_count(right_symbol_row, right_symbol_col); ++j)
		{
			int right_symbol = get_symbol_at(right_symbol_row, right_symbol_col, j);

			assign_rule_if_possible(rules_table, left_symbol, right_symbol, current_row, current_col);
		}
	}
}

CYK_TABLE(NOTHING) cyk_table()
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

CYK_TABLE(NOTHING) ~cyk_table()
{
	
}

CYK_TABLE(int) size() const
{
	return sentence_length;
}

CYK_TABLE(int*) first_symbol(int row, int col)
{
	return table[row][col] + special_field::enum_size;
}

CYK_TABLE(int*) last_symbol(int row, int col)
{
	return first_symbol(row, col) + table[row][col][special_field::symbol_count];
}

CYK_TABLE(int) get_symbol_at(int row, int col, int pos)
{
	return table[row][col][special_field::enum_size + pos];
}

CYK_TABLE(void) set_symbol_at(int row, int col, int pos, int value)
{
	table[row][col][special_field::enum_size + pos] = value;
}

CYK_TABLE(int) get_symbol_count(int row, int col)
{
	return table[row][col][special_field::symbol_count];
}

CYK_TABLE(int) max_num_of_symbols() const
{
	return max_symbol_length;
}

CYK_TABLE(void) fill_cell(int row, int col, cyk_rules_table<max_symbol_length> *rules_table)
{
	for (int i = 0; i < row; ++i)
	{
		assign_rules_for_two_cell_combination(i, row, col, rules_table);
	}
}

CYK_TABLE(int) get_cell_rule(int row, int col, int rule_number)
{
	return table[row][col][special_field::enum_size + rule_number];
}

#endif