#include "cuda_helper.h"

#include "preferences.h"

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

private:
	const int block_id;
	const char* source_code_localization;
	const int size_;
	const int depth_;

	int* prefs;
	int* table;
	int* table_header;

	CCM int set_symbol_at(int row, int col, int pos, int val);
};