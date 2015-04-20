#include "cuda_helper.h"

#if !defined(PREFERENCES_H)
#define PREFERENCES_H

class preferences
{
public:
	enum option : int
	{
		sentence_length,
		max_alphabet_size,
		max_symbols_in_cell,
		number_of_blocks,
		number_of_threads,
		enum_size	// DO NOT use it as enum, it represents size of this enum
	};

	static const int invalid_value = -1;

	CCM preferences(const int block_id, const char* source_code_localization_);

	CCM int get(int *preferences, option opt) const;

	CCM ~preferences();

private:
	CCM int get_index(int field_id) const;

	const int block_id;
	const char* source_code_localization;
};

#endif