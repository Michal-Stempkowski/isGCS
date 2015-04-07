#include "cuda_helper.h"

#include "cyk_common.h"
#include "cyk_table.h"
#include "cyk_rules_table.h"

#if !defined(DEVICE_MEMORY_H)
#define DEVICE_MEMORY_H

template <int sentence_length, int max_symbol_length>
class inner_device_memory
{
public:
	inner_device_memory(const char* source_code_localization_,
		cyk_table<sentence_length, max_symbol_length> cyk_table_,
		cyk_rules_table<max_symbol_length> cyk_rules_) :

		source_code_localization(source_code_localization_),
		cyk_table(cyk_table_),
		cyk_rules(cyk_rules_)
	{

	}

	~inner_device_memory()
	{
	}

private:
	const char* source_code_localization;
	cyk_table<sentence_length, max_symbol_length> cyk_table;
	cyk_rules_table<max_symbol_length> cyk_rules;
};

#endif