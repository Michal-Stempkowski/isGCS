#include "cuda_helper.h"

#include "cyk_common.h"

#if !defined(CYK_RULES_TABLE_H)
#define CYK_RULES_TABLE_H

template <int max_rules_length>
class cyk_rules_table
{
public:
	CCM cyk_rules_table(int size_, int** rules_table);
	CCM ~cyk_rules_table();

private:
	int size;
	int rules[max_rules_length][max_rules_length];
};

#define CYK_RULES_TABLE(type) template <int max_rules_length> CCM type cyk_rules_table<max_rules_length>::

CYK_RULES_TABLE(NOTHING) cyk_rules_table(int size_, int** rules_table) :
	size(size_)
{
	std::copy(rules_table, rules_table + max_rules_length * max_rules_length, rules);
}

CYK_RULES_TABLE(NOTHING) ~cyk_rules_table()
{
}

#endif