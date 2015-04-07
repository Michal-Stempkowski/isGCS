#include "cuda_helper.h"

#include "cyk_common.h"

#if !defined(CYK_RULES_TABLE_H)
#define CYK_RULES_TABLE_H

template <int max_rules_length>
class cyk_rules_table
{
public:
	CCM cyk_rules_table();
	cyk_rules_table(int (&rules_table)[max_rules_length][max_rules_length]);
	CCM ~cyk_rules_table();

	CCM int get_rule_by_right_side(int left_symbol, int right_symbol);

private:
	int rules[max_rules_length][max_rules_length];
};

#define CYK_RULES_TABLE(type) template <int max_rules_length> CCM type cyk_rules_table<max_rules_length>::

CYK_RULES_TABLE(NOTHING) cyk_rules_table()
{
	
}

template <int max_rules_length> 
cyk_rules_table<max_rules_length>::
cyk_rules_table(int(&rules_table)[max_rules_length][max_rules_length])
{
	std::copy(&rules_table[0][0], &rules_table[0][0] + max_rules_length * max_rules_length, &rules[0][0]);
}

CYK_RULES_TABLE(NOTHING) ~cyk_rules_table()
{
}

CYK_RULES_TABLE(int) get_rule_by_right_side(int left_symbol, int right_symbol)
{
	return rules[left_symbol][right_symbol];
}

#endif