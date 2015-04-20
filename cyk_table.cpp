#include "cyk_table.h"

CCM cyk_table::cyk_table(const int block_id_, const char* source_code_localization_, int* prefs_) :
block_id(block_id_), source_code_localization(source_code_localization_), prefs(prefs_)
{
}

CCM cyk_table::~cyk_table()
{
}

CCM int cyk_table::size() const
{
	return preferences(block_id, source_code_localization).get(prefs, preferences::sentence_length);
}