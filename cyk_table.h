#include "cuda_helper.h"

#include "preferences.h"

class cyk_table
{
public:
	CCM cyk_table(const int block_id_, const char* source_code_localization_, int* prefs_);
	CCM ~cyk_table();

	CCM int size() const;

private:
	const int block_id;
	const char* source_code_localization;
	int * prefs;
};