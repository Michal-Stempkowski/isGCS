#include "preferences.h"

CCM preferences::preferences(const int block_id_, const char* source_code_localization_) :
source_code_localization(source_code_localization_), block_id(block_id_)
{

}

CCM int preferences::get(int *preferences, option opt)
{
	if (opt < 0 || opt >= enum_size)
	{
		return invalid_value;
	}

	return preferences[get_index(opt)];
}

CCM preferences::~preferences()
{
}

CCM int preferences::get_index(int field_id)
{
	return block_id * enum_size + field_id;
}