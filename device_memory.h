#include "cuda_helper.h"

//#include "cyk_common.h"
//#include "cyk_table.h"
//#include "cyk_rules_table.h"
//#include "cuda_helper.h"
//
//#include "preferences.h"

#if !defined(DEVICE_MEMORY_H)
#define DEVICE_MEMORY_H

class device_memory
{
public:
	static const int INVALID_LOCATION = -1;

	CCM device_memory(const char* source_code_localization_) :
		source_code_localization(source_code_localization_)
	{

	}

	CCM ~device_memory()
	{
	}

	template<class T>
	CCM T* get_object(T** group)
	{
		return group[blockIdx.x];
	}

	//template <class prefs_t>
	//CCM int get_current_cyk_row(int job_id, prefs_t** prefs)
	//{
	//	int row = job_id / get_object(prefs)->get_sentence_length();

	//	return row != 0 ? row : INVALID_LOCATION;

	//}

	//template <class prefs_t>
	//CCM int get_current_cyk_col(int job_id, prefs_t** prefs)
	//{
	//	int row = get_current_cyk_row(job_id, prefs);
	//	int col = job_id % row;

	//	return col + row < get_object(prefs)->get_sentence_length() ? col : INVALID_LOCATION;
	//}

private:
	const char* source_code_localization;
};

//#define device_memory inner_device_memory(AT)

#endif