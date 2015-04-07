#include "cuda_helper.h"

#include "cyk_common.h"
#include "cyk_table.h"
#include "cyk_rules_table.h"

#if !defined(DEVICE_MEMORY_H)
#define DEVICE_MEMORY_H

class inner_device_memory
{
public:
	inner_device_memory(const char* source_code_localization_) :
		source_code_localization(source_code_localization_)
	{

	}

	~inner_device_memory()
	{
	}

	//template<class T>
	//T* get_object(T** group)
	//{
	//	// preferences - zawsze to samo
	//	// cyk_table - table per block, konkretna kom�rka per koordynaty (shared)
	//	// cyk_rules - rules per block (shared)
	//}

private:
	const char* source_code_localization;
};

#endif