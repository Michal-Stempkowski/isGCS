#include "cuda_helper.h"

#if !defined(ARRAY_H)
#define ARRAY_H

class cuda_array
{
public:
	enum special_field : int
	{
		dimension,
		max_size,
		enum_size	// DO NOT use it as enum, it represents size of this enum
	};

	enum error : int
	{
		no_error = 0,
		invalid_dimension = -1,
		index_out_of_bounds = -2
	};

	cuda_array(const char* source_code_localization_) :
		source_code_localization(source_code_localization_)
	{

	}

	~cuda_array()
	{
	}

	int get(int* arr, int x, int offset = 0)
	{
		int error = check_for_errors(arr, x, offset);

		return error ? error : arr[offset + enum_size + x];
	}

	int set(int* arr, int x, int value, int offset = 0)
	{
		int error = check_for_errors(arr, x, offset);

		if (!error)
		{
			arr[offset + enum_size + x] = value;
		}

		return error;
	}

	//enum option : int
	//{
	//	sentence_length,
	//	max_alphabet_size,
	//	max_symbols_in_cell,
	//	number_of_blocks,
	//	number_of_threads,
	//	enum_size	// DO NOT use it as enum, it represents size of this enum
	//};

private:
	int check_for_errors(int* arr, int x, int offset)
	{
		if (arr[offset] != 1)
		{
			return invalid_dimension;;
		}
		else if (x >= arr[offset + max_size])
		{
			return index_out_of_bounds;
		}

		return no_error;
	}

	const char* source_code_localization;
};

#endif