#include "cuda_helper.h"

const char margin[] = "\n-----------------------------------\n";
const char margin_lined[] = "\n===================================\n";
const char error[] = "+++++++++++++++++++++++++++++++++++";

void test_header(const char* name)
{
	std::cout << "+++" << name << std::endl;
}

//static bool bounds(int a, int max_a)
//{
//	return 
//}

static CCM int apply_param(int a, int a_max, int index)
{
	if (a < 0 || a >= a_max)
	{
		return error::index_out_of_bounds;
	}

	if (index != error::index_out_of_bounds)
	{
		index = index * a_max + a;
	}

	return index;
}

CCM int generate_absolute_index(int x, int x_max)
{
	return apply_param(x, x_max, 0);
}

CCM int generate_absolute_index(int x, int x_max, int y, int y_max)
{
	return apply_param(y, y_max, generate_absolute_index(x, x_max));
}

CCM int generate_absolute_index(int x, int x_max, int y, int y_max, int z, int z_max)
{
	return apply_param(z, z_max, generate_absolute_index(x, x_max, y, y_max));
}

CCM int generate_absolute_index(int x, int x_max, int y, int y_max, int z, int z_max, int i, int i_max)
{
	return apply_param(i, i_max, generate_absolute_index(x, x_max, y, y_max, z, z_max));
}

CCM int table_get(int* table, int absolute_index)
{
	return 
		absolute_index >= error::no_errors_occured ? 
		table[absolute_index] : 
		absolute_index;
}

CCM int table_set(int* table, int absolute_index, int value)
{
	return absolute_index >= error::no_errors_occured ? 
		table[absolute_index] = value, error::no_errors_occured : 
		absolute_index;
}
