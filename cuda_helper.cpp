#include "cuda_helper.h"

const char margin[] = "\n-----------------------------------\n";
const char margin_lined[] = "\n===================================\n";
const char error[] = "+++++++++++++++++++++++++++++++++++";

void test_header(const char* name)
{
	std::cout << "+++" << name << std::endl;
}
