#include <iostream>
#include <sstream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdarg>

#if !defined CUDA_DEBUG
#define CUDA_DEBUG 1
#endif

#if CUDA_DEBUG
#define log_debug(...) printf(__VA_ARGS__)
#else
#define log_debug(...)
#endif

#if !defined(CUDA_HELPER_H)
#define CUDA_HELPER_H

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define AT __FILE__ ":" TOSTRING(__LINE__)

#define CCM __host__ __device__

extern const char margin[];
extern const char margin_lined[];
extern const char error[];

//#define run_test(name, func) std::cout << margin << name << margin; func; std::cout << margin_lined << std::endl;

void test_header(const char* name);

template<class A, class B>
void expect_eq(A a, B b, const char *at)
{
	if (a != b)
	{
		std::cout << std::endl << error << std::endl <<
			"EXPECTATION FAILURE!" << std::endl <<
			"left = " << a << ";" << std::endl <<
			"right = " << b << ";" << std::endl <<
			"left != right." << std::endl <<
			"at: " << at << std::endl <<
			error << std::endl << std::endl;
	}
}

template<class A, class B>
void expect_table_eq(A a, B b, int size, const char *at)
{
	for (int i = 0; i < size; ++i)
	{
		if (a[i] != b[i])
		{
			std::cout << std::endl << error << std::endl <<
				"EXPECTATION FAILURE!" << std::endl <<
				"tables differ at: " << i <<std::endl <<
				"left = " << a[i] << ";" << std::endl <<
				"right = " << b[i] << ";" << std::endl <<
				"left != right." << std::endl <<
				"at: " << at << std::endl <<
				error << std::endl << std::endl;
		}
	}
}

class cuda_helper
{
public:
	cuda_helper(const char* source_code_localization_) :
		source_code_localization(source_code_localization_)
	{

	}

	template <class T>
	void allocate_on_device(T** dev_ptr, T size)
	{
		cudaError_t cuda_status = cudaMalloc(dev_ptr, size * sizeof(T));

		if (cuda_status != cudaSuccess)
		{
			std::stringstream ss;
			ss << "CudaMalloc  error (" << source_code_localization << ")!" << std::endl;

			throw std::runtime_error(ss.str());
		}
	}

	template <class T>
	void to_device(T* dev_ptr, const T* src, T size)
	{
		cudaError_t cuda_status = cudaMemcpy(dev_ptr, src, size * sizeof(T), cudaMemcpyHostToDevice);

		if (cuda_status != cudaSuccess)
		{
			std::stringstream ss;
			ss << "CudaMemcpy host --> device error (" << source_code_localization << ")!" << std::endl;

			throw std::runtime_error(ss.str());
		}
	}

	template <class T>
	void from_host_to_device(T** dev_ptr, const T* src, T size)
	{
		if (*dev_ptr == nullptr)
		{
			allocate_on_device(dev_ptr, size);
		}

		to_device(*dev_ptr, src, size);
	}

	void check_for_errors_after_launch()
	{
		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			std::stringstream ss;
			ss << "addKernel launch failed: " << cudaGetErrorString(cudaStatus) << "(" << source_code_localization << ")" << std::endl;

			throw std::runtime_error(ss.str());
		}
	}

	void device_synchronize()
	{
		cudaError_t cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			std::stringstream ss;
			ss << "cudaDeviceSynchronize returned error code " << cudaStatus <<
				" after launching kernel (" << source_code_localization << ")!" << std::endl;

			throw std::runtime_error(ss.str());
		}
	}

	template <class T>
	void host_memcpy(T *data, T *dev_handle, int size)
	{
		cudaError_t cudaStatus = cudaMemcpy(data, dev_handle, size * sizeof(T), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
		{
			std::stringstream ss;
			ss << "CudaMemcpy device --> host error (" << source_code_localization << ")!" << std::endl;

			throw std::runtime_error(ss.str());
		}
	}

	template <class T>
	void from_device_to_host(T *data, T *dev_handle, int size)
	{
		host_memcpy<T>(data, dev_handle, size);
	}

	template <class T>
	void free(T *dev_handle)
	{
		cudaError_t cudaStatus = cudaFree(dev_handle);
		if (cudaStatus == cudaSuccess)
		{
			dev_handle = nullptr;
		}
	}

private:
	const char* source_code_localization;
};

enum error : int
{
	no_errors_occured = 0,
	index_out_of_bounds = -1
};

CCM int generate_absolute_index(int x, int x_max);
CCM int generate_absolute_index(int x, int x_max, int y, int y_max);
CCM int generate_absolute_index(int x, int x_max, int y, int y_max, int z, int z_max);
CCM int generate_absolute_index(int x, int x_max, int y, int y_max, int z, int z_max, int i, int i_max);

CCM int table_get(int* table, int absolute_index);
CCM int table_set(int* table, int absolute_index, int value);

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

template <class T>
CCM T min(T a, T b)
{
	return a < b ? a : b;
}

template <class T>
CCM T max(T a, T b)
{
	return a > b ? a : b;
}

#endif
