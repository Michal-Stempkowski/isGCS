#include <new>
#include <sstream>
#include <string>

#if defined(CCM)

#elif  defined(__CUDACC__)
#define CCM __host__ __device__
#else
#define CCM
#endif

#if !defined(CUDA_HELPER_H)
#define CUDA_HELPER_H

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define AT __FILE__ ":" TOSTRING(__LINE__)

#define NOTHING

class inner_cuda
{
public:
	inner_cuda(const char* source_code_localization_) :
		source_code_localization(source_code_localization_)
	{

	}

	template <class T>
	void device_malloc(T **dev_handle, int size = sizeof(T))
	{
		cudaError_t cudaStatus = cudaMalloc((void**)dev_handle, size);
		if (cudaStatus != cudaSuccess) 
		{
			std::stringstream ss;
			ss << "CudaMalloc  error (" << source_code_localization << ")!" << std::endl;

			throw std::runtime_error(ss.str());
		}
	}

	template <class T>
	void device_memcpy(T *dev_handle, const T *data, int size = sizeof(T))
	{
		cudaError_t cudaStatus = cudaMemcpy(dev_handle, &data, size, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) 
		{
			std::stringstream ss;
			ss << "CudaMemcpy host --> device error (" << source_code_localization << ")!" << std::endl;

			throw std::runtime_error(ss.str());
		}
	}

	template <class T>
	void copy_to(T **dev_handle, const T *data, int size = sizeof(T))
	{
		if (*dev_handle == nullptr)
		{
			device_malloc(dev_handle, size);
		}

		device_memcpy(*dev_handle, data, size);
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
	void host_memcpy(T *data, T *dev_handle, int size = sizeof(T))
	{
		cudaError_t cudaStatus = cudaMemcpy(data, dev_handle, size, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) 
		{
			std::stringstream ss;
			ss << "CudaMemcpy device --> host error (" << source_code_localization << ")!" << std::endl;

			throw std::runtime_error(ss.str());
		}
	}

	template <class T>
	void copy_from(T *data, T *dev_handle, int size = sizeof(T))
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

#define cuda_helper inner_cuda(AT)

#define INNER_LAUNCH_KERNEL(kernel, x, y, ...) kernel << < x, y >> >(__VA_ARGS__)

template<class A, class B>
void expect_eq_with_location(A a, B b, const char *at)
{
	const char margin[] = "===================================";

	if (a != b)
	{
		std::cout << std::endl << margin << std::endl << 
			"EXPECTATION FAILURE!" << std::endl <<
			"left = " << a << ";" << std::endl <<
			"right = " << b << ";" << std::endl <<
			"left != right." << std::endl <<
			"at: " << at << std::endl <<
			margin << std::endl << std::endl;
	}
}

#define expect_eq(a, b) expect_eq_with_location(a, b, AT)

#endif
