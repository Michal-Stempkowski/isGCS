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

template <const char* source_code_localization>
class inner_cuda
{
public:
	template <class T>
	void device_malloc(T **dev_handle, int size = sizeof(T))
	{
		cudaError_t cudaStatus = cudaMalloc((void**)dev_handle, size);
		if (cudaStatus != cudaSuccess) {
			std::stringstream ss;
			ss << "CudaMalloc  error (" << source_code_localization << ")!" << std::endl;

			throw std::runtime_error(ss.str());
		}
	}

};


template <class T>
void device_malloc(T **dev_handle, const char *localization, int size = sizeof(T))
{
	cudaError_t cudaStatus = cudaMalloc((void**)dev_handle, size);
	if (cudaStatus != cudaSuccess) {
		std::stringstream ss;
		ss << "CudaMalloc  error (" << localization << ")!" << std::endl;

		throw std::runtime_error(ss.str());
	}
}

template <class T>
void device_memcpy(T *dev_handle, const T *data, const char *localization, int size = sizeof(T))
{
	cudaError_t cudaStatus = cudaMemcpy(dev_handle, &data, size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		std::stringstream ss;
		ss << "CudaMemcpy host --> device error (" << localization << ")!" << std::endl;

		throw std::runtime_error(ss.str());
	}
}

#define INNER_LAUNCH_KERNEL(kernel, x, y, ...) kernel << < x, y >> >(__VA_ARGS__)

void check_for_errors_after_launch(const char *localization)
{
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		std::stringstream ss;
		ss << "addKernel launch failed: " << cudaGetErrorString(cudaStatus) << "(" << localization << ")" << std::endl;

		throw std::runtime_error(ss.str());
	}
}


void device_synchronize(const char *localization)
{
	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		std::stringstream ss;
		ss << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching kernel (" << localization << ")!" << std::endl;

		throw std::runtime_error(ss.str());
	}
}

template <class T>
void host_memcpy(T *data, T *dev_handle, const char *localization, int size = sizeof(T))
{
	cudaError_t cudaStatus = cudaMemcpy(data, dev_handle, size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		std::stringstream ss;
		ss << "CudaMemcpy device --> host error (" << localization << ")!" << std::endl;

		throw std::runtime_error(ss.str());
	}
}

//template <class T>
//bool device_malloc(T **dev_handle, const char *localization)
//{
//	cudaError_t cudaStatus = cudaMalloc((void**)dev_handle, sizeof(T));
//	if (cudaStatus != cudaSuccess) {
//		std::cout << "CudaMalloc error (" << localization << ")!" << std::endl;
//
//		return false;
//	}
//
//	return true;
//
//	/*cudaStatus = cudaMalloc((void**)&dev_table, sizeof(cyk_table<4, 10>));
//	if (cudaStatus != cudaSuccess) {
//	fprintf(stderr, "cudaMalloc failed on cyk_table!");
//	goto Error;
//	}*/
//}

#endif
