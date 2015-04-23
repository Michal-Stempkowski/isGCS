#if !defined(CUDA_HELPER_H)
#define CUDA_HELPER_H

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


#endif