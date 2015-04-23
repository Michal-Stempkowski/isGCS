#if !defined CUDA_HELPER_COMMON_H
#define CUDA_HELPER_COMMON_H

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define AT __FILE__ ":" TOSTRING(__LINE__)

#define CCM __host__ __device__

#endif