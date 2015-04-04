#if defined(CCM)

#elif  defined(__CUDACC__)
#define CCM __host__ __device__
#else
#define CCM
#endif
