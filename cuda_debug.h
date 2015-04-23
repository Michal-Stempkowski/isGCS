#if !defined CUDA_DEBUG
#define CUDA_DEBUG 1
#endif

#if CUDA_DEBUG
#define log_debug(...) printf(__VA_ARGS__)
#else
#define log_debug(...)
#endif