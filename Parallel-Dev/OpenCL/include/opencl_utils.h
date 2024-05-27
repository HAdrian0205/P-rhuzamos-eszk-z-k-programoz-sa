#ifndef OPENCL_UTILS_H
#define OPENCL_UTILS_H

#include <CL/cl.h>

char* readKernelSource(const char* filename, size_t* length);
cl_kernel create_kernel(cl_program program, int mode, cl_int* ret);

#endif // OPENCL_UTILS_H
