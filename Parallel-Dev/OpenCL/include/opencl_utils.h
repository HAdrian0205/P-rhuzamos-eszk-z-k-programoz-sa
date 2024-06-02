#ifndef OPENCL_UTILS_H
#define OPENCL_UTILS_H

#include <CL/cl.h>

char* readKernelSource(const char* filename, size_t* length);
cl_kernel create_kernel(cl_program program, int mode, cl_int* ret);
cl_kernel select_function_kernel(cl_program program, int func, cl_int* ret);
double run_algorithm(double a, double b, int size, int mode, int func, double* final_result, double* exact_value, double* error);
double run_simpson_nd(double *lower, double *upper, int *n, int dim, int func, double *result);

#endif // OPENCL_UTILS_H
