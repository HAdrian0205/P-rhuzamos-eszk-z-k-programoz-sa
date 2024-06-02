#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <CL/cl.h>
#include <math.h>
#include "opencl_utils.h"
#include "time_utils.h"
#include "input_utils.h"

#define LOCAL_SIZE 128

char* readKernelSource(const char* filename, size_t* length) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    if (file_size == -1) {
        fprintf(stderr, "Failed to determine file size.\n");
        fclose(file);
        exit(1);
    }
    rewind(file);

    char* source = (char*)malloc(file_size + 1);
    if (!source) {
        fprintf(stderr, "Failed to allocate memory for kernel source.\n");
        fclose(file);
        exit(1);
    }

    size_t read_size = fread(source, 1, file_size, file);
    if (read_size != file_size) {
        fprintf(stderr, "Error reading kernel file.\n");
        free(source);
        fclose(file);
        exit(1);
    }
    source[file_size] = '\0';

    fclose(file);

    *length = file_size;
    return source;
}

cl_kernel create_kernel(cl_program program, int mode, cl_int* ret) {
    cl_kernel kernel;
    switch (mode) {
        case 0:
            kernel = clCreateKernel(program, "simpson_integral", ret);
            break;
        case 1:
            kernel = clCreateKernel(program, "rectangle_integral", ret);
            break;
        case 2:
            kernel = clCreateKernel(program, "trapezoidal_integral", ret);
            break;
        default:
            fprintf(stderr, "Invalid mode selected.\n");
            exit(1);
    }
    return kernel;
}

cl_kernel select_function_kernel(cl_program program, int func, cl_int* ret) {
    switch (func) {
        case 0: return clCreateKernel(program, "sin_kernel", ret);
        case 1: return clCreateKernel(program, "cos_kernel", ret);
        case 2: return clCreateKernel(program, "exp_kernel", ret);
        case 3: return clCreateKernel(program, "sqrt_kernel", ret);
        default: return clCreateKernel(program, "log_kernel", ret);
    }
}

double run_algorithm(double a, double b, int size, int mode, int func, double* final_result, double* exact_value, double* error) {
    double h = (b - a) / size;
    double *x = (double *)malloc((size + 1) * sizeof(double));
    get_points(x, a, b, size, h);

    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;

    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);

    size_t source_size;
    char *source_str = readKernelSource("kernels/integral_kernel.cl", &source_size);

    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    if (ret != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Error in kernel: %s\n", log);
        free(log);
        exit(1);
    }

    cl_kernel func_kernel = select_function_kernel(program, func, &ret);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to create function kernel. Error: %d\n", ret);
        exit(1);
    }

    cl_kernel integral_kernel = create_kernel(program, mode, &ret);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to create integration kernel. Error: %d\n", ret);
        exit(1);
    }

    cl_kernel final_sum_kernel = clCreateKernel(program, "final_sum_kernel", &ret);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to create final sum kernel. Error: %d\n", ret);
        exit(1);
    }

    cl_mem x_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, (size + 1) * sizeof(double), NULL, &ret);
    cl_mem results_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, (size + 1) * sizeof(double), NULL, &ret);
    cl_mem integral_results_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (size + 1) * sizeof(double), NULL, &ret);

    ret = clEnqueueWriteBuffer(command_queue, x_mem, CL_TRUE, 0, (size + 1) * sizeof(double), x, 0, NULL, NULL);

    ret = clSetKernelArg(func_kernel, 0, sizeof(cl_mem), (void *)&x_mem);
    ret = clSetKernelArg(func_kernel, 1, sizeof(cl_mem), (void *)&results_mem);

    size_t global_item_size = size + 1;
    size_t local_item_size = LOCAL_SIZE;

    if (global_item_size % local_item_size != 0) {
        global_item_size = (global_item_size / local_item_size + 1) * local_item_size;
    }

    cl_event func_event, integral_event, final_sum_event;

    // Function and integral kernels
    ret = clEnqueueNDRangeKernel(command_queue, func_kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, &func_event);
    ret = clWaitForEvents(1, &func_event);
    ret = clFinish(command_queue);

    ret = clSetKernelArg(integral_kernel, 0, sizeof(cl_mem), (void *)&results_mem);
    ret = clSetKernelArg(integral_kernel, 1, sizeof(cl_mem), (void *)&integral_results_mem);
    ret = clSetKernelArg(integral_kernel, 2, sizeof(int), (void *)&size);

    ret = clEnqueueNDRangeKernel(command_queue, integral_kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, &integral_event);
    ret = clWaitForEvents(1, &integral_event);
    ret = clFinish(command_queue);

    // Final summary kernel
    size_t num_work_groups = global_item_size / local_item_size;
    cl_mem final_sum_results_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, num_work_groups * sizeof(double), NULL, &ret);
    double *final_sum_results = (double *)malloc(num_work_groups * sizeof(double));

    ret = clSetKernelArg(final_sum_kernel, 0, sizeof(cl_mem), (void *)&integral_results_mem);
    ret = clSetKernelArg(final_sum_kernel, 1, sizeof(cl_mem), (void *)&final_sum_results_mem);
    ret = clSetKernelArg(final_sum_kernel, 2, local_item_size * sizeof(double), NULL);

    ret = clEnqueueNDRangeKernel(command_queue, final_sum_kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, &final_sum_event);
    ret = clWaitForEvents(1, &final_sum_event);
    ret = clFinish(command_queue);

    ret = clEnqueueReadBuffer(command_queue, final_sum_results_mem, CL_TRUE, 0, num_work_groups * sizeof(double), final_sum_results, 0, NULL, NULL);

    double sum = 0.0;
    for (size_t j = 0; j < num_work_groups; j++) {
        sum += final_sum_results[j];
    }

    switch (mode) {
        case 0:
            *final_result = (h / 3.0) * sum;
            break;
        case 1:
            *final_result = h * sum;
            break;
        case 2:
            *final_result = (h / 2.0) * sum;
            break;
        default:
            *final_result = sum;
            break;
    }

    // Get profiling info
    cl_ulong time_start, time_end;
    double nanoSeconds;

    clGetEventProfilingInfo(func_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(func_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    nanoSeconds = time_end - time_start;
    double func_time = nanoSeconds / 1000000000.0;

    clGetEventProfilingInfo(integral_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(integral_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    nanoSeconds = time_end - time_start;
    double integral_time = nanoSeconds / 1000000000.0;

    clGetEventProfilingInfo(final_sum_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(final_sum_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    nanoSeconds = time_end - time_start;
    double final_sum_time = nanoSeconds / 1000000000.0;

    /* printf("Function kernel execution time: %0.10f s\n", func_time);
    printf("Integral kernel execution time: %0.10f s\n", integral_time);
    printf("Final sum kernel execution time: %0.10f s\n", final_sum_time); */

    double elapsed_time = func_time + integral_time + final_sum_time;
    *exact_value = exact_integral(a, b, func);
    *error = fabs(*final_result - *exact_value);

    ret = clReleaseKernel(func_kernel);
    ret = clReleaseKernel(integral_kernel);
    ret = clReleaseKernel(final_sum_kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(x_mem);
    ret = clReleaseMemObject(results_mem);
    ret = clReleaseMemObject(integral_results_mem);
    ret = clReleaseMemObject(final_sum_results_mem);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(x);
    free(final_sum_results);
    free(source_str);

    return elapsed_time;
}

double run_simpson_nd(double *lower, double *upper, int *n, int dim, int func, double *result) {
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;

    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);

    size_t source_size;
    char *source_str = readKernelSource("kernels/integral_kernel.cl", &source_size);

    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    if (ret != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Error in kernel: %s\n", log);
        free(log);
        exit(1);
    }

    cl_kernel simpson_kernel = clCreateKernel(program, "simpson_kernel", &ret);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to create simspon kernel. Error: %d\n", ret);
        exit(1);
    }

    int total_points = 1;
    for (int i = 0; i < dim; i++) {
        total_points *= (n[i] + 1);
    }

    cl_mem lower_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, dim * sizeof(double), NULL, &ret);
    cl_mem upper_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, dim * sizeof(double), NULL, &ret);
    cl_mem n_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, dim * sizeof(int), NULL, &ret);
    cl_mem results_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, total_points * sizeof(double), NULL, &ret);

    ret = clEnqueueWriteBuffer(command_queue, lower_mem, CL_TRUE, 0, dim * sizeof(double), lower, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, upper_mem, CL_TRUE, 0, dim * sizeof(double), upper, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, n_mem, CL_TRUE, 0, dim * sizeof(int), n, 0, NULL, NULL);

    ret = clSetKernelArg(simpson_kernel, 0, sizeof(cl_mem), (void *)&lower_mem);
    ret = clSetKernelArg(simpson_kernel, 1, sizeof(cl_mem), (void *)&upper_mem);
    ret = clSetKernelArg(simpson_kernel, 2, sizeof(cl_mem), (void *)&n_mem);
    ret = clSetKernelArg(simpson_kernel, 3, sizeof(cl_mem), (void *)&results_mem);
    ret = clSetKernelArg(simpson_kernel, 4, sizeof(int), (void *)&dim);
    ret = clSetKernelArg(simpson_kernel, 5, sizeof(int), (void *)&func);

    size_t global_item_size = total_points;
    size_t local_item_size = LOCAL_SIZE;

    if (global_item_size % local_item_size != 0) {
        global_item_size = (global_item_size / local_item_size + 1) * local_item_size;
    }

    cl_event simpson_event;

    struct timespec start_t, end_t;
    clock_gettime(CLOCK_MONOTONIC, &start_t);

    ret = clEnqueueNDRangeKernel(command_queue, simpson_kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, &simpson_event);
    ret = clWaitForEvents(1, &simpson_event);
    ret = clFinish(command_queue);

    double *results = (double *)malloc(total_points * sizeof(double));
    ret = clEnqueueReadBuffer(command_queue, results_mem, CL_TRUE, 0, total_points * sizeof(double), results, 0, NULL, NULL);

    double sum = 0.0;
    for (int i = 0; i < total_points; i++) {
        sum += results[i];
    }

    clock_gettime(CLOCK_MONOTONIC, &end_t);

    double volume = 1.0;
    for (int i = 0; i < dim; i++) {
        volume *= (upper[i] - lower[i]) / (3.0 * n[i]);
    }

    *result = sum * volume;

    // Get profiling info
    cl_ulong time_start, time_end;
    double nanoSeconds;

    clGetEventProfilingInfo(simpson_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(simpson_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    nanoSeconds = time_end - time_start;
    double simpson_time = nanoSeconds / 1000000000.0;

    //printf("Simpson kernel execution time: %0.10f s\n", simpson_time);

    double elapsed_time = simpson_time;

    ret = clReleaseKernel(simpson_kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(lower_mem);
    ret = clReleaseMemObject(upper_mem);
    ret = clReleaseMemObject(n_mem);
    ret = clReleaseMemObject(results_mem);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(results);
    free(source_str);

    return elapsed_time;
}