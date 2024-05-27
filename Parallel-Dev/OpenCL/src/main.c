#define CL_TARGET_OPENCL_VERSION 120

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/cl.h>
#include <time.h>
#include <string.h>
#include "opencl_utils.h"
#include "input_utils.h"
#include "time_utils.h"

int main(int argc, char *argv[]) {
    if (argc != 6) {
        if (argc == 2 && (strcmp(argv[1], "help") == 0 || strcmp(argv[1], "--help") == 0)) {
            print_help(argv[0]);
            return 0;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    double a = atof(argv[1]);
    double b = atof(argv[2]);
    int size = atoi(argv[3]);
    int mode = atoi(argv[4]);
    int func = atoi(argv[5]);

    double h = (b - a) / size;

    double* x = (double*)malloc((size + 1) * sizeof(double));
    get_points(x, a, b, size, h);

    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;

    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    size_t source_size;
    char* source_str = readKernelSource("kernels/integral_kernel.cl", &source_size);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &ret);
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    cl_kernel kernel = create_kernel(program, mode, &ret);

    cl_mem x_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, (size + 1) * sizeof(double), NULL, &ret);
    cl_mem results_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (size + 1) * sizeof(double), NULL, &ret);

    ret = clEnqueueWriteBuffer(command_queue, x_mem, CL_TRUE, 0, (size + 1) * sizeof(double), x, 0, NULL, NULL);

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&x_mem);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&results_mem);
    ret = clSetKernelArg(kernel, 2, sizeof(int), (void*)&size);
    ret = clSetKernelArg(kernel, 3, sizeof(int), (void*)&func);
    ret = clSetKernelArg(kernel, 4, sizeof(double), (void*)&a);
    ret = clSetKernelArg(kernel, 5, sizeof(double), (void*)&b);

    size_t global_item_size = size + 1;
    size_t local_item_size = 64;

    if (global_item_size % local_item_size != 0) {
        global_item_size = (global_item_size / local_item_size + 1) * local_item_size;
    }

    struct timespec start_t, end_t;
    
    // Start time
    clock_gettime(CLOCK_MONOTONIC, &start_t);

    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to enqueue NDRange kernel. Error: %d\n", ret);
        return 1;
    }
    ret = clFinish(command_queue);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to finish command queue. Error: %d\n", ret);
        return 1;
    }

    // End time
    clock_gettime(CLOCK_MONOTONIC, &end_t);
    
    double* results = (double*)malloc((size + 1) * sizeof(double));
    ret = clEnqueueReadBuffer(command_queue, results_mem, CL_TRUE, 0, (size + 1) * sizeof(double), results, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to read buffer. Error: %d\n", ret);
        return 1;
    }

    double sum = 0.0;
    for (int i = 0; i <= size; i++) {
        sum += results[i];
    }

    double final_result;
    
    switch(mode) {
        case 0: final_result = (h / 3.0) * sum;
                break;
        case 1: final_result = h * sum;
                break;
        case 2: final_result = (h / 2.0) * sum;
                break;
        default: final_result = sum;
                break;
    }

    double elapsed_time = get_elapsed_time(start_t, end_t);
    double exact_value = exact_integral(a, b, func);
    double error = fabs(final_result - exact_value);

    printf("Az integral erteke: %.10f\n", final_result);
    printf("A pontos ertek: %.10f\n", exact_value);
    printf("A hiba: %.10f\n", error);
    printf("Eltelt ido: %.10f masodperc\n", elapsed_time);

    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(x_mem);
    ret = clReleaseMemObject(results_mem);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(x);
    free(results);
    free(source_str);

    return 0;
}
