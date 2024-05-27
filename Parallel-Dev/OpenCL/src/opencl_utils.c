#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "opencl_utils.h"

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
