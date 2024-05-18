#define CL_TARGET_OPENCL_VERSION 120

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/cl.h>
#include <time.h>

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

void get_size(double* a, double* b, int* size, double* h) {
    printf("Adja meg az intervallumot: \n");
    printf("Min(a): ");
    if (scanf("%lf", a) != 1) {
        fprintf(stderr, "Error reading input.\n");
        exit(1);
    }
    printf("Max(b): ");
    if (scanf("%lf", b) != 1) {
        fprintf(stderr, "Error reading input.\n");
        exit(1);
    }

    printf("Adja meg az n-et: ");
    if (scanf("%d", size) != 1) {
        fprintf(stderr, "Error reading input.\n");
        exit(1);
    }

    *h = (*b - *a) / (*size);
}

void get_points(double array[], double a, double b, int size, double h) {
    for (int i = 0; i <= size; i++) {
        array[i] = a + i * h;
    }
}

int main() {
    int size, mode, func;
    double a, b, h;

    get_size(&a, &b, &size, &h);

    double* x = (double*)malloc((size + 1) * sizeof(double));
    get_points(x, a, b, size, h);

    printf("Valassza ki, hogy melyik szabaly szerint integralna:\n");
    printf("0 - Osszetett simpson, 1 - Osszetett teglalap, 2 - Osszetett trapez\n");
    if (scanf("%d", &mode) != 1) {
        fprintf(stderr, "Error reading input.\n");
        exit(1);
    }
    printf("Valassza ki, hogy melyik fuggvenyt integralna:\n");
    printf("0 - sin, 1 - cos, 2 - exp, 3 - sqrt, 4 - log\n");
    if (scanf("%d", &func) != 1) {
        fprintf(stderr, "Error reading input.\n");
        exit(1);
    }

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
    char* source_str = readKernelSource("integral_kernel.cl", &source_size);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &ret);
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    cl_kernel kernel;
    switch (mode) {
        case 0:
            kernel = clCreateKernel(program, "simpson_integral", &ret);
            break;
        case 1:
            kernel = clCreateKernel(program, "rectangle_integral", &ret);
            break;
        case 2:
            kernel = clCreateKernel(program, "trapezoidal_integral", &ret);
            break;
        default:
            fprintf(stderr, "Invalid mode selected.\n");
            return -1;
    }

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
    size_t local_item_size = 1;
	
	clock_t start_t, end_t;
	
	start_t = clock();
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
    ret = clFinish(command_queue);
	
    double* results = (double*)malloc((size + 1) * sizeof(double));
    ret = clEnqueueReadBuffer(command_queue, results_mem, CL_TRUE, 0, (size + 1) * sizeof(double), results, 0, NULL, NULL);
	
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
		default: final_result = sum;
				break;
	}

	end_t = clock();
	
	double elapsed_time = (double)(end_t - start_t) / CLOCKS_PER_SEC;

    printf("Az integral erteke: %.10f\n", final_result);
	printf("Eltelt ido: %.10f\n", elapsed_time);

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