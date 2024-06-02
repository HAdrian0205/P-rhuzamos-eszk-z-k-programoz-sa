#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/cl.h>
#include <string.h>
#include "opencl_utils.h"
#include "input_utils.h"
#include "time_utils.h"

#define MAX_SIZE_VALUES 100

int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    if (argc == 2 && (strcmp(argv[1], "help") == 0 || strcmp(argv[1], "--help") == 0)) {
        print_help(argv[0]);
        return 0;
    }

    if (argc == 3 && strcmp(argv[1], "--complexity") == 0) {
        Params params[MAX_SIZE_VALUES];
        int count = read_params(argv[2], params);
        if (count == -1) {
            return 1;
        }

        double *sizes = (double *)malloc(count * sizeof(double));
        double *times = (double *)malloc(count * sizeof(double));

        for (int i = 0; i < count; i++) {
            double final_result, exact_value, error;
            double elapsed_time = run_algorithm(params[i].a, params[i].b, params[i].n, params[i].mode, params[i].func, &final_result, &exact_value, &error);

            sizes[i] = log10((double)params[i].n);
            times[i] = log10(elapsed_time);

           // printf("n = %d, elapsed_time = %.10f\n", params[i].n, elapsed_time);
        }

        double a, b;
        least_squares(sizes, times, count, &a, &b);
        printf("Time complexity: O(n^%.2f)\n", a);

        free(sizes);
        free(times);
        return 0;
    }

    if (argc >= 7 && strcmp(argv[1], "--simpson") == 0) {
        int dim = atoi(argv[2]);
        if (argc != 3 + 3 * dim + 1) {
            fprintf(stderr, "Wrong number of arguments.\n");
            return 1;
        }

        double lower[dim];
        double upper[dim];
        int n[dim];

        int index = 3;
        for (int i = 0; i < dim; i++) {
            lower[i] = atof(argv[index]);
            upper[i] = atof(argv[index + 1]);
            n[i] = atoi(argv[index + 2]);
            index += 3;
        }
        int func = atoi(argv[index]);

        double result;
        double elapsed_time = run_simpson_nd(lower, upper, n, dim, func, &result);

        printf("Value of the integral: %.10f\n", result);
        printf("Elapsed time: %.10f seconds\n", elapsed_time);
        return 0;
    }

    if (argc != 6) {
        print_usage(argv[0]);
        return 1;
    }

    double a = atof(argv[1]);
    double b = atof(argv[2]);
    int size = atoi(argv[3]);
    int mode = atoi(argv[4]);
    int func = atoi(argv[5]);

    double final_result, exact_value, error;
    double elapsed_time = run_algorithm(a, b, size, mode, func, &final_result, &exact_value, &error);

    printf("Value of the integral: %.10f\n", final_result);
    printf("Exact value of the integral: %.10f\n", exact_value);
    printf("Approximation error: %.10f\n", error);
    printf("Elapsed time: %.10f seconds\n", elapsed_time);

    return 0;
}