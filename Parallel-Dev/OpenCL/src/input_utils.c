#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "input_utils.h"

void get_points(double array[], double a, double b, int size, double h) {
    for (int i = 0; i <= size; i++) {
        array[i] = a + i * h;
    }
}

void print_usage(const char *prog_name) {
    printf("Usage: %s <a> <b> <n> <mode> <func> [--complexity <input_file>] [--simpson <dim> <lower0> <upper0> <n0> ... <lowerN> <upperN> <nN> <func>] [--help]\n", prog_name);
}

void print_help(const char *prog_name) {
    print_usage(prog_name);
    printf("Arguments:\n");
    printf("  a     - Lower bound of the integral (double)\n");
    printf("  b     - Upper bound of the integral (double)\n");
    printf("  n     - Number of intervals (int)\n");
    printf("  mode  - Integration method (0: Simpson, 1: Rectangle, 2: Trapezoidal)\n");
    printf("  func  - Function to integrate (0: sin, 1: cos, 2: exp, 3: sqrt, 4: log)\n");
    printf("Options:\n");
    printf("  --complexity <input_file> - Measure the complexity using parameters from the input file\n");
    printf("  --simpson <dim> <lower0> <upper0> <n0> ... <lowerN> <upperN> <nN> <func> - Perform multi-dimensional Simpson integration\n");
    printf("  --help                    - Show this help message\n");
}

double exact_integral(double a, double b, int func) {
    switch (func) {
        case 0: return -cos(b) + cos(a); // sin(x)
        case 1: return sin(b) - sin(a);  // cos(x)
        case 2: return exp(b) - exp(a);  // exp(x)
        case 3: return (2.0 / 3.0) * (pow(b, 1.5) - pow(a, 1.5)); // sqrt(x)
        case 4: return b * log(b) - b - (a * log(a) - a); // log(x)
        default: return 0.0;
    }
}

int read_params(const char *filename, Params *params) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "File couldn't be opened: %s\n", filename);
        return -1;
    }

    int count = 0;
    while (fscanf(file, "%lf %lf %d %d %d", &params[count].a, &params[count].b, &params[count].n, &params[count].mode, &params[count].func) != EOF) {
        count++;
    }

    fclose(file);
    return count;
}

void least_squares(double *x, double *y, int n, double *a, double *b) {
    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
    for (int i = 0; i < n; i++) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xx += x[i] * x[i];
        sum_xy += x[i] * y[i];
    }
    *a = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    *b = (sum_y - (*a) * sum_x) / n;
}
