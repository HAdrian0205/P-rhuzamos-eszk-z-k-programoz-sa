#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

void print_usage(const char *prog_name) {
    printf("Usage: %s <a> <b> <n> <mode> <func>\n", prog_name);
}

void print_help(const char *prog_name) {
    print_usage(prog_name);
    printf("Arguments:\n");
    printf("  a     - Lower bound of the integral (double)\n");
    printf("  b     - Upper bound of the integral (double)\n");
    printf("  n     - Number of intervals (int)\n");
    printf("  mode  - Integration method (0: Simpson, 1: Rectangle, 2: Trapezoidal)\n");
    printf("  func  - Function to integrate (0: sin, 1: cos, 2: exp, 3: sqrt, 4: log)\n");
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

double get_elapsed_time(struct timespec start, struct timespec end) {
    double start_sec = start.tv_sec + start.tv_nsec / 1.0e9;
    double end_sec = end.tv_sec + end.tv_nsec / 1.0e9;
    return end_sec - start_sec;
}

double integrableFunction(double x, int func)
{
    switch(func) {
        case 0: return sin(x);
        case 1: return cos(x);
        case 2: return exp(x);
        case 3: return sqrt(x);
        default: return log(x);
    }
}

double calculate_integral(double array_x[], double h, int size, int mode, int func, double a, double b)
{
    int n = size;
    int i;
    double sum = 0.0;

    if (mode == 0){ // simpson
        for (i = 1; i <= n-1; i++) {
            if(i % 2 != 0) {
                sum += 4 * integrableFunction(array_x[i], func);
            } else {
                sum += 2 * integrableFunction(array_x[i], func);
            }
        }
        return h/3 * (integrableFunction(a, func) + sum + integrableFunction(b, func));
        
    } else if (mode == 1) { // rectangle
        for (i = 0; i <= n; i++) {
            if(i == 0) {
                sum += integrableFunction(a, func);
            } else if (i == n) {
                sum += integrableFunction(b, func);
            } else if (i < n) {
                sum += integrableFunction(array_x[i], func);
            }
        }
        return h * sum;

    } else if (mode == 2) { // trapezoidal
        for (i = 0; i <= n; i++) {
            if(i == 0) {
                sum += integrableFunction(a, func);
            } else if (i == n) {
                sum += integrableFunction(b, func);
            } else if (i < n) {
                sum += 2 * integrableFunction(array_x[i], func);
            }
        }
        return h/2 * sum;
    } else {
        return -1;
    }
}

void get_points(double** array, double a, double b, int size, double h)
{
    *array = (double*)malloc(size * sizeof(double));
    if (*array == NULL) {
        printf("Error with memory allocation!\n");
        exit(1);
    }

    for (int i = 0; i < size; i++) {
        (*array)[i] = a + i * h;
    }
}

int main(int argc, char *argv[]) 
{
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
    double* x;

    double h = (b - a) / size;

    get_points(&x, a, b, size, h);

    struct timespec start_t, end_t;

    // Start time
    clock_gettime(CLOCK_MONOTONIC, &start_t);
    double integral = calculate_integral(x, h, size, mode, func, a, b);
    // End time
    clock_gettime(CLOCK_MONOTONIC, &end_t);

    double elapsed_time = get_elapsed_time(start_t, end_t);

    printf("Az integral erteke: %.10f\n", integral);
    printf("Eltelt ido: %.10f\n", elapsed_time);

    free(x);

    return 0;
}
