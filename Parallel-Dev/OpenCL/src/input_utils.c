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
