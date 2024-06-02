#ifndef INPUT_UTILS_H
#define INPUT_UTILS_H

typedef struct {
    double a;
    double b;
    int n;
    int mode;
    int func;
} Params;

void get_points(double array[], double a, double b, int size, double h);
void print_usage(const char *prog_name);
void print_help(const char *prog_name);
double exact_integral(double a, double b, int func);
int read_params(const char *filename, Params *params);
void least_squares(double *x, double *y, int n, double *a, double *b);

#endif // INPUT_UTILS_H
