#ifndef INPUT_UTILS_H
#define INPUT_UTILS_H

void get_points(double array[], double a, double b, int size, double h);
void print_usage(const char *prog_name);
void print_help(const char *prog_name);
double exact_integral(double a, double b, int func);

#endif // INPUT_UTILS_H
