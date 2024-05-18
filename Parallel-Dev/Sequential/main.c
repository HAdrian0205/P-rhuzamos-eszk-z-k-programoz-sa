#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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
    int i,j;
    double sum = 0.0;

    if (mode == 0){ // simpson
        for (i = 1; i <= n-1; i++) {
            if(i % 2 != 0) {
                sum += 4*integrableFunction(array_x[i], func);
            } else {
                sum += 2*integrableFunction(array_x[i], func);
            }
        }
        return h/3 * (integrableFunction(a, func) + sum + integrableFunction(b, func));
        
    } else if (mode == 1) { // rectangle
        for (i = 0; i < n; i++) {
            sum += integrableFunction(array_x[i], func);
        }
        return h*sum;

    } else if (mode == 2) { // trapezoidal
        for (i = 0; i < n; i++) {
            sum += ((array_x[i+1] - array_x[i]) / 2) * (integrableFunction(array_x[i], func) + integrableFunction(array_x[i+1], func));
        }
        return sum;
    } else {
        return -1;
    }
}

void get_size(double* a, double* b, int* size, double* h)
{
    printf("Adja meg az intervallumot: \n");
    printf("Min(a): ");
    scanf("%lf", a);
    printf("Max(b): ");
    scanf("%lf", b);

    printf("Adja meg az n-et: ");
    scanf("%d", size);
    
    *h = (*b-*a) / *size;
}

void get_points(double array[], double a, double b, int size, double h)
{
    int i = 0;

    for(i = 0; i < size; i++) {
        array[i] = a+i*h;
    }
}

int main() 
{
    int size;
    int mode, func;
    double a, b, h;

    get_size(&a, &b, &size, &h);

    double x[size];

    get_points(x, a, b, size, h);

    printf("Valassza ki, hogy melyik szabaly szerint integralna:\n");
    printf("0 - Osszetett simpson, 1 - Osszetett teglalap, 2 - Osszetett trapez\n");
    scanf("%d", &mode);
    printf("Valassza ki, hogy melyik fuggvenyt integralna:\n");
    printf("0 - sin, 1 - cos, 2 - exp, 3 - sqrt, 4 - log\n");
    scanf("%d", &func);

    clock_t start_t, end_t;
    start_t = clock();
    double integral = calculate_integral(x, h, size, mode, func, a, b);
    end_t = clock();

    double elapsed_time = (double)(end_t - start_t) / CLOCKS_PER_SEC;

    printf("Az integral erteke: %.10f\n", integral);
    printf("Eltelt ido: %.10f\n", elapsed_time);
    return 0;
}