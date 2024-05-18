inline double integrableFunction(double x, int func) {
    switch (func) {
        case 0: return sin(x);
        case 1: return cos(x);
        case 2: return exp(x);
        case 3: return sqrt(x);
        default: return log(x);
    }
}

__kernel void simpson_integral(__global double* x, __global double* results, int n, int func, double a, double b) {
    int gid = get_global_id(0);
    double local_result = 0.0;

    if (gid == 0) {
        local_result = integrableFunction(a, func);
    } else if (gid == n) {
        local_result = integrableFunction(b, func);
    } else if (gid % 2 == 1) {
        local_result = 4 * integrableFunction(x[gid], func);
    } else if (gid % 2 == 0) {
        local_result = 2 * integrableFunction(x[gid], func);
    }

    results[gid] = local_result;
}

__kernel void rectangle_integral(__global double* x, __global double* results, int n, int func) {
    int gid = get_global_id(0);

    if (gid < n) {
        results[gid] = integrableFunction(x[gid], func);
    }
}

__kernel void trapezoidal_integral(__global double* x, __global double* results, int n, int func) {
    int gid = get_global_id(0);

    if (gid < n) {
        results[gid] = ((x[gid+1] - x[gid]) / 2) * (integrableFunction(x[gid], func) + integrableFunction(x[gid+1], func));
    }
}
