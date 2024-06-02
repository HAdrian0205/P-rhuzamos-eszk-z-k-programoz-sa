__kernel void final_sum_kernel(__global double* input, __global double* output, __local double* local_mem) {
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    int local_size = get_local_size(0);

    local_mem[local_id] = input[global_id];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            local_mem[local_id] += local_mem[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        output[get_group_id(0)] = local_mem[0];
    }
}

__kernel void sin_kernel(__global double* x, __global double* results) {
    int gid = get_global_id(0);
    results[gid] = sin(x[gid]);
}

__kernel void cos_kernel(__global double* x, __global double* results) {
    int gid = get_global_id(0);
    results[gid] = cos(x[gid]);
}

__kernel void exp_kernel(__global double* x, __global double* results) {
    int gid = get_global_id(0);
    results[gid] = exp(x[gid]);
}

__kernel void sqrt_kernel(__global double* x, __global double* results) {
    int gid = get_global_id(0);
    results[gid] = sqrt(x[gid]);
}

__kernel void log_kernel(__global double* x, __global double* results) {
    int gid = get_global_id(0);
    results[gid] = log(x[gid]);
}

__kernel void simpson_integral(__global double* results, __global double* integral_results, int n) {
    int gid = get_global_id(0);
    double local_result = 0.0;

    if (gid == 0) {
        local_result = results[0];
    } else if (gid == n) {
        local_result = results[n];
    } else if (gid % 2 == 1) {
        local_result = 4 * results[gid];
    } else if (gid % 2 == 0) {
        local_result = 2 * results[gid];
    }

    integral_results[gid] = local_result;
}

__kernel void rectangle_integral(__global double* results, __global double* integral_results, int n) {
    int gid = get_global_id(0);
    double local_result = 0.0;

    if (gid == 0) {
        local_result = results[0];
    } else if (gid == n) {
        local_result = results[n];
    } else if (gid < n) {
        local_result = results[gid];
    }

    integral_results[gid] = local_result;
}

__kernel void trapezoidal_integral(__global double* results, __global double* integral_results, int n) {
    int gid = get_global_id(0);
    double local_result = 0.0;

    if (gid == 0) {
        local_result = results[0];
    } else if (gid == n) {
        local_result = results[n];
    } else if (gid < n) {
        local_result = 2 * results[gid];
    }

    integral_results[gid] = local_result;
}

__kernel void simpson_kernel(__global double* lower, __global double* upper, __global int* n, __global double* results, int dim, int func) {
    int gid = get_global_id(0);
    int total_points = 1;
    for (int i = 0; i < dim; i++) {
        total_points *= n[i] + 1;
    }

    __local double h[32];
    __local double x[32];
    __local int index[32];

    if (dim > 32) {
        return;
    }

    for (int i = 0; i < dim; i++) {
        h[i] = (upper[i] - lower[i]) / n[i];
    }

    int temp = gid;
    double coeff = 1.0;

    for (int j = 0; j < dim; j++) {
        index[j] = temp % (n[j] + 1);
        temp /= (n[j] + 1);
        x[j] = lower[j] + index[j] * h[j];
        if (index[j] == 0 || index[j] == n[j]) {
            coeff *= 1.0;
        } else if (index[j] % 2 == 1) {
            coeff *= 4.0;
        } else {
            coeff *= 2.0;
        }
    }

    double f_val;
    switch (func) {
        case 0:
            f_val = exp(-x[0] * x[0] - x[1] * x[1] - x[2] * x[2]);
            break;
        case 1:
            f_val = sin(x[0] + x[1] + x[2]);
            break;
        case 2:
            f_val = cos(x[0] * x[1] * x[2]);
            break;
        default:
            f_val = 1.0;
            break;
    }

    results[gid] = coeff * f_val;
}
