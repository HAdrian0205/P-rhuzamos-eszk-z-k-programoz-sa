#include <time.h>
#include "time_utils.h"

double get_elapsed_time(struct timespec start, struct timespec end) {
    double start_sec = start.tv_sec + start.tv_nsec / 1.0e9;
    double end_sec = end.tv_sec + end.tv_nsec / 1.0e9;
    return end_sec - start_sec;
}
