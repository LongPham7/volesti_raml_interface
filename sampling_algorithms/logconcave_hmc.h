#ifndef LOGCONCAVE_HMC
#define LOGCONCAVE_HMC

#include "runtime_data.h"

#ifdef __cplusplus
extern "C" {
#endif

double *hmc_function_pointer_interface(
    unsigned int const num_rows, unsigned int const num_cols,
    double *coefficients_A, double *coefficients_b, double const L,
    double const m, unsigned int const num_samples,
    unsigned int const walk_length, double const step_size,
    double *starting_point, double (*neg_log_prob)(const double *),
    double *(*neg_gradient_log_prob)(const double *));

double *hmc_runtime_data_interface(
    unsigned int const num_rows, unsigned int const num_cols,
    double *coefficients_A, double *coefficients_b, double const L,
    double const m, unsigned int const num_samples,
    unsigned int const walk_length, double const step_size,
    double *starting_point, runtime_data_sample *runtime_data,
    unsigned int const num_samples_in_runtime_data);

#ifdef __cplusplus
}
#endif

#endif
