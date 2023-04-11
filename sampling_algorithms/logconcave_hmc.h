#ifndef LOGCONCAVE_HMC
#define LOGCONCAVE_HMC

#include "probability_distributions.h"
#include "runtime_data.h"

#ifdef __cplusplus
extern "C" {
#endif

double *hmc_function_pointer_interface(
    int const num_rows, int const num_cols, double *coefficients_A,
    double *coefficients_b, double const L, double const m,
    int const num_samples, int const walk_length, double const step_size,
    double *starting_point, double (*neg_log_prob)(const double *),
    double *(*neg_gradient_log_prob)(const double *));

double *hmc_runtime_data_interface(
    int const num_rows, int const num_cols, double *coefficients_A,
    double *coefficients_b, double const L, double const m,
    int const num_samples, int const walk_length, double const step_size,
    double *starting_point, runtime_data_sample *runtime_data,
    int const num_samples_in_runtime_data,
    distribution_type coefficient_distribution, distribution_type cost_model);

#ifdef __cplusplus
}
#endif

#endif
