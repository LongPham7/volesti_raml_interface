#ifndef LOGCONCAVE_HMC
#define LOGCONCAVE_HMC

#ifdef __cplusplus
extern "C" {
#endif

double *run_hmc(unsigned int const num_rows, unsigned int const num_cols,
                double *coefficients_A, double *coefficients_b, double const L,
                double const m, unsigned int const num_samples,
                unsigned int const walk_length, double const step_size,
                double *starting_point, double (*neg_log_prob)(const double *),
                double *(*neg_gradient_log_prob)(const double *));

#ifdef __cplusplus
}
#endif

#endif
