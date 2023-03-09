#ifndef HIT_AND_RUN
#define HIT_AND_RUN

#ifdef __cplusplus
extern "C" {
#endif

double *run_gaussian_rdhr(unsigned int const num_rows,
                          unsigned int const num_cols, double *coefficients_A,
                          double *coefficients_b, double const variance,
                          unsigned int const num_samples,
                          unsigned int const walk_length);

double *run_uniform_rdhr(unsigned int const num_rows,
                         unsigned int const num_cols, double *coefficients_A,
                         double *coefficients_b, unsigned int const num_samples,
                         unsigned int const walk_length);
#ifdef __cplusplus
}
#endif

#endif
