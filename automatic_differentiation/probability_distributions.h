#ifndef PROBABILITY_DISTRIBUTIONS
#define PROBABILITY_DISTRIBUTIONS

#include <autodiff/reverse/var.hpp>
using namespace autodiff;

var weibull_log_pdf(const var& alpha, const var& sigma, const var& x);
var gaussian_log_pdf(const var& mu, const var& sigma, const var& x);
void test_weibull(void);
void test_gaussian(void);

#endif
