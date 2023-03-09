#ifndef PROBABILITY_DISTRIBUTIONS
#define PROBABILITY_DISTRIBUTIONS

#include <autodiff/reverse/var.hpp>
using namespace autodiff;

var weibull_log_pdf(var alpha, var sigma, var x);
var gaussian_log_pdf(var mu, var sigma, var x);
void test_weibull(void);
void test_gaussian(void);

#endif
