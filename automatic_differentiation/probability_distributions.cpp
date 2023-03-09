#include "probability_distributions.h"

#include <autodiff/reverse/var.hpp>
#include <iostream>
#include <limits>
using namespace autodiff;

// Constant log (sqrt (2 * pi)). It is used in the calculation of the Gaussian
// log pdf.
#define LOG_SQRT_TWO_PI 0.91893853320467267

var weibull_log_pdf(var alpha, var sigma, var x) {
  var negative_infinity = -std::numeric_limits<double>::infinity();
  var x_normalized = x / sigma;
  return condition(
      x < 0, negative_infinity,
      condition(x == 0, condition(alpha == 1, -log(sigma), negative_infinity),
                condition(alpha == 1, -(x / sigma) - log(sigma),
                          log(alpha) - log(sigma) - pow(x_normalized, alpha) +
                              (alpha - 1) * log(x_normalized))));
}

var gaussian_log_pdf(var mu, var sigma, var x) {
  var x_normalized = x / sigma;
  return -(x_normalized * x_normalized / 2) - LOG_SQRT_TWO_PI - log(sigma);
}

// Sanity check of a Weibull distribution
void test_weibull() {
  var alpha = 1;
  var sigma = 6;
  var x = 1.1;
  var y = weibull_log_pdf(alpha, sigma, x);
  auto [pdf_gradient] = derivatives(y, wrt(x));

  std::cout << "Log pdf of Weibull(alpha = " << alpha << ", sigma = " << sigma
            << ") at x = " << x << ": " << y << std::endl;
  std::cout << "Log pdf gradient of Weibull(alpha = " << alpha
            << ", sigma = " << sigma << ") at x = " << x << ": " << pdf_gradient
            << std::endl;
}

// Sanity check of a Gaussian distribution
void test_gaussian() {
  var mu = 0;
  var sigma = 2;
  var x = 1.1;
  var y = gaussian_log_pdf(mu, sigma, x);
  auto [pdf_gradient] = derivatives(y, wrt(x));

  std::cout << "Log pdf of Gaussian(mu = " << mu << ", sigma = " << sigma
            << ") at x = " << x << ": " << y << std::endl;
  std::cout << "Log pdf gradient of Gaussian(mu = " << mu
            << ", sigma = " << sigma << ") at x = " << x << ": " << pdf_gradient
            << std::endl;
}
