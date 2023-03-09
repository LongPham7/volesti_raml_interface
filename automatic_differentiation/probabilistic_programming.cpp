#include <iostream>

// Volesti include
#include "random_walks/random_walks.hpp"

// autodiff include
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
using namespace autodiff;

#include "probabilistic_programming.h"
#include "probability_distributions.h"
#include "runtime_data.h"

probability_distribution_statistical_aara::
    probability_distribution_statistical_aara(
        runtime_data_sample *runtime_data_, int num_samples_, int dim_)
    : runtime_data(runtime_data_), num_samples(num_samples_), dim(dim_) {}

var probability_distribution_statistical_aara::coefficient_log_pdf(
    const ArrayXvar &x) {
  double mu = 0;
  double sigma = 0.2;
  var cumulative_log_pdf = 0;
  for (auto i = 0; i != x.rows(); i++) {
    cumulative_log_pdf += gaussian_log_pdf(mu, sigma, x[i]);
  }
  return cumulative_log_pdf;
}

var probability_distribution_statistical_aara::cost_gap_log_pdf(
    const ArrayXvar &x) {
  double alpha = 1;
  double sigma = 6;
  var cumulative_log_pdf = 0;

  for (auto i = 0; i != num_samples; i++) {
    const runtime_data_sample &current_sample{runtime_data[i]};
    const int *array_cindices = current_sample.array_cindices;
    const double *potential_of_cindices = current_sample.potential_of_cindices;
    const int num_cindices = current_sample.num_cindices;

    const int *array_indices = current_sample.array_indices;
    const double *potential_of_indices = current_sample.potential_of_indices;
    const int num_indices = current_sample.num_indices;

    const double cost = current_sample.cost;

    ArrayXvar array_input_potential(num_cindices);
    for (auto j = 0; j != num_cindices; j++) {
      int index = array_cindices[j];
      double potential = potential_of_cindices[j];
      array_input_potential[j] = x[index] * potential;
    }

    ArrayXvar array_output_potential(num_indices);
    for (auto j = 0; j != num_indices; j++) {
      int index = array_indices[j];
      double potential = potential_of_indices[j];
      array_output_potential[j] = x[index] * potential;
    }

    var total_input_potential = x[0] + array_input_potential.sum();
    var total_output_potential = array_output_potential.sum();

    cumulative_log_pdf += weibull_log_pdf(
        alpha, sigma, total_input_potential - total_output_potential - cost);
  }
  return cumulative_log_pdf;
}

var probability_distribution_statistical_aara::log_pdf(const ArrayXvar &x) {
  assert(x.rows() == dim);
  return coefficient_log_pdf(x) + cost_gap_log_pdf(x);
}

double probability_distribution_statistical_aara::log_pdf_point_interface(
    const Point &point_x) {
  ArrayXvar x{point_x.getCoefficients()};
  var y = log_pdf(x);
  return (double)y;
}

Eigen::VectorXd probability_distribution_statistical_aara::gradient_log_pdf(
    ArrayXvar x) {
  assert(x.rows() == dim);
  var y = log_pdf(x);
  return gradient(y, x);
}

Point probability_distribution_statistical_aara::
    gradient_log_pdf_point_interface(const Point &point_x) {
  ArrayXvar x{point_x.getCoefficients()};
  Eigen::VectorXd dydx = gradient_log_pdf(x);
  Point point_dydx{dydx};
  return point_dydx;
}

void test_automatic_differentiation() {
  runtime_data_sample *runtime_data_for_testing =
      create_runtime_data_for_testing();
  probability_distribution_statistical_aara
      probability_distribution_for_testing{runtime_data_for_testing, 5, 3};

  ArrayXvar x(3);
  x << 0.01, 1.1, 0.1;
  var y = probability_distribution_for_testing.log_pdf(x);
  Eigen::VectorXd dydx =
      probability_distribution_for_testing.gradient_log_pdf(x);
  std::cout << "y = " << y << std::endl;
  std::cout << "dy/dx = \n" << dydx << std::endl;
}
