#ifndef PROBABILISTIC_PROGRAMMING
#define PROBABILISTIC_PROGRAMMING

// Volesti include
#include "random_walks/random_walks.hpp"

// autodiff include
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
using namespace autodiff;

#include "probability_distributions.h"
#include "runtime_data.h"

typedef double NT;
typedef Cartesian<NT> Kernel;
typedef typename Kernel::Point Point;

struct probability_distribution_statistical_aara {
  runtime_data_sample *runtime_data;
  unsigned int num_samples;
  unsigned int dim;
  distribution_type coefficient_distribution;
  distribution_type cost_model;

  probability_distribution_statistical_aara(
      runtime_data_sample *runtime_data_, unsigned int num_samples_,
      unsigned int dim_, distribution_type coefficient_distribution_,
      distribution_type cost_model_);

  var coefficient_log_pdf(const ArrayXvar &x);
  var cost_gap_log_pdf(const ArrayXvar &x);
  var log_pdf(const ArrayXvar &x);
  double log_pdf_point_interface(const Point &point_x);
  Eigen::VectorXd gradient_log_pdf(ArrayXvar x);
  Point gradient_log_pdf_point_interface(Point point_x);
};

void test_automatic_differentiation(void);

#endif
