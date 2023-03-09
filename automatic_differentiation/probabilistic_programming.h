#ifndef PROBABILISTIC_PROGRAMMING
#define PROBABILISTIC_PROGRAMMING

// Volesti include
#include "random_walks/random_walks.hpp"

// autodiff include
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
using namespace autodiff;

#include "runtime_data.h"

typedef double NT;
typedef Cartesian<NT> Kernel;
typedef typename Kernel::Point Point;

struct probability_distribution_statistical_aara {
  runtime_data_sample *runtime_data;
  int num_samples;
  int dim;

  probability_distribution_statistical_aara(runtime_data_sample *runtime_data_,
                                            int num_samples_, int dim_);

  var coefficient_log_pdf(const ArrayXvar &x);
  var cost_gap_log_pdf(const ArrayXvar &x);
  var log_pdf(const ArrayXvar &x);
  double log_pdf_point_interface(const Point &point_x);
  Eigen::VectorXd gradient_log_pdf(ArrayXvar x);
  Point gradient_log_pdf_point_interface(const Point &point_x);
};

void test_automatic_differentiation(void);

#endif
