#include "hit_and_run.h"

#include <iostream>

#include "create_polytope.h"
#include "random_walks/random_walks.hpp"

double *run_gaussian_rdhr(unsigned int const num_rows,
                          unsigned int const num_cols, double *coefficients_A,
                          double *coefficients_b, double const variance,
                          unsigned int const num_samples,
                          unsigned int const walk_length) {
  typedef double NT;
  typedef Cartesian<NT> Kernel;
  typedef typename Kernel::Point Point;
  typedef HPolytope<Point> Hpolytope;
  typedef BoostRandomNumberGenerator<boost::mt19937, NT> RandomNumberGenerator;

  unsigned int dim = num_cols;
  RandomNumberGenerator rng(dim);

  // Define a polytope
  Hpolytope P =
      create_polytope(num_rows, num_cols, coefficients_A, coefficients_b);

  // Check the feasibility of the linear program by computing the inner ball of
  // P
  std::pair<Point, NT> InnerBall = P.ComputeInnerBall();
  if (InnerBall.second < 0.0)
    throw std::invalid_argument("The linear program is infeasible");

  // Point x, which tracks the current sample, is initialized to the center of
  // the inner ball.
  Point x = InnerBall.first;

  // During the sampling, once we choose a chord inside the polytope P, we draw
  // a sample from the (unnormalized) Gaussian distribution exp(- (1 / (2 *
  // variance)) ||x||^2) on the chord. The mode of the distribution is zero by
  // default. If we want to use a different mode, we should shift the entire
  // polytope.
  GaussianRDHRWalk::Walk<Hpolytope, RandomNumberGenerator> gaussian_rdhr_walk(
      P, x, 1 / (2 * variance), rng);

  // Samples drawn from the Gaussian RDHR sampler are stored in array_samples.
  double *array_samples = new double[num_samples * dim];

  // Perform Gaussian RDHR. If the variance is too small, we can only break the
  // infinite loop (i.e. the loop with the loop guard being true) in
  // volesti/include/random_walks/gaussian_rdhr_walk.hpp with an extremely low
  // probability. As a result, the apply function of Gaussian RDHR may never
  // terminates. This happens, for example, when we use the variance of 1.0 in
  // the Gaussian RDHR for warmup in the pure Bayesian resource analysis of the
  // append function. 
  for (int i = 0; i < num_samples; i++) {
    gaussian_rdhr_walk.apply(P, x, 1 / (2 * variance), walk_length, rng);
    const typename Point::Coeff sample = x.getCoefficients();
    for (auto j = 0; j != dim; j++) {
      array_samples[i * dim + j] = sample(j);
    }
  }

  return array_samples;
}

double *run_uniform_rdhr(unsigned int const num_rows,
                         unsigned int const num_cols, double *coefficients_A,
                         double *coefficients_b, unsigned int const num_samples,
                         unsigned int const walk_length) {
  typedef double NT;
  typedef Cartesian<NT> Kernel;
  typedef typename Kernel::Point Point;
  typedef HPolytope<Point> Hpolytope;
  typedef BoostRandomNumberGenerator<boost::mt19937, NT> RandomNumberGenerator;

  unsigned int dim = num_cols;
  RandomNumberGenerator rng(dim);

  // Define a polytope
  Hpolytope P =
      create_polytope(num_rows, num_cols, coefficients_A, coefficients_b);

  // Check the feasibility of the linear program by computing the inner ball of
  // P
  std::pair<Point, NT> InnerBall = P.ComputeInnerBall();
  if (InnerBall.second < 0.0)
    throw std::invalid_argument("The linear program is infeasible");

  // Point x, which tracks the current sample, is initialized to the center of
  // the inner ball.
  Point x = InnerBall.first;

  RDHRWalk::Walk<Hpolytope, RandomNumberGenerator> uniform_rdhr_walk(P, x, rng);

  // Samples drawn from the uniform RDHR sampler are stored in array_samples.
  double *array_samples = new double[num_samples * dim];

  // Perform uniform RDHR
  for (int i = 0; i < num_samples; i++) {
    uniform_rdhr_walk.apply(P, x, walk_length, rng);
    const typename Point::Coeff sample = x.getCoefficients();
    for (auto j = 0; j != dim; j++) {
      array_samples[i * dim + j] = sample(j);
    }
  }

  return array_samples;
}
