// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2020 Vissarion Fisikopoulos
// Copyright (c) 2018-2020 Apostolos Chalkis
// Copyright (c) 2020-2020 Marios Papachristou

// Modified by Long Pham

// Licensed under GNU LGPL.3, see LICENCE file

#include "logconcave_hmc.h"

#include <iostream>
#include <vector>

#include "create_polytope.h"
#include "ode_solvers/ode_solvers.hpp"
#include "random_walks/random_walks.hpp"

#define DEBUG

// HMCFunctor's code is attributed to the following source files in volesti's
// project directory: (i) volesti/examples/logconcave/simple_hmc.cpp and (ii)
// volesti/R-proj/src/oracle_functors_rcpp.h.
struct HMCFunctor {
  template <typename NT>
  struct parameters {
    unsigned int dim;    // Dimension
    unsigned int order;  // For HMC, the order is 2
    NT L;                // Lipschitz constant for gradient
    NT m;                // Strong convexity constant
    NT kappa;            // Condition number

    parameters(unsigned int dim_, NT L_, NT m_)
        // The value of kappa is set to L / m in
        // R-proj/src/oracle_functors_rcpp.h, so I adopt the same definition.
        : dim(dim_), order(2), L(L_), m(m_), kappa(L_ / m_){}
  };

  // Functor representing the negative log pdf of the prior probability
  // distribution in Bayesian inference. Note that it is "negative" log pdf,
  // rather than the ordinary, positive log pdf. This can be seen from how
  // FunctionFunctor is implemented in
  // volesti/R-proj/src/oracle_functors_rcpp.h, which serves as the R-interface
  // of volesti.
  template <typename Point>
  struct FunctionFunctor {
    typedef typename Point::FT NT;

    parameters<NT> &params;

    // Function pointer to the negative log pdf
    NT(*neg_log_pdf)
    (const NT *);

    FunctionFunctor(parameters<NT> &params_, NT (*neg_log_pdf_)(const NT *))
        : params(params_), neg_log_pdf(neg_log_pdf_){}

    // The index i represents the state vector index
    NT operator()(Point const &x) const {
      const NT *current_point_array = x.getCoefficients().data();
      return neg_log_pdf(current_point_array);
    }
  };

  // Functor representing the gradient of the (positive) log pdf of the prior
  // distribution. Bizarrely, this is the gradient of the "positive" log pdf,
  // while FunctionFunctor captures the "negative" log pdf. The fact that
  // GradientFunctor captures the positive log pdf's gradient can be seen from
  // volesti/R-proj/src/oracle_functors_rcpp.h, where GradientFunctor negates
  // the negative log pdf's gradient.
  template <typename Point>
  struct GradientFunctor {
    typedef typename Point::FT NT;
    typedef std::vector<Point> pts;

    parameters<NT> &params;

    // Function pointer for the gradient of the (positive) log pdf
    NT *(*gradient_log_pdf)(const NT *);

    GradientFunctor(parameters<NT> &params_,
                    NT *(*gradient_log_pdf_)(const NT *))
        : params(params_), gradient_log_pdf(gradient_log_pdf_){}

    // The index i represents the state vector index
    Point operator()(unsigned int const &i, pts const &xs, NT const &t) const {
      if (i == params.order - 1) {
        const NT *current_point_array = xs[0].getCoefficients().data();
        const NT *gradient_array = gradient_log_pdf(current_point_array);
        Point gradient_point(params.dim);
        for (auto i = 0; i != params.dim; i++) {
          gradient_point.set_coord(i, gradient_array[i]);
        }
        // If we run this file from OCaml and comment out the following line,
        // the double-free error happens. So the output (i.e. a pointer to an
        // array) of neg_gradient_log_pdf should not be deleted here by us, but
        // by the caller.
        // delete[] neg_gradient_array;
        return gradient_point;
      } else {
        return xs[i + 1];  // returns velocity (i.e. the derivative of x)
      }
    }
  };
};

double *run_hmc(unsigned int const num_rows, unsigned int const num_cols,
                double *coefficients_A, double *coefficients_b, double const L,
                double const m, unsigned int const num_samples,
                unsigned int const walk_length, double const step_size,
                double *starting_point, double (*neg_log_pdf)(const double *),
                double *(*gradient_log_pdf)(const double *)) {
  typedef double NT;
  typedef Cartesian<NT> Kernel;
  typedef typename Kernel::Point Point;
  typedef HPolytope<Point> Hpolytope;
  typedef typename Hpolytope::MT MT;
  typedef typename Hpolytope::VT VT;
  typedef BoostRandomNumberGenerator<boost::mt19937, NT> RandomNumberGenerator;

  unsigned int dim = num_cols;
  RandomNumberGenerator rng(dim);

  // Define a polytope
  Hpolytope P =
      create_polytope(num_rows, num_cols, coefficients_A, coefficients_b);

  // It is important to compute the inner ball of P. It normalizes P via the
  // normalize() function. Without this normalization, HMC does work properly.
  std::pair<Point, NT> InnerBall = P.ComputeInnerBall();
  if (InnerBall.second < 0.0)
    throw std::invalid_argument("The linear program is infeasible");

  // Define functors for the negative log pdf and the gradient of the (positive)
  // log pdf
  typedef HMCFunctor::FunctionFunctor<Point> NegativeLogprobFunctor;
  typedef HMCFunctor::GradientFunctor<Point> GradientFunctor;
  HMCFunctor::parameters<NT> params(dim, L, m);
  NegativeLogprobFunctor f(params, neg_log_pdf);
  GradientFunctor F(params, gradient_log_pdf);

  // Define a starting point
  Point x0(dim);
  for (auto i = 0; i != dim; i++) {
    x0.set_coord(i, starting_point[i]);
  }

  if (P.is_in(x0) == 0) {
    std::cerr << "Starting point of reflective HMC: "
              << x0.getCoefficients().transpose() << std::endl;
    throw std::invalid_argument(
        "The starting point is not in the interior of the polytope.");
  }

#ifdef DEBUG
  // Test neg_log_pdf and neg_gradient_log_pdf at the starting point
  double neg_log_pdf_result = neg_log_pdf(starting_point);
  double *gradient_log_pdf_result = gradient_log_pdf(starting_point);
  Point gradient_log_pdf_result_point(dim);
  for (auto i = 0; i != dim; i++) {
    gradient_log_pdf_result_point.set_coord(i, gradient_log_pdf_result[i]);
  }
  std::cout << "Neg log pdf at the starting point: " << neg_log_pdf_result
            << std::endl;
  std::cout << "Gradient at the starting point: "
            << gradient_log_pdf_result_point.getCoefficients().transpose()
            << std::endl;
#endif

  // Define HMC walk
  HamiltonianMonteCarloWalk::parameters<NT, GradientFunctor> hmc_params(F, dim);
  hmc_params.eta = step_size;
  typedef LeapfrogODESolver<Point, NT, Hpolytope, GradientFunctor> Solver;
  HamiltonianMonteCarloWalk::Walk<Point, Hpolytope, RandomNumberGenerator,
                                  GradientFunctor, NegativeLogprobFunctor,
                                  Solver>
      hmc_walk(&P, x0, F, f, hmc_params);

  // hmc.disable_adaptive();

  unsigned int num_burns = num_samples / 2;  // Half will be burned
  unsigned int num_samples_after_burns = num_samples - num_burns;

  // Samples drawn from the HMC sampler are stored in array_samples.
  double *array_samples = new double[num_samples_after_burns * dim];

  // Perform HMC
  for (int i = 0; i < num_samples; i++) {
    hmc_walk.apply(rng, walk_length);
    if (i >= num_burns) {
      const typename Point::Coeff sample = hmc_walk.x.getCoefficients();
      for (auto j = 0; j != dim; j++) {
        array_samples[(i - num_burns) * dim + j] = sample(j);
      }
    }
  }

#ifdef DEBUG
  std::cerr << "Reflective HMC statistics:" << std::endl;
  std::cerr << "Step size (final): " << hmc_walk.solver->eta << std::endl;
  std::cerr << "Discard Ratio: " << hmc_walk.discard_ratio << std::endl;
  std::cerr << "Average Acceptance Log-prob: "
            << exp(hmc_walk.average_acceptance_log_prob) << std::endl;
#endif

  return array_samples;
}