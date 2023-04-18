#include "create_polytope.h"

#include <iostream>

#include "generators/known_polytope_generators.h"

// #define DEBUG

typedef double NT;
typedef Cartesian<NT> Kernel;
typedef typename Kernel::Point Point;
typedef HPolytope<Point> Hpolytope;
typedef typename Hpolytope::MT MT;
typedef typename Hpolytope::VT VT;

MT create_matrix_A(int const &num_rows, int const &num_cols,
                   double *coefficients) {
  MT A;
  A.resize(num_rows, num_cols);

  for (auto i = 0; i != num_rows; i++) {
    for (auto j = 0; j != num_cols; j++) {
      A(i, j) = coefficients[i * num_cols + j];
    }
  }
  return A;
}

VT create_vector_b(int const &size, double *coefficients) {
  VT b;
  b.resize(size);
  for (auto i = 0; i != size; i++) {
    b(i) = coefficients[i];
  }
  return b;
}

Hpolytope create_polytope(int const &num_rows, int const &num_cols,
                          double *coefficients_A, double *coefficients_b) {
  MT A = create_matrix_A(num_rows, num_cols, coefficients_A);
  VT b = create_vector_b(num_rows, coefficients_b);
  Hpolytope P(num_cols, A, b);
#ifdef DEBUG
  std::cerr << "Here is polytope P:";
  P.print();
#endif
  return P;
}

double compute_chebyshev_radius(int const num_rows, int const num_cols,
                                double *coefficients_A,
                                double *coefficients_b) {
  Hpolytope P =
      create_polytope(num_rows, num_cols, coefficients_A, coefficients_b);
  std::pair<Point, NT> InnerBall = P.ComputeInnerBall();
  if (InnerBall.second < 0.0) {
    throw std::invalid_argument("The linear program is infeasible");
  }
#ifdef DEBUG
  std::cout << "Polytope P's Chebyshev raidus: " << InnerBall.second
            << std::endl;
#endif
  return InnerBall.second;
}
