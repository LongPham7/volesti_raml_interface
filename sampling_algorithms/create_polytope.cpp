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

MT create_matrix_A(unsigned int const &num_rows, unsigned int const &num_cols,
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

VT create_vector_b(unsigned int const &size, double *coefficients) {
  VT b;
  b.resize(size);
  for (auto i = 0; i != size; i++) {
    b(i) = coefficients[i];
  }
  return b;
}

Hpolytope create_polytope(unsigned int const &num_rows,
                          unsigned int const &num_cols, double *coefficients_A,
                          double *coefficients_b) {
  MT A = create_matrix_A(num_rows, num_cols, coefficients_A);
  VT b = create_vector_b(num_rows, coefficients_b);
  Hpolytope P(num_cols, A, b);
#ifdef DEBUG
  std::cerr << "Here is polytope P:";
  P.print();
#endif
  return P;
}