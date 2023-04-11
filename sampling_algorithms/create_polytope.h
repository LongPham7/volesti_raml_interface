#ifndef CREATE_POLYTOPE
#define CREATE_POLYTOPE

#include "generators/known_polytope_generators.h"
#include "random_walks/random_walks.hpp"

typename HPolytope<typename Cartesian<double>::Point>::MT create_matrix_A(
    int const &num_rows, int const &num_cols, double *coefficients);

typename HPolytope<typename Cartesian<double>::Point>::VT create_vector_b(
    int const &size, double *coefficients);

HPolytope<typename Cartesian<double>::Point> create_polytope(
    int const &num_rows, int const &num_cols, double *coefficients_A,
    double *coefficients_b);

#endif
