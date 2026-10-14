#include <pybind11/pybind11.h>
#include <tuple>
#include "ik_solver/solver.h"
#include "ik_solver/vec2.h"

PYBIND11_MODULE(ik2d, m) {
    m.doc() = "Python bindings for my 2D IK solver";

    m.def("is_possible", [](double l1, double l2, double x, double y) {
        return is_possible(l1, l2, Vector{x, y});
    });

    m.def("solve", [](double l1, double l2, double x, double y) {
        Vector t{x, y};
        double alpha = angle_calc(l1, l2, t, 'a');
        double beta = angle_calc(l1, l2, t, 'b');
        int side = static_cast<int>(angle_calc(l1, l2, t, 'c'));
        return std::make_tuple(alpha, beta, side);
    });
}