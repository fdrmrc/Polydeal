// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) XXXX - YYYY by the polyDEAL authors
//
// This file is part of the polyDEAL library.
//
// Detailed license information governing the source code
// can be found in LICENSE.md at the top level directory.
//
// -----------------------------------------------------------------------------

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/point.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "continuous_agglo_utils.h"

static constexpr double       tolerance = 1e-12;
static constexpr unsigned int dim       = 2;
using namespace dealii;

int
main()
{
  static constexpr unsigned int min_elem_per_node = 2;
  static constexpr unsigned int max_elem_per_node = 4;
  static constexpr unsigned int mg_levels         = 3;
  static constexpr bool         skip_leaves       = true;

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0., 1.);
  tria.refine_global(4);

  DoFHandler<dim> dof_handler(tria);
  FE_Q<dim>       fe(1);
  dof_handler.distribute_dofs(fe);

  std::vector<Point<dim>> support_points_vector(dof_handler.n_dofs());
  MappingQ1<dim>          mapping;

  DoFTools::map_dofs_to_support_points(mapping,
                                       dof_handler,
                                       support_points_vector);

  // Point agglo test
  {
    std::vector<SparseMatrix<double>> injection_matrices;
    std::vector<SparsityPattern>      injection_sparsity_patterns;
    std::vector<unsigned int>         coarse_space_degrees(mg_levels - 1, 1);
    std::vector<std::unique_ptr<Triangulation<dim>>> triangulations;
    std::vector<std::unique_ptr<DoFHandler<dim>>>    support_dof_handlers;

    ContinuousAggloUtils::PointsAgglo::
      agglomerate_and_compute_injection_matrices<dim,
                                                 min_elem_per_node,
                                                 max_elem_per_node>(
        support_points_vector,
        skip_leaves,
        injection_matrices,
        injection_sparsity_patterns,
        mg_levels,
        coarse_space_degrees,
        triangulations,
        support_dof_handlers);

    std::cout << "Built " << injection_matrices.size() << " injection matrices"
              << std::endl;

    for (unsigned int level = 0; level < injection_matrices.size(); ++level)
      {
        const auto &mat = injection_matrices[level];

        std::vector<Point<dim>> coarse_support_points(
          support_dof_handlers[level]->n_dofs());
        DoFTools::map_dofs_to_support_points(mapping,
                                             *support_dof_handlers[level],
                                             coarse_support_points);

        AssertThrow(mat.n() == coarse_support_points.size(),
                    ExcDimensionMismatch(mat.n(),
                                         coarse_support_points.size()));

        Vector<double> coarse_values(coarse_support_points.size());
        for (unsigned int i = 0; i < coarse_values.size(); ++i)
          coarse_values[i] =
            coarse_support_points[i][0] + coarse_support_points[i][1];

        if (level < mg_levels - 2)
          {
            std::vector<Point<dim>> fine_support_points(
              support_dof_handlers[level + 1]->n_dofs());
            DoFTools::map_dofs_to_support_points(
              mapping, *support_dof_handlers[level + 1], fine_support_points);

            AssertThrow(mat.m() == fine_support_points.size(),
                        ExcDimensionMismatch(mat.m(),
                                             fine_support_points.size()));

            Vector<double> fine_values(fine_support_points.size());
            for (unsigned int i = 0; i < fine_values.size(); ++i)
              fine_values[i] =
                fine_support_points[i][0] + fine_support_points[i][1];

            Vector<double> transferred_values(fine_values.size());
            mat.vmult(transferred_values, coarse_values);

            transferred_values -= fine_values;
            const double l2_error = transferred_values.l2_norm();

            AssertThrow(l2_error < tolerance,
                        ExcMessage(
                          "Agglomerated transfer " + std::to_string(level) +
                          " (coarse->fine) failed linear test with error " +
                          std::to_string(l2_error)));

            std::cout << "Level " << level << " (" << mat.m() << " x "
                      << mat.n() << "): L2 error = " << l2_error << std::endl;
          }
        else
          {
            AssertThrow(mat.m() == support_points_vector.size(),
                        ExcDimensionMismatch(mat.m(),
                                             support_points_vector.size()));

            Vector<double> fine_values(support_points_vector.size());
            for (unsigned int i = 0; i < fine_values.size(); ++i)
              fine_values[i] =
                support_points_vector[i][0] + support_points_vector[i][1];

            Vector<double> transferred_values(fine_values.size());
            mat.vmult(transferred_values, coarse_values);

            transferred_values -= fine_values;
            const double l2_error = transferred_values.l2_norm();

            AssertThrow(
              l2_error < tolerance,
              ExcMessage(
                "Final transfer (agglo->original) failed linear test with error " +
                std::to_string(l2_error)));

            std::cout << "Level " << level << " (" << mat.m() << " x "
                      << mat.n() << "): L2 error = " << l2_error
                      << " (agglo->original)" << std::endl;
          }
      }

    std::cout << "All point-agglo continuous transfer tests: OK" << std::endl;
  }

  // Cell agglo test
  {
    std::vector<SparseMatrix<double>> injection_matrices;
    std::vector<SparsityPattern>      injection_sparsity_patterns;
    std::vector<unsigned int>         coarse_space_degrees(mg_levels - 1, 1);
    std::vector<std::unique_ptr<Triangulation<dim>>> triangulations;
    std::vector<std::unique_ptr<DoFHandler<dim>>>    support_dof_handlers;

    ContinuousAggloUtils::CellsAgglo::
      agglomerate_and_compute_injection_matrices<dim,
                                                 min_elem_per_node,
                                                 max_elem_per_node>(
        dof_handler,
        mapping,
        skip_leaves,
        injection_matrices,
        injection_sparsity_patterns,
        mg_levels,
        coarse_space_degrees,
        triangulations,
        support_dof_handlers);

    std::cout << "Built " << injection_matrices.size() << " injection matrices"
              << std::endl;

    for (unsigned int level = 0; level < injection_matrices.size(); ++level)
      {
        const auto &mat = injection_matrices[level];

        std::vector<Point<dim>> coarse_support_points(
          support_dof_handlers[level]->n_dofs());
        DoFTools::map_dofs_to_support_points(mapping,
                                             *support_dof_handlers[level],
                                             coarse_support_points);

        AssertThrow(mat.n() == coarse_support_points.size(),
                    ExcDimensionMismatch(mat.n(),
                                         coarse_support_points.size()));

        Vector<double> coarse_values(coarse_support_points.size());
        for (unsigned int i = 0; i < coarse_values.size(); ++i)
          coarse_values[i] =
            coarse_support_points[i][0] + coarse_support_points[i][1];

        if (level < mg_levels - 2)
          {
            std::vector<Point<dim>> fine_support_points(
              support_dof_handlers[level + 1]->n_dofs());
            DoFTools::map_dofs_to_support_points(
              mapping, *support_dof_handlers[level + 1], fine_support_points);

            AssertThrow(mat.m() == fine_support_points.size(),
                        ExcDimensionMismatch(mat.m(),
                                             fine_support_points.size()));

            Vector<double> fine_values(fine_support_points.size());
            for (unsigned int i = 0; i < fine_values.size(); ++i)
              fine_values[i] =
                fine_support_points[i][0] + fine_support_points[i][1];

            Vector<double> transferred_values(fine_values.size());
            mat.vmult(transferred_values, coarse_values);

            transferred_values -= fine_values;
            const double l2_error = transferred_values.l2_norm();

            AssertThrow(l2_error < tolerance,
                        ExcMessage(
                          "Agglomerated transfer " + std::to_string(level) +
                          " (coarse->fine) failed linear test with error " +
                          std::to_string(l2_error)));

            std::cout << "Level " << level << " (" << mat.m() << " x "
                      << mat.n() << "): L2 error = " << l2_error << std::endl;
          }
        else
          {
            AssertThrow(mat.m() == support_points_vector.size(),
                        ExcDimensionMismatch(mat.m(),
                                             support_points_vector.size()));

            Vector<double> fine_values(support_points_vector.size());
            for (unsigned int i = 0; i < fine_values.size(); ++i)
              fine_values[i] =
                support_points_vector[i][0] + support_points_vector[i][1];

            Vector<double> transferred_values(fine_values.size());
            mat.vmult(transferred_values, coarse_values);

            transferred_values -= fine_values;
            const double l2_error = transferred_values.l2_norm();

            AssertThrow(
              l2_error < tolerance,
              ExcMessage(
                "Final transfer (agglo->original) failed linear test with error " +
                std::to_string(l2_error)));

            std::cout << "Level " << level << " (" << mat.m() << " x "
                      << mat.n() << "): L2 error = " << l2_error
                      << " (agglo->original)" << std::endl;
          }
      }

    std::cout << "All cell-agglo continuous transfer tests: OK" << std::endl;
  }
}
