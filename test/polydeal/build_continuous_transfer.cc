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

static constexpr double       tolerance  = 1e-12;
static constexpr unsigned int dim        = 2;
static constexpr bool         use_points = true;
using namespace dealii;

int
main()
{
  namespace bgi                                   = boost::geometry::index;
  static constexpr unsigned int min_elem_per_node = 2;
  static constexpr unsigned int max_elem_per_node = 4;

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0., 1.);
  tria.refine_global(4);

  DoFHandler<dim> dof_handler(tria);
  FE_Q<dim>       fe(1);
  FE_DGQ<dim>     fe_dgq(1);
  dof_handler.distribute_dofs(fe);

  std::vector<Point<dim>> support_points_vector(dof_handler.n_dofs());
  MappingQ1<dim>          mapping;

  DoFTools::map_dofs_to_support_points(mapping,
                                       dof_handler,
                                       support_points_vector);

  auto tree =
    pack_rtree_of_indices<bgi::rstar<max_elem_per_node, min_elem_per_node>>(
      support_points_vector);

  std::cout << "R-tree built with " << n_levels(tree) << " levels "
            << std::endl;
  unsigned int                  fine_level   = n_levels(tree);
  unsigned int                  coarse_level = fine_level - 1;
  std::vector<BoundingBox<dim>> coarse_level_boxes;
  std::vector<BoundingBox<dim>> fine_level_boxes;

  Triangulation<dim> fine_tria;
  Triangulation<dim> coarse_tria;
  DoFHandler<dim>    fine_dof_handler(fine_tria);
  DoFHandler<dim>    coarse_dof_handler(coarse_tria);

  {
    CellsAgglomerator<dim, decltype(tree), use_points>      agglomerator{tree,
                                                                    fine_level};
    const std::vector<std::vector<types::global_dof_index>> agglomerates =
      agglomerator.extract_agglomerates();
    fine_level_boxes.reserve(agglomerates.size());
    for (const std::vector<types::global_dof_index> &agglo : agglomerates)
      {
        std::vector<Point<dim>> points_in_current_agglomerate;
        points_in_current_agglomerate.reserve(agglo.size());

        for (const auto &index : agglo)
          points_in_current_agglomerate.push_back(support_points_vector[index]);

        fine_level_boxes.emplace_back(points_in_current_agglomerate);
      }
    ContinuousAggloUtils::PointsAgglo::create_triangulation_from_bounding_boxes(
      fine_tria, fine_level_boxes);
    fine_dof_handler.distribute_dofs(fe_dgq);
  }

  SparseMatrix<double> transfer_matrix;
  SparsityPattern      sparsity_pattern;

  {
    CellsAgglomerator<dim, decltype(tree), use_points> agglomerator{
      tree, coarse_level};
    const std::vector<std::vector<types::global_dof_index>> agglomerates =
      agglomerator.extract_agglomerates();
    coarse_level_boxes.reserve(agglomerates.size());
    for (const std::vector<types::global_dof_index> &agglo : agglomerates)
      {
        std::vector<Point<dim>> points_in_current_agglomerate;
        points_in_current_agglomerate.reserve(agglo.size());

        for (const auto &index : agglo)
          points_in_current_agglomerate.push_back(support_points_vector[index]);

        coarse_level_boxes.emplace_back(points_in_current_agglomerate);
      }
    ContinuousAggloUtils::PointsAgglo::create_triangulation_from_bounding_boxes(
      coarse_tria, coarse_level_boxes);
    coarse_dof_handler.distribute_dofs(fe_dgq);

    const std::map<
      std::pair<types::global_cell_index, types::global_cell_index>,
      std::vector<types::global_cell_index>> &parent_to_child_info =
      agglomerator.get_hierarchy();

    std::vector<Point<dim>> coarse_support_points(coarse_dof_handler.n_dofs());
    std::vector<Point<dim>> fine_support_points(fine_dof_handler.n_dofs());

    MappingQ1<dim> mapping;
    DoFTools::map_dofs_to_support_points(mapping,
                                         coarse_dof_handler,
                                         coarse_support_points);
    DoFTools::map_dofs_to_support_points(mapping,
                                         fine_dof_handler,
                                         fine_support_points);

    ContinuousAggloUtils::PointsAgglo::fill_injection_matrix<dim>(
      coarse_dof_handler,
      fine_dof_handler,
      sparsity_pattern,
      transfer_matrix,
      parent_to_child_info,
      coarse_level_boxes,
      fine_level_boxes,
      coarse_level - 1);

    AssertThrow(transfer_matrix.m() == fine_dof_handler.n_dofs(),
                ExcDimensionMismatch(transfer_matrix.m(),
                                     fine_dof_handler.n_dofs()));
    AssertThrow(transfer_matrix.n() == coarse_dof_handler.n_dofs(),
                ExcDimensionMismatch(transfer_matrix.n(),
                                     coarse_dof_handler.n_dofs()));

    Vector<double> coarse_values(coarse_dof_handler.n_dofs());
    Vector<double> fine_values(fine_dof_handler.n_dofs());
    Vector<double> transferred_values(fine_dof_handler.n_dofs());

    for (unsigned int i = 0; i < coarse_values.size(); ++i)
      coarse_values[i] =
        coarse_support_points[i][0] + coarse_support_points[i][1];

    for (unsigned int i = 0; i < fine_values.size(); ++i)
      fine_values[i] = fine_support_points[i][0] + fine_support_points[i][1];

    transfer_matrix.vmult(transferred_values, coarse_values);

    transferred_values -= fine_values;
    const double l2_error = transferred_values.l2_norm();

    AssertThrow(l2_error < tolerance,
                ExcMessage(
                  "Continuous transfer matrix failed the linear field test."));

    std::cout << "Coarse DoFs: " << coarse_dof_handler.n_dofs() << std::endl;
    std::cout << "Fine DoFs: " << fine_dof_handler.n_dofs() << std::endl;
    std::cout << "Transfer matrix: " << transfer_matrix.m() << " x "
              << transfer_matrix.n() << std::endl;
    std::cout << "Linear transfer L2 error: " << l2_error << std::endl;
    std::cout << "Continuous transfer matrix test: OK" << std::endl;
  }

  // Check  that dimensions are coherent
  Assert(transfer_matrix.m() == fine_dof_handler.n_dofs(),
         ExcDimensionMismatch(transfer_matrix.m(), fine_dof_handler.n_dofs()));
}