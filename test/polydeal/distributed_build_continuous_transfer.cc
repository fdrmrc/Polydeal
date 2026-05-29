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

// Note for me: in this test the results are slightly different from the
// sequential one. It does make sense to me since the agglomeration is different
// since i have more than one tree and more boxes might come out since all boxes
// might not be filled up. Maybe check in with Marco too.

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "continuous_agglo_utils.h"

static constexpr double       tolerance  = 1e-12;
static constexpr unsigned int dim        = 2;
static constexpr bool         use_points = true;
using namespace dealii;

namespace
{
  template <int dim>
  class LinearField : public Function<dim>
  {
  public:
    LinearField()
      : Function<dim>()
    {}

    double
    value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      (void)component;
      double sum = 0.0;
      for (unsigned int d = 0; d < dim; ++d)
        sum += p[d];
      return sum;
    }
  };
} // namespace

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  const MPI_Comm                  &comm = MPI_COMM_WORLD;
  const unsigned int n_ranks            = Utilities::MPI::n_mpi_processes(comm);
  const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);

  AssertThrow(n_ranks == 3,
              ExcMessage(
                "This test is meant to be run with 3 MPI ranks only."));

  if (my_rank == 0)
    std::cout << "Running distributed continuous transfer test with " << n_ranks
              << " MPI ranks." << std::endl;

  namespace bgi                                   = boost::geometry::index;
  static constexpr unsigned int min_elem_per_node = 2;
  static constexpr unsigned int max_elem_per_node = 4;

  Triangulation<dim> starting_tria;
  GridGenerator::hyper_cube(starting_tria, 0.0, 1.0);
  starting_tria.refine_global(4);
  GridTools::partition_triangulation(n_ranks, starting_tria);

  const auto construction_data =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      starting_tria, comm);

  parallel::fullydistributed::Triangulation<dim, dim> starting_tria_pft(comm);
  starting_tria_pft.create_triangulation(construction_data);

  const unsigned int locally_owned_active_cells =
    starting_tria_pft.n_locally_owned_active_cells();
  const unsigned int global_active_cells =
    starting_tria_pft.n_global_active_cells();
  const unsigned int summed_local_active_cells =
    Utilities::MPI::sum(locally_owned_active_cells, comm);

  AssertThrow(
    summed_local_active_cells == global_active_cells,
    ExcMessage(
      "Sum of locally owned cells does not match the global cell count."));

  DoFHandler<dim> dof_handler(starting_tria_pft);
  FE_Q<dim>       fe(1);
  FE_DGQ<dim>     fe_dgq(1);
  dof_handler.distribute_dofs(fe);

  const unsigned int locally_owned_dofs_number =
    dof_handler.locally_owned_dofs().n_elements();
  const unsigned int global_dofs = dof_handler.n_dofs();
  const unsigned int summed_local_dofs =
    Utilities::MPI::sum(locally_owned_dofs_number, comm);

  AssertThrow(
    summed_local_dofs == global_dofs,
    ExcMessage(
      "Sum of locally owned DoFs does not match the global DoF count."));

  MappingQ1<dim>                                mapping;
  std::map<types::global_dof_index, Point<dim>> support_points =
    DoFTools::map_dofs_to_support_points(mapping, dof_handler);

  const IndexSet &locally_owned_dofs_set = dof_handler.locally_owned_dofs();
  std::vector<types::global_dof_index> local_dof_indices;
  std::vector<Point<dim>>              local_support_points_vector;
  local_support_points_vector.reserve(locally_owned_dofs_number);
  local_dof_indices.reserve(locally_owned_dofs_number);
  for (const auto &entry : support_points)
    if (locally_owned_dofs_set.is_element(entry.first))
      {
        local_dof_indices.push_back(entry.first);
        local_support_points_vector.push_back(entry.second);
      }

  AssertThrow(
    local_dof_indices.size() == locally_owned_dofs_number,
    ExcMessage(
      "Filtered support-point count does not match the locally owned DoF count."));

  auto local_tree =
    pack_rtree_of_indices<bgi::rstar<max_elem_per_node, min_elem_per_node>>(
      local_support_points_vector);

  const unsigned int fine_level   = n_levels(local_tree);
  const unsigned int coarse_level = fine_level - 1;

  std::vector<BoundingBox<dim>> fine_level_boxes;
  std::vector<BoundingBox<dim>> coarse_level_boxes;

  parallel::fullydistributed::Triangulation<dim, dim> fine_tria(comm);
  parallel::fullydistributed::Triangulation<dim, dim> coarse_tria(comm);
  DoFHandler<dim>                                     fine_dh(fine_tria);
  DoFHandler<dim>                                     coarse_dh(coarse_tria);

  {
    CellsAgglomerator<dim, decltype(local_tree), use_points> agglomerator{
      local_tree, fine_level};
    const auto agglomerates = agglomerator.extract_agglomerates();

    fine_level_boxes.reserve(agglomerates.size());
    for (const auto &agglo : agglomerates)
      {
        std::vector<Point<dim>> points_in_current_agglomerate;
        points_in_current_agglomerate.reserve(agglo.size());

        for (const auto &index : agglo)
          points_in_current_agglomerate.push_back(
            local_support_points_vector[index]);

        fine_level_boxes.emplace_back(points_in_current_agglomerate);
      }

    ContinuousAggloUtils::PointsAgglo::create_distributed_tria_from_local_boxes(
      fine_tria, fine_level_boxes, comm);

    const unsigned int fine_local_boxes = fine_level_boxes.size();
    const unsigned int fine_locally_owned_cells =
      fine_tria.n_locally_owned_active_cells();
    const unsigned int fine_global_cells = fine_tria.n_global_active_cells();
    const unsigned int fine_summed_local_cells =
      Utilities::MPI::sum(fine_locally_owned_cells, comm);

    AssertThrow(
      fine_locally_owned_cells == fine_local_boxes,
      ExcMessage(
        "Local fine boxes do not match the number of locally owned fine cells."));
    AssertThrow(
      fine_summed_local_cells == fine_global_cells,
      ExcMessage(
        "Sum of locally owned fine cells does not match the global fine cell count."));

    // This check might be useless, in a sense complements the check for
    // create_distributed_tria_from_local_boxes.
    for (const auto &cell : fine_tria.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const Point<dim> center = cell->center();
          const Point<dim> p_min  = cell->vertex(0);
          const Point<dim> p_max  = cell->vertex(3);
          AssertThrow(center[0] >= p_min[0] && center[0] <= p_max[0],
                      ExcMessage(
                        "Fine cell center is outside the expected x-range."));
          AssertThrow(center[1] >= p_min[1] && center[1] <= p_max[1],
                      ExcMessage(
                        "Fine cell center is outside the expected y-range."));
        }
    fine_dh.distribute_dofs(fe_dgq);
  }

  TrilinosWrappers::SparseMatrix    transfer_matrix;
  TrilinosWrappers::SparsityPattern sparsity_pattern;

  {
    CellsAgglomerator<dim, decltype(local_tree), use_points> agglomerator{
      local_tree, coarse_level};
    const auto agglomerates = agglomerator.extract_agglomerates();

    coarse_level_boxes.reserve(agglomerates.size());
    for (const auto &agglo : agglomerates)
      {
        std::vector<Point<dim>> points_in_current_agglomerate;
        points_in_current_agglomerate.reserve(agglo.size());

        for (const auto &index : agglo)
          points_in_current_agglomerate.push_back(
            local_support_points_vector[index]);

        coarse_level_boxes.emplace_back(points_in_current_agglomerate);
      }

    ContinuousAggloUtils::PointsAgglo::create_distributed_tria_from_local_boxes(
      coarse_tria, coarse_level_boxes, comm);

    const unsigned int coarse_local_boxes = coarse_level_boxes.size();
    const unsigned int coarse_locally_owned_cells =
      coarse_tria.n_locally_owned_active_cells();
    const unsigned int coarse_global_cells =
      coarse_tria.n_global_active_cells();
    const unsigned int coarse_summed_local_cells =
      Utilities::MPI::sum(coarse_locally_owned_cells, comm);

    AssertThrow(
      coarse_locally_owned_cells == coarse_local_boxes,
      ExcMessage(
        "Local coarse boxes do not match the number of locally owned coarse cells."));
    AssertThrow(
      coarse_summed_local_cells == coarse_global_cells,
      ExcMessage(
        "Sum of locally owned coarse cells does not match the global coarse cell count."));

    coarse_dh.distribute_dofs(fe_dgq);

    const std::map<
      std::pair<types::global_cell_index, types::global_cell_index>,
      std::vector<types::global_cell_index>> &parent_to_child_info =
      agglomerator.get_hierarchy();


    // print the whole content of parent_to_child_info for debugging
    // if (my_rank == 0)
    //   {
    //     std::cout << "Parent to child info on rank 0 content:" << std::endl;
    //     for (const auto &entry : parent_to_child_info)
    //       {
    //         std::cout << "Parent cell index: " << entry.first.first
    //                   << ", Parent level: " << entry.first.second
    //                   << ", Child cell indices: ";
    //         for (const auto &child_idx : entry.second)
    //           std::cout << child_idx << " ";
    //         std::cout << std::endl;
    //       }
    //   }

    ContinuousAggloUtils::PointsAgglo::fill_injection_matrix<dim>(
      coarse_dh,
      fine_dh,
      sparsity_pattern,
      transfer_matrix,
      parent_to_child_info,
      coarse_level_boxes,
      fine_level_boxes,
      coarse_level - 1);
  }

  AssertThrow(transfer_matrix.m() == fine_dh.n_dofs(),
              ExcDimensionMismatch(transfer_matrix.m(), fine_dh.n_dofs()));
  AssertThrow(transfer_matrix.n() == coarse_dh.n_dofs(),
              ExcDimensionMismatch(transfer_matrix.n(), coarse_dh.n_dofs()));

  LinearField<dim> linear_field;

  TrilinosWrappers::MPI::Vector coarse_values(
    transfer_matrix.locally_owned_domain_indices());
  TrilinosWrappers::MPI::Vector fine_values(
    transfer_matrix.locally_owned_range_indices());
  TrilinosWrappers::MPI::Vector transferred_values(
    transfer_matrix.locally_owned_range_indices());

  VectorTools::interpolate(mapping, coarse_dh, linear_field, coarse_values);
  VectorTools::interpolate(mapping, fine_dh, linear_field, fine_values);

  transfer_matrix.vmult(transferred_values, coarse_values);
  transferred_values -= fine_values;

  const double l2_error = transferred_values.l2_norm();

  AssertThrow(
    l2_error < tolerance,
    ExcMessage(
      "Distributed continuous transfer matrix failed the linear field test."));

  if (my_rank == 0)
    {
      std::cout << "Coarse DoFs: " << coarse_dh.n_dofs() << std::endl;
      std::cout << "Fine DoFs: " << fine_dh.n_dofs() << std::endl;
      std::cout << "Transfer matrix: " << transfer_matrix.m() << " x "
                << transfer_matrix.n() << std::endl;
      std::cout << "Linear transfer L2 error: " << l2_error << std::endl;
      std::cout << "Distributed continuous transfer matrix test: OK"
                << std::endl;
    }

  return 0;
}
