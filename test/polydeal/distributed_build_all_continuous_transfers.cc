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

// Distributed version of build_all_continuous_transfers.cc: builds all
// injection matrices using
// parallel_agglomerate_and_compute_injection_matrices and verifies that each
// transfer reproduces a linear field exactly.

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

static constexpr double       tolerance = 1e-12;
static constexpr unsigned int dim       = 2;
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
      return p[0] + p[1];
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

  static constexpr unsigned int min_elem_per_node = 2;
  static constexpr unsigned int max_elem_per_node = 4;
  static constexpr unsigned int mg_levels         = 3;
  static constexpr bool         skip_leaves       = true;

  // Build a distributed triangulation
  Triangulation<dim> tria_serial;
  GridGenerator::hyper_cube(tria_serial, 0., 1.);
  tria_serial.refine_global(4);
  GridTools::partition_triangulation(n_ranks, tria_serial);

  const auto construction_data =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      tria_serial, comm);

  parallel::fullydistributed::Triangulation<dim, dim> tria_pft(comm);
  tria_pft.create_triangulation(construction_data);

  DoFHandler<dim> dof_handler(tria_pft);
  FE_Q<dim>       fe(1);
  dof_handler.distribute_dofs(fe);

  MappingQ1<dim> mapping;

  // Point agglo part
  {
    // Build injection matrices for all levels
    std::vector<TrilinosWrappers::SparseMatrix>    injection_matrices;
    std::vector<TrilinosWrappers::SparsityPattern> injection_sparsity_patterns;
    std::vector<unsigned int> coarse_space_degrees(mg_levels - 1, 1);
    std::vector<
      std::unique_ptr<parallel::fullydistributed::Triangulation<dim, dim>>>
                                                  triangulations;
    std::vector<std::unique_ptr<DoFHandler<dim>>> support_dof_handlers;

    ContinuousAggloUtils::PointsAgglo::
      parallel_agglomerate_and_compute_injection_matrices<dim,
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

    if (my_rank == 0)
      std::cout << "Built " << injection_matrices.size()
                << " injection matrices" << std::endl;

    LinearField<dim> linear_field;

    for (unsigned int level = 0; level < injection_matrices.size(); ++level)
      {
        const auto &mat = injection_matrices[level];

        TrilinosWrappers::MPI::Vector coarse_values(
          mat.locally_owned_domain_indices());
        TrilinosWrappers::MPI::Vector fine_values(
          mat.locally_owned_range_indices());
        TrilinosWrappers::MPI::Vector transferred_values(
          mat.locally_owned_range_indices());

        if (level < mg_levels - 2)
          {
            VectorTools::interpolate(mapping,
                                     *support_dof_handlers[level],
                                     linear_field,
                                     coarse_values);
            VectorTools::interpolate(mapping,
                                     *support_dof_handlers[level + 1],
                                     linear_field,
                                     fine_values);
          }
        else
          {
            VectorTools::interpolate(mapping,
                                     *support_dof_handlers[level],
                                     linear_field,
                                     coarse_values);
            VectorTools::interpolate(mapping,
                                     dof_handler,
                                     linear_field,
                                     fine_values);
          }

        mat.vmult(transferred_values, coarse_values);

        transferred_values -= fine_values;
        const double l2_error = transferred_values.l2_norm();

        AssertThrow(l2_error < tolerance,
                    ExcMessage(std::string("Injection matrix at level ") +
                               std::to_string(level) +
                               " failed the linear field test with error " +
                               std::to_string(l2_error)));

        if (my_rank == 0)
          {
            if (level < mg_levels - 2)
              std::cout << "Level " << level << " (" << mat.m() << " x "
                        << mat.n() << "): L2 error = " << l2_error << std::endl;
            else
              std::cout << "Level " << level << " (" << mat.m() << " x "
                        << mat.n() << "): L2 error = " << l2_error
                        << " (agglo->original)" << std::endl;
          }
      }

    if (my_rank == 0)
      std::cout << "All point-agglo continuous transfer tests: OK" << std::endl;
  }

  // Cells agglo part
  {
    // Build injection matrices for all levels
    std::vector<TrilinosWrappers::SparseMatrix>    injection_matrices;
    std::vector<TrilinosWrappers::SparsityPattern> injection_sparsity_patterns;
    std::vector<unsigned int> coarse_space_degrees(mg_levels - 1, 1);
    std::vector<
      std::unique_ptr<parallel::fullydistributed::Triangulation<dim, dim>>>
                                                  triangulations;
    std::vector<std::unique_ptr<DoFHandler<dim>>> support_dof_handlers;

    ContinuousAggloUtils::CellsAgglo::
      parallel_agglomerate_and_compute_injection_matrices<dim,
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

    if (my_rank == 0)
      std::cout << "Built " << injection_matrices.size()
                << " injection matrices" << std::endl;

    LinearField<dim> linear_field;

    for (unsigned int level = 0; level < injection_matrices.size(); ++level)
      {
        const auto &mat = injection_matrices[level];

        TrilinosWrappers::MPI::Vector coarse_values(
          mat.locally_owned_domain_indices());
        TrilinosWrappers::MPI::Vector fine_values(
          mat.locally_owned_range_indices());
        TrilinosWrappers::MPI::Vector transferred_values(
          mat.locally_owned_range_indices());

        if (level < mg_levels - 2)
          {
            VectorTools::interpolate(mapping,
                                     *support_dof_handlers[level],
                                     linear_field,
                                     coarse_values);
            VectorTools::interpolate(mapping,
                                     *support_dof_handlers[level + 1],
                                     linear_field,
                                     fine_values);
          }
        else
          {
            VectorTools::interpolate(mapping,
                                     *support_dof_handlers[level],
                                     linear_field,
                                     coarse_values);
            VectorTools::interpolate(mapping,
                                     dof_handler,
                                     linear_field,
                                     fine_values);
          }

        mat.vmult(transferred_values, coarse_values);

        transferred_values -= fine_values;
        const double l2_error = transferred_values.l2_norm();

        AssertThrow(l2_error < tolerance,
                    ExcMessage(std::string("Injection matrix at level ") +
                               std::to_string(level) +
                               " failed the linear field test with error " +
                               std::to_string(l2_error)));

        if (my_rank == 0)
          {
            if (level < mg_levels - 2)
              std::cout << "Level " << level << " (" << mat.m() << " x "
                        << mat.n() << "): L2 error = " << l2_error << std::endl;
            else
              std::cout << "Level " << level << " (" << mat.m() << " x "
                        << mat.n() << "): L2 error = " << l2_error
                        << " (agglo->original)" << std::endl;
          }
      }

    if (my_rank == 0)
      std::cout << "All cell-agglo continuous transfer tests: OK" << std::endl;
  }
  return 0;
}
