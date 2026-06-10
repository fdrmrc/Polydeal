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
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <cmath>
#include <iostream>
#include <vector>

using namespace dealii;

#include "continuous_agglo_utils.h"
#include "multigrid_amg.h"
#include "utils.h"

static constexpr double       tolerance  = 1e-6;
static constexpr unsigned int dim        = 2;
static constexpr bool         use_points = true;

class LinearSumFunction : public Function<dim>
{
public:
  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override
  {
    (void)component;
    return p[0] + p[1];
  }
};

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

  DoFHandler<dim> dof_handler(starting_tria_pft);
  FE_Q<dim>       fe(1);
  dof_handler.distribute_dofs(fe);
  MappingQ1<dim> mapping;

  std::vector<TrilinosWrappers::SparseMatrix>    injection_matrices;
  std::vector<TrilinosWrappers::SparsityPattern> injection_sparsity_patterns;
  unsigned int                                   mg_levels            = 3;
  std::vector<unsigned int>                      coarse_space_degrees = {1, 1};

  std::vector<
    std::unique_ptr<parallel::fullydistributed::Triangulation<dim, dim>>>
                                                triangulations(mg_levels - 1);
  std::vector<std::unique_ptr<DoFHandler<dim>>> support_dof_handlers(mg_levels -
                                                                     1);

  ContinuousAggloUtils::PointsAgglo::
    parallel_agglomerate_and_compute_injection_matrices<dim,
                                                        min_elem_per_node,
                                                        max_elem_per_node>(
      dof_handler,
      mapping,
      true,
      injection_matrices,
      injection_sparsity_patterns,
      mg_levels,
      coarse_space_degrees,
      triangulations,
      support_dof_handlers);

  // Set up Dirichlet boundary conditions using AffineConstraints
  const IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
  const IndexSet locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(dof_handler);

  AffineConstraints<double> constraints;
  constraints.clear();
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  LinearSumFunction linear_bc;
  VectorTools::interpolate_boundary_values(
    mapping, dof_handler, 0, linear_bc, constraints);
  constraints.close();

  // Build distributed sparsity pattern and system matrix
  TrilinosWrappers::SparsityPattern sparsity_pattern(locally_owned_dofs, comm);
  DoFTools::make_sparsity_pattern(dof_handler,
                                  sparsity_pattern,
                                  constraints,
                                  false);
  sparsity_pattern.compress();
  TrilinosWrappers::SparseMatrix system_matrix;
  system_matrix.reinit(sparsity_pattern);

  // Assemble Laplace matrix
  FEValues<dim>             fe_values(mapping,
                          fe,
                          QGauss<dim>(fe.degree + 1),
                          update_gradients | update_JxW_values |
                            update_quadrature_points);
  const unsigned int        dofs_per_cell = fe.n_dofs_per_cell();
  FullMatrix<double>        cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>            cell_rhs(dofs_per_cell);
  std::vector<unsigned int> local_dof_indices(dofs_per_cell);

  LinearAlgebra::distributed::Vector<double> system_rhs(locally_owned_dofs,
                                                        locally_relevant_dofs,
                                                        comm);
  system_rhs = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        cell_matrix = 0.0;
        cell_rhs    = 0.0;
        fe_values.reinit(cell);

        for (const unsigned int q : fe_values.quadrature_point_indices())
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              cell_matrix(i, j) += fe_values.shape_grad(i, q) *
                                   fe_values.shape_grad(j, q) *
                                   fe_values.JxW(q);

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
      }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  // Set up level matrices via AmgProjector
  AmgProjector<dim, TrilinosWrappers::SparseMatrix, double> amg_projector(
    injection_matrices);

  MGLevelObject<std::unique_ptr<TrilinosWrappers::SparseMatrix>>
    multigrid_matrices(0, mg_levels - 1);

  multigrid_matrices[multigrid_matrices.max_level()] =
    std::make_unique<TrilinosWrappers::SparseMatrix>();
  multigrid_matrices[multigrid_matrices.max_level()]->reinit(system_matrix);
  multigrid_matrices[multigrid_matrices.max_level()]->copy_from(system_matrix);

  amg_projector.compute_level_matrices(multigrid_matrices);

  // Set up multigrid solver
  using LevelMatrixType = TrilinosWrappers::SparseMatrix;
  using VectorType      = LinearAlgebra::distributed::Vector<double>;
  mg::Matrix<VectorType> mg_matrix(multigrid_matrices);

  using SmootherType = PreconditionChebyshev<LevelMatrixType, VectorType>;
  mg::SmootherRelaxation<SmootherType, VectorType>     mg_smoother;
  MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
  smoother_data.resize(0, mg_levels - 1);

  VectorType diag_inverse(system_matrix.locally_owned_range_indices(), comm);
  for (unsigned int row = system_matrix.local_range().first;
       row < system_matrix.local_range().second;
       ++row)
    diag_inverse[row] = 1. / system_matrix.diag_element(row);
  diag_inverse.compress(VectorOperation::insert);

  std::vector<VectorType> diag_inverses(mg_levels);
  diag_inverses[mg_levels - 1] = diag_inverse;

  smoother_data[mg_levels - 1].preconditioner =
    std::make_shared<DiagonalMatrix<VectorType>>(diag_inverses[mg_levels - 1]);

  for (unsigned int level = 0; level < mg_levels - 1; ++level)
    {
      smoother_data[level].smoothing_range = 8;
      diag_inverses[level].reinit(
        multigrid_matrices[level]->locally_owned_range_indices(), comm);
      for (unsigned int row = multigrid_matrices[level]->local_range().first;
           row < multigrid_matrices[level]->local_range().second;
           ++row)
        diag_inverses[level][row] =
          1. / multigrid_matrices[level]->diag_element(row);
      diag_inverses[level].compress(VectorOperation::insert);

      smoother_data[level].preconditioner =
        std::make_shared<DiagonalMatrix<VectorType>>(diag_inverses[level]);
    }

  for (unsigned int level = 0; level < mg_levels; ++level)
    {
      if (level > 0)
        {
          smoother_data[level].smoothing_range     = 20.;
          smoother_data[level].degree              = 3;
          smoother_data[level].eig_cg_n_iterations = 20;
        }
      else
        {
          smoother_data[0].smoothing_range     = 1e-3;
          smoother_data[0].degree              = 3;
          smoother_data[0].eig_cg_n_iterations = 20;
        }
    }

  mg_smoother.set_steps(2);
  mg_smoother.initialize(multigrid_matrices, smoother_data);

  Utils::MGCoarseDirect<VectorType,
                        TrilinosWrappers::SparseMatrix,
                        TrilinosWrappers::SolverDirect>
    mg_coarse(*multigrid_matrices[0]);

  MGLevelObject<TrilinosWrappers::SparseMatrix *> mg_level_transfers(0,
                                                                     mg_levels -
                                                                       1);
  for (unsigned int l = 0; l < mg_levels - 1; ++l)
    mg_level_transfers[l] = &injection_matrices[l];

  std::vector<DoFHandler<dim> *> dof_handlers(support_dof_handlers.size() + 1);
  for (unsigned int i = 0; i < support_dof_handlers.size(); ++i)
    dof_handlers[i] = support_dof_handlers[i].get();
  dof_handlers[support_dof_handlers.size()] = &dof_handler;

  MGTransferAgglomeration<dim, VectorType> mg_transfer(mg_level_transfers,
                                                       dof_handlers);

  Multigrid<VectorType> mg(mg_matrix,
                           mg_coarse,
                           mg_transfer,
                           mg_smoother,
                           mg_smoother,
                           0,
                           numbers::invalid_unsigned_int,
                           Multigrid<VectorType>::v_cycle);

  PreconditionMG<dim, VectorType, MGTransferAgglomeration<dim, VectorType>>
    preconditioner(dof_handler, mg, mg_transfer);

  // Solve with CG
  VectorType dist_solution(locally_owned_dofs, comm);
  dist_solution = 0.0;

  ReductionControl     solver_control(1000, 1e-12, 1e-9);
  SolverCG<VectorType> cg(solver_control);
  cg.solve(system_matrix, dist_solution, system_rhs, preconditioner);
  if (my_rank == 0)
    std::cout << "CG converged in " << solver_control.last_step()
              << " steps, final residual = " << solver_control.last_value()
              << std::endl;

  constraints.distribute(dist_solution);

  const double rhs_norm      = system_rhs.l2_norm();
  const double solution_norm = dist_solution.l2_norm();

  if (my_rank == 0)
    std::cout << "RHS norm = " << rhs_norm
              << ", solution norm = " << solution_norm << std::endl;

  // Compute L2 error against exact solution u(x,y) = x + y
  VectorType solution_ghosted(locally_owned_dofs, locally_relevant_dofs, comm);
  solution_ghosted = dist_solution;
  solution_ghosted.update_ghost_values();

  Vector<double> error_per_cell(starting_tria_pft.n_active_cells());
  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    solution_ghosted,
                                    linear_bc,
                                    error_per_cell,
                                    QGauss<dim>(fe.degree + 2),
                                    VectorTools::L2_norm);

  double l2_error =
    std::sqrt(Utilities::MPI::sum(error_per_cell.norm_sqr(), comm));

  if (my_rank == 0)
    std::cout << "L2 error: " << l2_error << std::endl;

  Assert(l2_error < tolerance,
         ExcMessage("L2 error too large: " + std::to_string(l2_error)));
}