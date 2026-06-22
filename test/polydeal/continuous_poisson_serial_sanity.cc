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
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
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
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "agglomeration_handler.h"
#include "continuous_agglo_utils.h"
#include "poly_utils.h"

static constexpr double       tolerance = 1e-6;
static constexpr unsigned int dim       = 2;
using namespace dealii;

// Custom linear function: f(x,y) = x + y
class LinearSumFunction : public Function<2>
{
public:
  virtual double
  value(const Point<2> &p, const unsigned int component = 0) const override
  {
    (void)component;
    return p[0] + p[1];
  }
};

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  namespace bgi                                   = boost::geometry::index;
  static constexpr unsigned int min_elem_per_node = 2;
  static constexpr unsigned int max_elem_per_node = 4;

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0., 1.);
  tria.refine_global(5);

  DoFHandler<dim> dof_handler(tria);
  FE_Q<dim>       fe(1);
  dof_handler.distribute_dofs(fe);

  std::vector<Point<dim>> support_points_vector(dof_handler.n_dofs());
  MappingQ1<dim>          mapping;

  DoFTools::map_dofs_to_support_points(mapping,
                                       dof_handler,
                                       support_points_vector);

  // Point agglo section
  {
    std::vector<SparseMatrix<double>> injection_matrices;
    std::vector<SparsityPattern>      injection_sparsity_patterns;
    std::vector<unsigned int>         coarse_space_degrees = {1, 1};
    const unsigned int                mg_levels            = 3;

    // Output vectors for triangulations and DoFHandlers
    std::vector<std::unique_ptr<Triangulation<dim>>> triangulations;
    std::vector<std::unique_ptr<DoFHandler<dim>>>    support_dof_handlers;

    ContinuousAggloUtils::PointsAgglo::
      agglomerate_and_compute_injection_matrices<2,
                                                 min_elem_per_node,
                                                 max_elem_per_node>(
        support_points_vector,
        true, // skip_leaves
        injection_matrices,
        injection_sparsity_patterns,
        mg_levels,
        coarse_space_degrees,
        triangulations,
        support_dof_handlers);

    for (unsigned int i = 0; i < support_dof_handlers.size(); ++i)
      {
        std::cout << "Level " << i << ":" << std::endl;
        std::cout << "  - Triangulation cells: "
                  << triangulations[i]->n_active_cells() << std::endl;
        std::cout << "  - DoFs: " << support_dof_handlers[i]->n_dofs()
                  << std::endl;
        if (i < injection_matrices.size())
          {
            std::cout << "  - Injection matrix: " << injection_matrices[i].m()
                      << " x " << injection_matrices[i].n() << std::endl;
          }
        std::cout << std::endl;
      }

    std::cout << "Level " << support_dof_handlers.size()
              << " (finest level):" << std::endl;
    std::cout << "  - Triangulation cells: " << tria.n_active_cells()
              << std::endl;
    std::cout << "  - DoFs: " << dof_handler.n_dofs() << std::endl;
    std::cout << std::endl;

    // Check that the sizes of the matrices match expectations
    Assert(injection_matrices.size() == mg_levels - 1,
           ExcMessage(
             "The number of injection matrices should be mg_levels - 1."));
    Assert(
      injection_sparsity_patterns.size() == mg_levels - 1,
      ExcMessage(
        "The number of injection sparsity patterns should be mg_levels - 1."));
    Assert(
      injection_matrices[mg_levels - 2].m() == dof_handler.n_dofs(),
      ExcMessage(
        "The number of rows of the first injection matrix should match the number of DoFs on the finest level."));
    Assert(
      injection_matrices[mg_levels - 3].m() ==
        injection_matrices[mg_levels - 2].n(),
      ExcMessage(
        "The number of rows of the second injection matrix should match the number of columns of the first injection matrix."));

    std::vector<TrilinosWrappers::SparseMatrix> trilinos_transfer_matrices(
      mg_levels - 1);

    for (unsigned int level = 0; level < mg_levels - 1; ++level)
      {
        trilinos_transfer_matrices[level].reinit(injection_matrices[level]);
      }

    AmgProjector<dim, TrilinosWrappers::SparseMatrix, double> amg_projector(
      trilinos_transfer_matrices);

    MGLevelObject<std::unique_ptr<TrilinosWrappers::SparseMatrix>>
      multigrid_matrices(0, mg_levels - 1);

    // Create Laplace matrix for Poisson problem
    SparseMatrix<double> system_matrix;
    SparsityPattern      sparsity_pattern;
    Vector<double>       system_rhs(dof_handler.n_dofs());
    system_rhs = 0.0;
    Vector<double> solution(dof_handler.n_dofs());

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);

    MatrixCreator::create_laplace_matrix(mapping,
                                         dof_handler,
                                         QGauss<dim>(fe.degree + 1),
                                         system_matrix);

    // Interpolate linear boundary conditions: f(x,y) = x + y
    std::map<types::global_dof_index, double> boundary_values;
    LinearSumFunction                         linear_bc;
    VectorTools::interpolate_boundary_values(mapping,
                                             dof_handler,
                                             0,         // boundary id
                                             linear_bc, // f(x,y) = x + y
                                             boundary_values);

    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       solution,
                                       system_rhs);

    multigrid_matrices[multigrid_matrices.max_level()] =
      std::make_unique<TrilinosWrappers::SparseMatrix>();

    multigrid_matrices[multigrid_matrices.max_level()]->reinit(system_matrix);

    amg_projector.compute_level_matrices(multigrid_matrices);

    using LevelMatrixType = TrilinosWrappers::SparseMatrix;
    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    mg::Matrix<VectorType> mg_matrix(multigrid_matrices);

    using SmootherType = PreconditionChebyshev<LevelMatrixType, VectorType>;
    mg::SmootherRelaxation<SmootherType, VectorType>     mg_smoother;
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, mg_levels - 1);

    VectorType diag_inverse(system_matrix.m());
    for (unsigned int row = 0; row < system_matrix.m(); ++row)
      diag_inverse[row] = 1. / system_matrix.diag_element(row);
    diag_inverse.compress(VectorOperation::insert);

    std::vector<VectorType> diag_inverses(mg_levels);
    diag_inverses[mg_levels - 1] = diag_inverse;

    smoother_data[mg_levels - 1].preconditioner =
      std::make_shared<DiagonalMatrix<VectorType>>(
        diag_inverses[mg_levels - 1]);

    for (unsigned int level = 0; level < mg_levels - 1; ++level)
      {
        // For simplicity using the same degree for all levels
        smoother_data[level].smoothing_range = 8;
        diag_inverses[level].reinit(
          multigrid_matrices[level]->m()); // need to reinit
        for (unsigned int row = 0; row < multigrid_matrices[level]->m(); ++row)
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
            smoother_data[level].smoothing_range = 20.; // 15.;
            smoother_data[level].degree = 3; // parameters.smoother_steps
            smoother_data[level].eig_cg_n_iterations = 20;
          }
        else
          {
            smoother_data[0].smoothing_range = 1e-3;
            smoother_data[0].degree = 3; // numbers::invalid_unsigned_int;
            smoother_data[0].eig_cg_n_iterations = 20;
          }
      }

    mg_smoother.set_steps(2);
    mg_smoother.initialize(multigrid_matrices, smoother_data);

    Utils::MGCoarseDirect<VectorType,
                          TrilinosWrappers::SparseMatrix,
                          TrilinosWrappers::SolverDirect>
      mg_coarse(*multigrid_matrices[0]);

    MGLevelObject<TrilinosWrappers::SparseMatrix *> mg_level_transfers(
      0, mg_levels - 1);
    for (unsigned int l = 0; l < mg_levels - 1; ++l)
      mg_level_transfers[l] = &trilinos_transfer_matrices[l];

    std::vector<DoFHandler<dim> *> dof_handlers(support_dof_handlers.size() +
                                                1);

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
                             0, // min_level
                             numbers::invalid_unsigned_int,
                             Multigrid<VectorType>::v_cycle);

    PreconditionMG<dim, VectorType, MGTransferAgglomeration<dim, VectorType>>
      preconditioner(dof_handler, mg, mg_transfer);


    VectorType dist_solution;
    VectorType dist_rhs;
    dist_solution.reinit(dof_handler.n_dofs());
    dist_rhs.reinit(dof_handler.n_dofs());
    for (unsigned int i = 0; i < system_rhs.size(); ++i)
      dist_rhs[i] = system_rhs[i];
    dist_rhs.compress(VectorOperation::insert);

    ReductionControl solver_control(100, 1e-9, 1e-6);

    SolverCG<VectorType> cg(solver_control);
    cg.solve(system_matrix, dist_solution, dist_rhs, preconditioner);

    for (unsigned int i = 0; i < solution.size(); ++i)
      solution[i] = dist_solution[i];

    // Verify solution against analytical solution u(x,y) = x + y
    Vector<double> error_per_cell(tria.n_active_cells());

    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      solution,
                                      linear_bc,
                                      error_per_cell,
                                      QGauss<dim>(fe.degree + 2),
                                      VectorTools::L2_norm);

    double l2_error = error_per_cell.l2_norm();

    Assert(l2_error < tolerance,
           ExcMessage("L2 error too large: " + std::to_string(l2_error)));
  }

  // Cell agglo section
  {
    std::vector<SparseMatrix<double>> injection_matrices;
    std::vector<SparsityPattern>      injection_sparsity_patterns;
    std::vector<unsigned int>         coarse_space_degrees = {1, 1};
    const unsigned int                mg_levels            = 3;

    // Output vectors for triangulations and DoFHandlers
    std::vector<std::unique_ptr<Triangulation<dim>>> triangulations;
    std::vector<std::unique_ptr<DoFHandler<dim>>>    support_dof_handlers;

    ContinuousAggloUtils::CellsAgglo::
      agglomerate_and_compute_injection_matrices<2,
                                                 min_elem_per_node,
                                                 max_elem_per_node>(
        dof_handler,
        mapping,
        true, // skip_leaves
        injection_matrices,
        injection_sparsity_patterns,
        mg_levels,
        coarse_space_degrees,
        triangulations,
        support_dof_handlers);

    for (unsigned int i = 0; i < support_dof_handlers.size(); ++i)
      {
        std::cout << "Level " << i << ":" << std::endl;
        std::cout << "  - Triangulation cells: "
                  << triangulations[i]->n_active_cells() << std::endl;
        std::cout << "  - DoFs: " << support_dof_handlers[i]->n_dofs()
                  << std::endl;
        if (i < injection_matrices.size())
          {
            std::cout << "  - Injection matrix: " << injection_matrices[i].m()
                      << " x " << injection_matrices[i].n() << std::endl;
          }
        std::cout << std::endl;
      }

    std::cout << "Level " << support_dof_handlers.size()
              << " (finest level):" << std::endl;
    std::cout << "  - Triangulation cells: " << tria.n_active_cells()
              << std::endl;
    std::cout << "  - DoFs: " << dof_handler.n_dofs() << std::endl;
    std::cout << std::endl;

    // Check that the sizes of the matrices match expectations
    Assert(injection_matrices.size() == mg_levels - 1,
           ExcMessage(
             "The number of injection matrices should be mg_levels - 1."));
    Assert(
      injection_sparsity_patterns.size() == mg_levels - 1,
      ExcMessage(
        "The number of injection sparsity patterns should be mg_levels - 1."));
    Assert(
      injection_matrices[mg_levels - 2].m() == dof_handler.n_dofs(),
      ExcMessage(
        "The number of rows of the first injection matrix should match the number of DoFs on the finest level."));
    Assert(
      injection_matrices[mg_levels - 3].m() ==
        injection_matrices[mg_levels - 2].n(),
      ExcMessage(
        "The number of rows of the second injection matrix should match the number of columns of the first injection matrix."));

    std::vector<TrilinosWrappers::SparseMatrix> trilinos_transfer_matrices(
      mg_levels - 1);

    for (unsigned int level = 0; level < mg_levels - 1; ++level)
      {
        trilinos_transfer_matrices[level].reinit(injection_matrices[level]);
      }

    AmgProjector<dim, TrilinosWrappers::SparseMatrix, double> amg_projector(
      trilinos_transfer_matrices);

    MGLevelObject<std::unique_ptr<TrilinosWrappers::SparseMatrix>>
      multigrid_matrices(0, mg_levels - 1);

    // Create Laplace matrix for Poisson problem
    SparseMatrix<double> system_matrix;
    SparsityPattern      sparsity_pattern;
    Vector<double>       system_rhs(dof_handler.n_dofs());
    system_rhs = 0.0;
    Vector<double> solution(dof_handler.n_dofs());

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);

    MatrixCreator::create_laplace_matrix(mapping,
                                         dof_handler,
                                         QGauss<dim>(fe.degree + 1),
                                         system_matrix);

    // Interpolate linear boundary conditions: f(x,y) = x + y
    std::map<types::global_dof_index, double> boundary_values;
    LinearSumFunction                         linear_bc;
    VectorTools::interpolate_boundary_values(mapping,
                                             dof_handler,
                                             0,         // boundary id
                                             linear_bc, // f(x,y) = x + y
                                             boundary_values);

    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       solution,
                                       system_rhs);

    multigrid_matrices[multigrid_matrices.max_level()] =
      std::make_unique<TrilinosWrappers::SparseMatrix>();

    multigrid_matrices[multigrid_matrices.max_level()]->reinit(system_matrix);

    amg_projector.compute_level_matrices(multigrid_matrices);

    using LevelMatrixType = TrilinosWrappers::SparseMatrix;
    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    mg::Matrix<VectorType> mg_matrix(multigrid_matrices);

    using SmootherType = PreconditionChebyshev<LevelMatrixType, VectorType>;
    mg::SmootherRelaxation<SmootherType, VectorType>     mg_smoother;
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, mg_levels - 1);

    VectorType diag_inverse(system_matrix.m());
    for (unsigned int row = 0; row < system_matrix.m(); ++row)
      diag_inverse[row] = 1. / system_matrix.diag_element(row);
    diag_inverse.compress(VectorOperation::insert);

    std::vector<VectorType> diag_inverses(mg_levels);
    diag_inverses[mg_levels - 1] = diag_inverse;

    smoother_data[mg_levels - 1].preconditioner =
      std::make_shared<DiagonalMatrix<VectorType>>(
        diag_inverses[mg_levels - 1]);

    for (unsigned int level = 0; level < mg_levels - 1; ++level)
      {
        // For simplicity using the same degree for all levels
        smoother_data[level].smoothing_range = 8;
        diag_inverses[level].reinit(
          multigrid_matrices[level]->m()); // need to reinit
        for (unsigned int row = 0; row < multigrid_matrices[level]->m(); ++row)
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
            smoother_data[level].smoothing_range = 20.; // 15.;
            smoother_data[level].degree = 3; // parameters.smoother_steps
            smoother_data[level].eig_cg_n_iterations = 20;
          }
        else
          {
            smoother_data[0].smoothing_range = 1e-3;
            smoother_data[0].degree = 3; // numbers::invalid_unsigned_int;
            smoother_data[0].eig_cg_n_iterations = 20;
          }
      }

    mg_smoother.set_steps(2);
    mg_smoother.initialize(multigrid_matrices, smoother_data);

    Utils::MGCoarseDirect<VectorType,
                          TrilinosWrappers::SparseMatrix,
                          TrilinosWrappers::SolverDirect>
      mg_coarse(*multigrid_matrices[0]);

    MGLevelObject<TrilinosWrappers::SparseMatrix *> mg_level_transfers(
      0, mg_levels - 1);
    for (unsigned int l = 0; l < mg_levels - 1; ++l)
      mg_level_transfers[l] = &trilinos_transfer_matrices[l];

    std::vector<DoFHandler<dim> *> dof_handlers(support_dof_handlers.size() +
                                                1);

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
                             0, // min_level
                             numbers::invalid_unsigned_int,
                             Multigrid<VectorType>::v_cycle);

    PreconditionMG<dim, VectorType, MGTransferAgglomeration<dim, VectorType>>
      preconditioner(dof_handler, mg, mg_transfer);


    VectorType dist_solution;
    VectorType dist_rhs;
    dist_solution.reinit(dof_handler.n_dofs());
    dist_rhs.reinit(dof_handler.n_dofs());
    for (unsigned int i = 0; i < system_rhs.size(); ++i)
      dist_rhs[i] = system_rhs[i];
    dist_rhs.compress(VectorOperation::insert);

    ReductionControl solver_control(100, 1e-9, 1e-6);

    SolverCG<VectorType> cg(solver_control);
    cg.solve(system_matrix, dist_solution, dist_rhs, preconditioner);

    for (unsigned int i = 0; i < solution.size(); ++i)
      solution[i] = dist_solution[i];

    // Verify solution against analytical solution u(x,y) = x + y
    Vector<double> error_per_cell(tria.n_active_cells());

    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      solution,
                                      linear_bc,
                                      error_per_cell,
                                      QGauss<dim>(fe.degree + 2),
                                      VectorTools::L2_norm);

    double l2_error = error_per_cell.l2_norm();

    Assert(l2_error < tolerance,
           ExcMessage("L2 error too large: " + std::to_string(l2_error)));
  }
}