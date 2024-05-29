// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) XXXX - YYYY by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------


// Check the operator restriction on coarser, agglomerate, levels.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/multigrid.h>

#ifdef DEAL_II_WITH_TRILINOS
#  include <EpetraExt_RowMatrixOut.h>
#endif

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_integrate_difference.templates.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <agglomeration_handler.h>
#include <agglomerator.h>
#include <poly_utils.h>
#include <utils.h>

using namespace dealii;

static constexpr unsigned int degree_finite_element = 2;
static constexpr unsigned int n_components          = 1;

enum class GridType
{
  grid_generator, // hyper_cube or hyper_ball
  unstructured    // square generated with gmsh, unstructured
};


void
write_to_matrix_market_format(const std::string                    &filename,
                              const std::string                    &matrix_name,
                              const TrilinosWrappers::SparseMatrix &matrix)
{
#ifdef DEAL_II_WITH_TRILINOS
  const Epetra_CrsMatrix &trilinos_matrix = matrix.trilinos_matrix();

  const int ierr =
    EpetraExt::RowMatrixToMatrixMarketFile(filename.c_str(),
                                           trilinos_matrix,
                                           matrix_name.c_str(),
                                           0 /*description field empty*/,
                                           true /*write header*/);
  AssertThrow(ierr == 0, ExcTrilinosError(ierr));
#else
  (void)filename;
  (void)matrix_name;
  (void)matrix;
#endif
}

template <int dim, int degree, int n_qpoints, typename number = double>
class LaplaceOperatorDGSystemMatrix : public Subscriptor
{
public:
  using value_type          = number;
  using VectorizedArrayType = VectorizedArray<number>;
  using VectorType          = LinearAlgebra::distributed::Vector<number>;


  LaplaceOperatorDGSystemMatrix(){};

  void
  reinit(const Mapping<dim>    &mapping,
         const DoFHandler<dim> &dof_handler,
         const unsigned int     level = numbers::invalid_unsigned_int)
  {
    fe_degree = dof_handler.get_fe().degree;

    const QGauss<1>                                  quad(n_qpoints);
    typename MatrixFree<dim, number>::AdditionalData addit_data;
    addit_data.tasks_parallel_scheme =
      MatrixFree<dim, number>::AdditionalData::none;
    addit_data.tasks_block_size = 3;
    addit_data.mg_level         = level;
    addit_data.mapping_update_flags_inner_faces =
      (update_gradients | update_JxW_values);
    addit_data.mapping_update_flags_boundary_faces =
      (update_gradients | update_JxW_values);
    constraints.close();

    data.reinit(mapping, dof_handler, constraints, quad, addit_data);

    // compute_inverse_diagonal();
  }

  void
  vmult(LinearAlgebra::distributed::Vector<number>       &dst,
        const LinearAlgebra::distributed::Vector<number> &src) const
  {
    dst = 0;
    vmult_add(dst, src);
  }

  void
  Tvmult(LinearAlgebra::distributed::Vector<number>       &dst,
         const LinearAlgebra::distributed::Vector<number> &src) const
  {
    dst = 0;
    vmult_add(dst, src);
  }

  void
  Tvmult_add(LinearAlgebra::distributed::Vector<number>       &dst,
             const LinearAlgebra::distributed::Vector<number> &src) const
  {
    vmult_add(dst, src);
  }

  void
  vmult_add(LinearAlgebra::distributed::Vector<number>       &dst,
            const LinearAlgebra::distributed::Vector<number> &src) const
  {
    if (!src.partitioners_are_globally_compatible(
          *data.get_dof_info(0).vector_partitioner))
      {
        LinearAlgebra::distributed::Vector<number> src_copy;
        src_copy.reinit(data.get_dof_info().vector_partitioner);
        src_copy = src;
        const_cast<LinearAlgebra::distributed::Vector<number> &>(src).swap(
          src_copy);
      }
    if (!dst.partitioners_are_globally_compatible(
          *data.get_dof_info(0).vector_partitioner))
      {
        LinearAlgebra::distributed::Vector<number> dst_copy;
        dst_copy.reinit(data.get_dof_info().vector_partitioner);
        dst_copy = dst;
        dst.swap(dst_copy);
      }
    dst.zero_out_ghost_values();
    data.loop(&LaplaceOperatorDGSystemMatrix::local_apply,
              &LaplaceOperatorDGSystemMatrix::local_apply_face,
              &LaplaceOperatorDGSystemMatrix::local_apply_boundary,
              this,
              dst,
              src);
  }

  types::global_dof_index
  m() const
  {
    return data.get_vector_partitioner()->size();
  }

  types::global_dof_index
  n() const
  {
    return data.get_vector_partitioner()->size();
  }

  number
  el(const unsigned int row, const unsigned int col) const
  {
    (void)row;
    (void)col;
    AssertThrow(false,
                ExcMessage("Matrix-free does not allow for entry access"));
    return number();
  }

  void
  initialize_dof_vector(
    LinearAlgebra::distributed::Vector<number> &vector) const
  {
    data.initialize_dof_vector(vector);
  }


  const DoFHandler<dim> &
  get_dof_handler() const
  {
    return data.get_dof_handler();
  }


  const Triangulation<dim> &
  get_triangulation() const
  {
    return data.get_dof_handler().get_triangulation();
  }

  const LinearAlgebra::distributed::Vector<number> &
  get_matrix_diagonal_inverse() const
  {
    return inverse_diagonal_entries;
  }


  const MatrixFree<dim, number> *
  get_matrix_free() const
  {
    return &data;
  }



  const TrilinosWrappers::SparseMatrix &
  get_system_matrix() const
  {
    // Uncomment when PR 17029 is merged upstream

    // // Boilerplate for SIP-DG form. TODO: unify interface.
    // //////////////////////////////////////////////////
    // const auto cell_operation = [&](auto &phi) {
    //   phi.evaluate(EvaluationFlags::gradients);
    //   for (unsigned int q = 0; q < phi.n_q_points; ++q)
    //     phi.submit_gradient(phi.get_gradient(q), q);
    //   phi.integrate(EvaluationFlags::gradients);
    // };

    // const auto face_operation = [&](auto &phi_m, auto &phi_p) {
    //   phi_m.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
    //   phi_p.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    //   VectorizedArrayType sigmaF =
    //     (std::abs(
    //        (phi_m.normal_vector(0) * phi_m.inverse_jacobian(0))[dim - 1]) +
    //      std::abs(
    //        (phi_m.normal_vector(0) * phi_p.inverse_jacobian(0))[dim - 1])) *
    //     (number)(std::max(fe_degree, 1) * (fe_degree + 1.0));

    //   for (unsigned int q = 0; q < phi_m.n_q_points; ++q)
    //     {
    //       VectorizedArrayType average_value =
    //         (phi_m.get_value(q) - phi_p.get_value(q)) * 0.5;
    //       VectorizedArrayType average_valgrad =
    //         phi_m.get_normal_derivative(q) + phi_p.get_normal_derivative(q);
    //       average_valgrad = average_value * 2. * sigmaF - average_valgrad *
    //       0.5; phi_m.submit_normal_derivative(-average_value, q);
    //       phi_p.submit_normal_derivative(-average_value, q);
    //       phi_m.submit_value(average_valgrad, q);
    //       phi_p.submit_value(-average_valgrad, q);
    //     }
    //   phi_m.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
    //   phi_p.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
    // };

    // const auto boundary_operation = [&](auto &phi_m) {
    //   phi_m.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
    //   VectorizedArrayType sigmaF =
    //     std::abs(
    //       (phi_m.normal_vector(0) * phi_m.inverse_jacobian(0))[dim - 1]) *
    //     number(std::max(fe_degree, 1) * (fe_degree + 1.0)) * 2.0;

    //   for (unsigned int q = 0; q < phi_m.n_q_points; ++q)
    //     {
    //       VectorizedArrayType average_value   = phi_m.get_value(q);
    //       VectorizedArrayType average_valgrad =
    //       -phi_m.get_normal_derivative(q); average_valgrad += average_value *
    //       sigmaF * 2.0; phi_m.submit_normal_derivative(-average_value, q);
    //       phi_m.submit_value(average_valgrad, q);
    //     }

    //   phi_m.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
    // };

    // //////////////////////////////////////////////////


    // // Check if matrix has already been set up.
    // if (system_matrix.m() == 0 && system_matrix.n() == 0)
    //   {
    //     // Set up sparsity pattern of system matrix.
    //     const auto &dof_handler = data.get_dof_handler();

    //     TrilinosWrappers::SparsityPattern dsp(
    //       data.get_mg_level() != numbers::invalid_unsigned_int ?
    //         dof_handler.locally_owned_mg_dofs(data.get_mg_level()) :
    //         dof_handler.locally_owned_dofs(),
    //       data.get_task_info().communicator);

    //     if (data.get_mg_level() != numbers::invalid_unsigned_int)
    //       MGTools::make_flux_sparsity_pattern(dof_handler,
    //                                           dsp,
    //                                           data.get_mg_level(),
    //                                           constraints);
    //     else
    //       DoFTools::make_flux_sparsity_pattern(dof_handler, dsp,
    //       constraints);

    //     dsp.compress();
    //     system_matrix.reinit(dsp);

    //     // Assemble system matrix. Notice that degree 1 has been hardcoded.

    //     MatrixFreeTools::compute_matrix<dim,
    //                                     degree,
    //                                     n_qpoints,
    //                                     n_components,
    //                                     number,
    //                                     VectorizedArrayType>(
    //       data,
    //       constraints,
    //       system_matrix,
    //       cell_operation,
    //       face_operation,
    //       boundary_operation);
    //   }

    return system_matrix;
  }



  void
  rhs(LinearAlgebra::distributed::Vector<number> &b) const
  {
    const int dummy = 0;


    data.template cell_loop<LinearAlgebra::distributed::Vector<number>, int>(
      [](const auto &matrix_free, auto &dst, const auto &, const auto cells) {
        FEEvaluation<dim, -1, 0, n_components, number> phi(matrix_free, cells);
        for (unsigned int cell = cells.first; cell < cells.second; ++cell)
          {
            phi.reinit(cell);
            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              {
                if constexpr (n_components == 1)
                  {
                    phi.submit_value(1.0, q);
                  }
                else
                  {
                    Tensor<1, n_components, VectorizedArray<number>> temp;
                    for (unsigned int v = 0;
                         v < VectorizedArray<number>::size();
                         ++v)
                      {
                        for (unsigned int i = 0; i < n_components; i++)
                          temp[i][v] = 1.;
                      }
                    phi.submit_value(temp, q);
                  }
              }

            phi.integrate_scatter(EvaluationFlags::values, dst);
          }
      },
      b,
      dummy,
      true);
  }



  void
  compute_inverse_diagonal(
    LinearAlgebra::distributed::Vector<number> &inverse_diagonal_entries)
  {
    data.initialize_dof_vector(inverse_diagonal_entries);
    unsigned int dummy = 0;
    data.loop(&LaplaceOperatorDGSystemMatrix::local_diagonal_cell,
              &LaplaceOperatorDGSystemMatrix::local_diagonal_face,
              &LaplaceOperatorDGSystemMatrix::local_diagonal_boundary,
              this,
              inverse_diagonal_entries,
              dummy);

    for (unsigned int i = 0; i < inverse_diagonal_entries.locally_owned_size();
         ++i)
      if (std::abs(inverse_diagonal_entries.local_element(i)) > 1e-10)
        inverse_diagonal_entries.local_element(i) =
          1. / inverse_diagonal_entries.local_element(i);
  }


private:
  void
  local_apply(const MatrixFree<dim, number>                    &data,
              LinearAlgebra::distributed::Vector<number>       &dst,
              const LinearAlgebra::distributed::Vector<number> &src,
              const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FEEvaluation<dim, -1, 0, 1, number> phi(data);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.read_dof_values(src);
        phi.evaluate(EvaluationFlags::gradients);
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          phi.submit_gradient(phi.get_gradient(q), q);
        phi.integrate(EvaluationFlags::gradients);
        phi.distribute_local_to_global(dst);
      }
  }

  void
  local_apply_face(
    const MatrixFree<dim, number>                    &data,
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    const std::pair<unsigned int, unsigned int>      &face_range) const
  {
    FEFaceEvaluation<dim, -1, 0, 1, number> fe_eval(data, true);
    FEFaceEvaluation<dim, -1, 0, 1, number> fe_eval_neighbor(data, false);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        fe_eval.reinit(face);
        fe_eval_neighbor.reinit(face);

        fe_eval.read_dof_values(src);
        fe_eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
        fe_eval_neighbor.read_dof_values(src);
        fe_eval_neighbor.evaluate(EvaluationFlags::values |
                                  EvaluationFlags::gradients);
        VectorizedArray<number> sigmaF =
          (std::abs((fe_eval.normal_vector(0) *
                     fe_eval.inverse_jacobian(0))[dim - 1]) +
           std::abs((fe_eval.normal_vector(0) *
                     fe_eval_neighbor.inverse_jacobian(0))[dim - 1])) *
          (number)(std::max(fe_degree, 1) * (fe_degree + 1.0));

        for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
          {
            VectorizedArray<number> average_value =
              (fe_eval.get_value(q) - fe_eval_neighbor.get_value(q)) * 0.5;
            VectorizedArray<number> average_valgrad =
              fe_eval.get_normal_derivative(q) +
              fe_eval_neighbor.get_normal_derivative(q);
            average_valgrad =
              average_value * 2. * sigmaF - average_valgrad * 0.5;
            fe_eval.submit_normal_derivative(-average_value, q);
            fe_eval_neighbor.submit_normal_derivative(-average_value, q);
            fe_eval.submit_value(average_valgrad, q);
            fe_eval_neighbor.submit_value(-average_valgrad, q);
          }
        fe_eval.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        fe_eval.distribute_local_to_global(dst);
        fe_eval_neighbor.integrate(EvaluationFlags::values |
                                   EvaluationFlags::gradients);
        fe_eval_neighbor.distribute_local_to_global(dst);
      }
  }

  void
  local_apply_boundary(
    const MatrixFree<dim, number>                    &data,
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    const std::pair<unsigned int, unsigned int>      &face_range) const
  {
    FEFaceEvaluation<dim, -1, 0, 1, number> fe_eval(data, true);
    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        fe_eval.reinit(face);
        fe_eval.read_dof_values(src);
        fe_eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
        VectorizedArray<number> sigmaF =
          std::abs(
            (fe_eval.normal_vector(0) * fe_eval.inverse_jacobian(0))[dim - 1]) *
          number(std::max(fe_degree, 1) * (fe_degree + 1.0)) * 2.0;

        for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
          {
            VectorizedArray<number> average_value = fe_eval.get_value(q);
            VectorizedArray<number> average_valgrad =
              -fe_eval.get_normal_derivative(q);
            average_valgrad += average_value * sigmaF * 2.0;
            fe_eval.submit_normal_derivative(-average_value, q);
            fe_eval.submit_value(average_valgrad, q);
          }

        fe_eval.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        fe_eval.distribute_local_to_global(dst);
      }
  }


  void
  local_diagonal_cell(
    const MatrixFree<dim, number>              &data,
    LinearAlgebra::distributed::Vector<number> &dst,
    const unsigned int &,
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FEEvaluation<dim, -1, 0, 1, number>    phi(data);
    AlignedVector<VectorizedArray<number>> local_diagonal_vector(
      phi.dofs_per_cell);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);

        for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
              phi.begin_dof_values()[j] = VectorizedArray<number>();
            phi.begin_dof_values()[i] = 1.;
            phi.evaluate(EvaluationFlags::gradients);
            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              phi.submit_gradient(phi.get_gradient(q), q);
            phi.integrate(EvaluationFlags::gradients);
            local_diagonal_vector[i] = phi.begin_dof_values()[i];
          }
        for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
          phi.begin_dof_values()[i] = local_diagonal_vector[i];
        phi.distribute_local_to_global(dst);
      }
  }

  void
  local_diagonal_face(
    const MatrixFree<dim, number>              &data,
    LinearAlgebra::distributed::Vector<number> &dst,
    const unsigned int &,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceEvaluation<dim, -1, 0, 1, number> phi(data, true);
    FEFaceEvaluation<dim, -1, 0, 1, number> phi_outer(data, false);
    AlignedVector<VectorizedArray<number>>  local_diagonal_vector(
      phi.dofs_per_cell);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi.reinit(face);
        phi_outer.reinit(face);

        VectorizedArray<number> sigmaF =
          (std::abs((phi.normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]) +
           std::abs(
             (phi.normal_vector(0) * phi_outer.inverse_jacobian(0))[dim - 1])) *
          (number)(std::max(fe_degree, 1) * (fe_degree + 1.0));

        // Compute phi part
        for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
          phi_outer.begin_dof_values()[j] = VectorizedArray<number>();
        phi_outer.evaluate(EvaluationFlags::values |
                           EvaluationFlags::gradients);
        for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
              phi.begin_dof_values()[j] = VectorizedArray<number>();
            phi.begin_dof_values()[i] = 1.;
            phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              {
                VectorizedArray<number> average_value =
                  (phi.get_value(q) - phi_outer.get_value(q)) * 0.5;
                VectorizedArray<number> average_valgrad =
                  phi.get_normal_derivative(q) +
                  phi_outer.get_normal_derivative(q);
                average_valgrad =
                  average_value * 2. * sigmaF - average_valgrad * 0.5;
                phi.submit_normal_derivative(-average_value, q);
                phi.submit_value(average_valgrad, q);
              }
            phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
            local_diagonal_vector[i] = phi.begin_dof_values()[i];
          }
        for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
          phi.begin_dof_values()[i] = local_diagonal_vector[i];
        phi.distribute_local_to_global(dst);

        // Compute phi_outer part
        for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
          phi.begin_dof_values()[j] = VectorizedArray<number>();
        phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
        for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
              phi_outer.begin_dof_values()[j] = VectorizedArray<number>();
            phi_outer.begin_dof_values()[i] = 1.;
            phi_outer.evaluate(EvaluationFlags::values |
                               EvaluationFlags::gradients);

            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              {
                VectorizedArray<number> average_value =
                  (phi.get_value(q) - phi_outer.get_value(q)) * 0.5;
                VectorizedArray<number> average_valgrad =
                  phi.get_normal_derivative(q) +
                  phi_outer.get_normal_derivative(q);
                average_valgrad =
                  average_value * 2. * sigmaF - average_valgrad * 0.5;
                phi_outer.submit_normal_derivative(-average_value, q);
                phi_outer.submit_value(-average_valgrad, q);
              }
            phi_outer.integrate(EvaluationFlags::values |
                                EvaluationFlags::gradients);
            local_diagonal_vector[i] = phi_outer.begin_dof_values()[i];
          }
        for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
          phi_outer.begin_dof_values()[i] = local_diagonal_vector[i];
        phi_outer.distribute_local_to_global(dst);
      }
  }

  void
  local_diagonal_boundary(
    const MatrixFree<dim, number>              &data,
    LinearAlgebra::distributed::Vector<number> &dst,
    const unsigned int &,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceEvaluation<dim, -1, 0, 1, number> phi(data);
    AlignedVector<VectorizedArray<number>>  local_diagonal_vector(
      phi.dofs_per_cell);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi.reinit(face);

        VectorizedArray<number> sigmaF =
          std::abs((phi.normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]) *
          number(std::max(fe_degree, 1) * (fe_degree + 1.0)) * 2.0;

        for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
              phi.begin_dof_values()[j] = VectorizedArray<number>();
            phi.begin_dof_values()[i] = 1.;
            phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              {
                VectorizedArray<number> average_value = phi.get_value(q);
                VectorizedArray<number> average_valgrad =
                  -phi.get_normal_derivative(q);
                average_valgrad += average_value * sigmaF * 2.0;
                phi.submit_normal_derivative(-average_value, q);
                phi.submit_value(average_valgrad, q);
              }

            phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
            local_diagonal_vector[i] = phi.begin_dof_values()[i];
          }
        for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
          phi.begin_dof_values()[i] = local_diagonal_vector[i];
        phi.distribute_local_to_global(dst);
      }
  }



  MatrixFree<dim, number>                    data;
  LinearAlgebra::distributed::Vector<number> inverse_diagonal_entries;
  int                                        fe_degree;
  mutable TrilinosWrappers::SparseMatrix     system_matrix;
  AffineConstraints<number>                  constraints;
};



template <int dim, int spacedim, typename MatrixType>
void
fill_interpolation_matrix(
  const AgglomerationHandler<dim, spacedim> &agglomeration_handler,
  MatrixType                                &interpolation_matrix)
{
  Assert((dim == spacedim), ExcNotImplemented());

  using NumberType = typename MatrixType::value_type;
  constexpr bool is_trilinos_matrix =
    std::is_same_v<MatrixType, TrilinosWrappers::MPI::Vector>;

  [[maybe_unused]]
  typename std::conditional_t<!is_trilinos_matrix, SparsityPattern, void *>
    sp;

  // Get some info from the handler
  const DoFHandler<dim, spacedim> &agglo_dh = agglomeration_handler.agglo_dh;

  DoFHandler<dim, spacedim> *output_dh =
    const_cast<DoFHandler<dim, spacedim> *>(&agglomeration_handler.output_dh);
  const FiniteElement<dim, spacedim> &fe = agglomeration_handler.get_fe();
  const Triangulation<dim, spacedim> &tria =
    agglomeration_handler.get_triangulation();
  const auto &bboxes = agglomeration_handler.get_local_bboxes();

  // Setup an auxiliary DoFHandler for output purposes
  output_dh->reinit(tria);
  output_dh->distribute_dofs(fe);

  const IndexSet &locally_owned_dofs = output_dh->locally_owned_dofs();
  const IndexSet  locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(*output_dh);

  const IndexSet &locally_owned_dofs_agglo = agglo_dh.locally_owned_dofs();


  DynamicSparsityPattern dsp(output_dh->n_dofs(),
                             agglo_dh.n_dofs(),
                             output_dh->locally_owned_dofs());

  std::vector<types::global_dof_index> agglo_dof_indices(fe.dofs_per_cell);
  std::vector<types::global_dof_index> standard_dof_indices(fe.dofs_per_cell);
  std::vector<types::global_dof_index> output_dof_indices(fe.dofs_per_cell);

  Quadrature<dim>         quad(fe.get_unit_support_points());
  FEValues<dim, spacedim> output_fe_values(fe, quad, update_quadrature_points);

  for (const auto &cell : agglo_dh.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        if (agglomeration_handler.is_master_cell(cell))
          {
            auto slaves = agglomeration_handler.get_slaves_of_idx(
              cell->active_cell_index());
            slaves.emplace_back(cell);

            cell->get_dof_indices(agglo_dof_indices);

            for (const auto &slave : slaves)
              {
                // addd master-slave relationship
                const auto slave_output =
                  slave->as_dof_handler_iterator(*output_dh);
                slave_output->get_dof_indices(output_dof_indices);
                for (const auto row : output_dof_indices)
                  dsp.add_entries(row,
                                  agglo_dof_indices.begin(),
                                  agglo_dof_indices.end());
              }
          }
      }


  const auto assemble_interpolation_matrix = [&]() {
    FullMatrix<NumberType>  local_matrix(fe.dofs_per_cell, fe.dofs_per_cell);
    std::vector<Point<dim>> reference_q_points(fe.dofs_per_cell);

    // Dummy AffineConstraints, only needed for loc2glb
    AffineConstraints<NumberType> c;
    c.close();

    for (const auto &cell : agglo_dh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          if (agglomeration_handler.is_master_cell(cell))
            {
              auto slaves = agglomeration_handler.get_slaves_of_idx(
                cell->active_cell_index());
              slaves.emplace_back(cell);

              cell->get_dof_indices(agglo_dof_indices);

              const types::global_cell_index polytope_index =
                agglomeration_handler.cell_to_polytope_index(cell);

              // Get the box of this agglomerate.
              const BoundingBox<dim> &box = bboxes[polytope_index];

              for (const auto &slave : slaves)
                {
                  // add master-slave relationship
                  const auto slave_output =
                    slave->as_dof_handler_iterator(*output_dh);

                  slave_output->get_dof_indices(output_dof_indices);
                  output_fe_values.reinit(slave_output);

                  local_matrix = 0.;

                  const auto &q_points =
                    output_fe_values.get_quadrature_points();
                  for (const auto i : output_fe_values.dof_indices())
                    {
                      const auto &p = box.real_to_unit(q_points[i]);
                      for (const auto j : output_fe_values.dof_indices())
                        {
                          local_matrix(i, j) = fe.shape_value(j, p);
                        }
                    }
                  c.distribute_local_to_global(local_matrix,
                                               output_dof_indices,
                                               agglo_dof_indices,
                                               interpolation_matrix);
                }
            }
        }
  };


  if constexpr (std::is_same_v<MatrixType, TrilinosWrappers::SparseMatrix>)
    {
      const MPI_Comm &communicator = tria.get_communicator();
      SparsityTools::distribute_sparsity_pattern(dsp,
                                                 locally_owned_dofs,
                                                 communicator,
                                                 locally_relevant_dofs);

      interpolation_matrix.reinit(locally_owned_dofs,
                                  locally_owned_dofs_agglo,
                                  dsp,
                                  communicator);
      assemble_interpolation_matrix();
    }
  else if constexpr (std::is_same_v<MatrixType, SparseMatrix<NumberType>>)
    {
      sp.copy_from(dsp);
      interpolation_matrix.reinit(sp);
      assemble_interpolation_matrix();
    }
  else
    {
      // PETSc, LA::d::v options not implemented.
      (void)agglomeration_handler;
      AssertThrow(false, ExcNotImplemented());
    }

  // If tria is distributed
  if (dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(&tria) !=
      nullptr)
    interpolation_matrix.compress(VectorOperation::add);
}



/**
 * Utility to compute jump terms when the interface is locally owned, i.e. both
 * elements are locally owned.
 */
template <int dim>
void
assemble_local_jumps_and_averages(FullMatrix<double>      &M11,
                                  FullMatrix<double>      &M12,
                                  FullMatrix<double>      &M21,
                                  FullMatrix<double>      &M22,
                                  const FEValuesBase<dim> &fe_faces0,
                                  const FEValuesBase<dim> &fe_faces1,
                                  const double             penalty_constant,
                                  const double             h_f)
{
  const std::vector<Tensor<1, dim>> &normals = fe_faces0.get_normal_vectors();
  const unsigned int                 dofs_per_cell =
    M11.m(); // size of local matrices equals the #DoFs

  for (unsigned int q_index : fe_faces0.quadrature_point_indices())
    {
      const Tensor<1, dim> &normal = normals[q_index];
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              M11(i, j) +=
                (-0.5 * fe_faces0.shape_grad(i, q_index) * normal *
                   fe_faces0.shape_value(j, q_index) -
                 0.5 * fe_faces0.shape_grad(j, q_index) * normal *
                   fe_faces0.shape_value(i, q_index) +
                 (penalty_constant / h_f) * fe_faces0.shape_value(i, q_index) *
                   fe_faces0.shape_value(j, q_index)) *
                fe_faces0.JxW(q_index);

              M12(i, j) +=
                (0.5 * fe_faces0.shape_grad(i, q_index) * normal *
                   fe_faces1.shape_value(j, q_index) -
                 0.5 * fe_faces1.shape_grad(j, q_index) * normal *
                   fe_faces0.shape_value(i, q_index) -
                 (penalty_constant / h_f) * fe_faces0.shape_value(i, q_index) *
                   fe_faces1.shape_value(j, q_index)) *
                fe_faces1.JxW(q_index);


              M21(i, j) +=
                (-0.5 * fe_faces1.shape_grad(i, q_index) * normal *
                   fe_faces0.shape_value(j, q_index) +
                 0.5 * fe_faces0.shape_grad(j, q_index) * normal *
                   fe_faces1.shape_value(i, q_index) -
                 (penalty_constant / h_f) * fe_faces1.shape_value(i, q_index) *
                   fe_faces0.shape_value(j, q_index)) *
                fe_faces1.JxW(q_index);


              M22(i, j) +=
                (0.5 * fe_faces1.shape_grad(i, q_index) * normal *
                   fe_faces1.shape_value(j, q_index) +
                 0.5 * fe_faces1.shape_grad(j, q_index) * normal *
                   fe_faces1.shape_value(i, q_index) +
                 (penalty_constant / h_f) * fe_faces1.shape_value(i, q_index) *
                   fe_faces1.shape_value(j, q_index)) *
                fe_faces1.JxW(q_index);
            }
        }
    }
}



/**
 * Same as above, but for a ghosted neighbor.
 */
template <int dim>
void
assemble_local_jumps_and_averages_ghost(
  FullMatrix<double>                             &M11,
  FullMatrix<double>                             &M12,
  FullMatrix<double>                             &M21,
  FullMatrix<double>                             &M22,
  const FEValuesBase<dim>                        &fe_faces0,
  const std::vector<std::vector<double>>         &recv_values,
  const std::vector<std::vector<Tensor<1, dim>>> &recv_gradients,
  const std::vector<double>                      &recv_jxws,
  const double                                    penalty_constant,
  const double                                    h_f)
{
  Assert(
    (recv_values.size() > 0 && recv_gradients.size() && recv_jxws.size()),
    ExcMessage(
      "Not possible to assemble jumps and averages at a ghosted interface."));
  const unsigned int dofs_per_cell = M11.m();

  const std::vector<Tensor<1, dim>> &normals = fe_faces0.get_normal_vectors();
  for (unsigned int q_index : fe_faces0.quadrature_point_indices())
    {
      const Tensor<1, dim> &normal = normals[q_index];
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              M11(i, j) +=
                (-0.5 * fe_faces0.shape_grad(i, q_index) * normal *
                   fe_faces0.shape_value(j, q_index) -
                 0.5 * fe_faces0.shape_grad(j, q_index) * normal *
                   fe_faces0.shape_value(i, q_index) +
                 (penalty_constant / h_f) * fe_faces0.shape_value(i, q_index) *
                   fe_faces0.shape_value(j, q_index)) *
                fe_faces0.JxW(q_index);

              M12(i, j) +=
                (0.5 * fe_faces0.shape_grad(i, q_index) * normal *
                   recv_values[j][q_index] -
                 0.5 * recv_gradients[j][q_index] * normal *
                   fe_faces0.shape_value(i, q_index) -
                 (penalty_constant / h_f) * fe_faces0.shape_value(i, q_index) *
                   recv_values[j][q_index]) *
                recv_jxws[q_index];


              M21(i, j) += (-0.5 * recv_gradients[i][q_index] * normal *
                              fe_faces0.shape_value(j, q_index) +
                            0.5 * fe_faces0.shape_grad(j, q_index) * normal *
                              recv_values[i][q_index] -
                            (penalty_constant / h_f) * recv_values[i][q_index] *
                              fe_faces0.shape_value(j, q_index)) *
                           recv_jxws[q_index];


              M22(i, j) += (0.5 * recv_gradients[i][q_index] * normal *
                              recv_values[j][q_index] +
                            0.5 * recv_gradients[j][q_index] * normal *
                              recv_values[i][q_index] +
                            (penalty_constant / h_f) * recv_values[i][q_index] *
                              recv_values[j][q_index]) *
                           recv_jxws[q_index];
            }
        }
    }
}



template <int dim, typename MatrixType>
void
assemble_matrix(AgglomerationHandler<dim> *ah,
                MatrixType                &system_matrix,
                const MPI_Comm            &comm)
{
  AffineConstraints<double> constraints;
  constraints.close();
  DynamicSparsityPattern sparsity_pattern;
  ah->create_agglomeration_sparsity_pattern(sparsity_pattern);

  const double reaction_coefficient = 0.;
  const double penalty_constant =
    20. * degree_finite_element * degree_finite_element;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  locally_owned_dofs    = ah->agglo_dh.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(ah->agglo_dh);

  system_matrix.reinit(locally_owned_dofs,
                       locally_owned_dofs,
                       sparsity_pattern,
                       comm);

  const unsigned int dofs_per_cell = ah->get_fe().n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

  FullMatrix<double> M11(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> M12(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> M21(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> M22(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices_neighbor(
    dofs_per_cell);

  auto polytope = ah->begin();
  for (; polytope != ah->end(); ++polytope)
    {
      if (polytope->is_locally_owned())
        {
          cell_matrix = 0.;

          const auto &agglo_values = ah->reinit(polytope);


          for (unsigned int q_index : agglo_values.quadrature_point_indices())
            {
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      cell_matrix(i, j) +=
                        (agglo_values.shape_grad(i, q_index) *
                           agglo_values.shape_grad(j, q_index) +
                         reaction_coefficient *
                           agglo_values.shape_value(i, q_index) *
                           agglo_values.shape_value(j, q_index)) *
                        agglo_values.JxW(q_index);
                    }
                }
            }

          // get volumetric DoFs
          polytope->get_dof_indices(local_dof_indices);


          // Assemble face terms
          unsigned int n_faces = polytope->n_faces();

          const double h_f = polytope->diameter();
          for (unsigned int f = 0; f < n_faces; ++f)
            {
              if (polytope->at_boundary(f))
                {
                  // Get normal vectors seen from each agglomeration.
                  const auto &fe_face = ah->reinit(polytope, f);
                  const auto &normals = fe_face.get_normal_vectors();

                  for (unsigned int q_index :
                       fe_face.quadrature_point_indices())
                    {
                      const Tensor<1, dim> &normal = normals[q_index];
                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                              cell_matrix(i, j) +=
                                (-fe_face.shape_value(i, q_index) *
                                   fe_face.shape_grad(j, q_index) * normal -
                                 fe_face.shape_grad(i, q_index) * normal *
                                   fe_face.shape_value(j, q_index) +
                                 (penalty_constant / h_f) *
                                   fe_face.shape_value(i, q_index) *
                                   fe_face.shape_value(j, q_index)) *
                                fe_face.JxW(q_index);
                            }
                        }
                    }
                }
              else
                {
                  const auto &neigh_polytope = polytope->neighbor(f);
                  if (polytope->id() < neigh_polytope->id())
                    {
                      unsigned int nofn =
                        polytope->neighbor_of_agglomerated_neighbor(f);

                      Assert(neigh_polytope->neighbor(nofn)->id() ==
                               polytope->id(),
                             ExcMessage("Mismatch."));

                      const auto &fe_faces =
                        ah->reinit_interface(polytope, neigh_polytope, f, nofn);
                      const auto &fe_faces0 = fe_faces.first;

                      if (neigh_polytope->is_locally_owned())
                        {
                          // use both fevalues
                          const auto &fe_faces1 = fe_faces.second;

                          M11 = 0.;
                          M12 = 0.;
                          M21 = 0.;
                          M22 = 0.;

                          assemble_local_jumps_and_averages(M11,
                                                            M12,
                                                            M21,
                                                            M22,
                                                            fe_faces0,
                                                            fe_faces1,
                                                            penalty_constant,
                                                            h_f);

                          // distribute DoFs accordingly
                          // fluxes
                          neigh_polytope->get_dof_indices(
                            local_dof_indices_neighbor);

                          constraints.distribute_local_to_global(
                            M11, local_dof_indices, system_matrix);
                          constraints.distribute_local_to_global(
                            M12,
                            local_dof_indices,
                            local_dof_indices_neighbor,
                            system_matrix);
                          constraints.distribute_local_to_global(
                            M21,
                            local_dof_indices_neighbor,
                            local_dof_indices,
                            system_matrix);
                          constraints.distribute_local_to_global(
                            M22, local_dof_indices_neighbor, system_matrix);
                        }
                      else
                        {
                          // neigh polytope is ghosted, so retrieve necessary
                          // metadata.

                          types::subdomain_id neigh_rank =
                            neigh_polytope->subdomain_id();

                          const auto &recv_jxws =
                            ah->recv_jxws.at(neigh_rank)
                              .at({neigh_polytope->id(), nofn});

                          const auto &recv_values =
                            ah->recv_values.at(neigh_rank)
                              .at({neigh_polytope->id(), nofn});

                          const auto &recv_gradients =
                            ah->recv_gradients.at(neigh_rank)
                              .at({neigh_polytope->id(), nofn});

                          M11 = 0.;
                          M12 = 0.;
                          M21 = 0.;
                          M22 = 0.;

                          // there's no FEFaceValues on the other side (it's
                          // ghosted), so we just pass the actual data we have
                          // recevied from the neighboring ghosted polytope
                          assemble_local_jumps_and_averages_ghost(
                            M11,
                            M12,
                            M21,
                            M22,
                            fe_faces0,
                            recv_values,
                            recv_gradients,
                            recv_jxws,
                            penalty_constant,
                            h_f);


                          // distribute DoFs accordingly
                          // fluxes
                          neigh_polytope->get_dof_indices(
                            local_dof_indices_neighbor);

                          constraints.distribute_local_to_global(
                            M11, local_dof_indices, system_matrix);
                          constraints.distribute_local_to_global(
                            M12,
                            local_dof_indices,
                            local_dof_indices_neighbor,
                            system_matrix);
                          constraints.distribute_local_to_global(
                            M21,
                            local_dof_indices_neighbor,
                            local_dof_indices,
                            system_matrix);
                          constraints.distribute_local_to_global(
                            M22, local_dof_indices_neighbor, system_matrix);
                        } // ghosted polytope case


                    } // only once
                }     // internal face
            }         // face loop

          constraints.distribute_local_to_global(cell_matrix,
                                                 local_dof_indices,
                                                 system_matrix);

        } // locally owned polytopes
    }

  system_matrix.compress(VectorOperation::add);
}



template <int dim>
class ExportMGMatrices
{
public:
  ExportMGMatrices(const GridType    &grid_type,
                   const unsigned int degree,
                   const MPI_Comm     comm);
  void
  run();

private:
  void
  make_fine_grid(const unsigned int);
  void
  agglomerate_and_compute_level_matrices();

  const MPI_Comm                                 comm;
  const unsigned int                             n_ranks;
  FE_DGQ<dim>                                    fe_dg;
  const GridType                                &grid_type;
  parallel::fullydistributed::Triangulation<dim> tria_pft;
  DoFHandler<dim>                                dof_handler;
  ConditionalOStream                             pcout;


  std::vector<std::unique_ptr<AgglomerationHandler<dim>>>
    agglomeration_handlers;

  std::vector<TrilinosWrappers::SparseMatrix> injection_matrices_two_level;
  std::unique_ptr<AgglomerationHandler<dim>>  agglomeration_handler_coarse;
};



template <int dim>
ExportMGMatrices<dim>::ExportMGMatrices(const GridType    &grid_type_,
                                        const unsigned int degree,
                                        const MPI_Comm     communicator)
  : comm(communicator)
  , n_ranks(Utilities::MPI::n_mpi_processes(comm))
  , fe_dg(degree)
  , grid_type(grid_type_)
  , tria_pft(comm)
  , dof_handler(tria_pft)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(comm) == 0))
{
  pcout << "Running with " << n_ranks << " MPI ranks." << std::endl;
  pcout << "Grid type:";
  grid_type == GridType::grid_generator ?
    pcout << " Structured square" << std::endl :
    pcout << " Unstructured square" << std::endl;
}



template <int dim>
void
ExportMGMatrices<dim>::make_fine_grid(const unsigned int n_global_refinements)
{
  Triangulation<dim> tria;

  if (grid_type == GridType::unstructured)
    {
      GridIn<dim> grid_in;
      grid_in.attach_triangulation(tria);
      //   std::ifstream abaqus_file("../../meshes/piston_3.inp"); // piston
      //   mesh
      std::ifstream gmsh_file(
        "meshes/t3.msh"); // unstructured square [0,1]^2
                          //   grid_in.read_abaqus(abaqus_file);
      grid_in.read_msh(gmsh_file);
      tria.refine_global(n_global_refinements - 2);
    }
  else if (grid_type == GridType::grid_generator)
    {
      GridGenerator::hyper_cube(tria, 0., 1.);
      tria.refine_global(n_global_refinements + 1);
    }
  else
    {
      Assert(false, ExcInternalError());
    }

  // Partition serial triangulation:
  GridTools::partition_triangulation(n_ranks, tria);

  // Create building blocks:
  const TriangulationDescription::Description<dim, dim> description =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      tria, comm);

  tria_pft.create_triangulation(description);
}



template <int dim>
void
ExportMGMatrices<dim>::agglomerate_and_compute_level_matrices()
{
  using VectorType = LinearAlgebra::distributed::Vector<double>;
  GridTools::Cache<dim> cached_tria(tria_pft);

  // Partition with Rtree locally to each partition.
  MappingQ1<dim> mapping; // use standard mapping

  namespace bgi = boost::geometry::index;
  static constexpr unsigned int max_elem_per_node =
    PolyUtils::constexpr_pow(2, dim); // 2^dim
  std::vector<std::pair<BoundingBox<dim>,
                        typename Triangulation<dim>::active_cell_iterator>>
               boxes(tria_pft.n_locally_owned_active_cells());
  unsigned int i = 0;
  for (const auto &cell : tria_pft.active_cell_iterators())
    if (cell->is_locally_owned())
      boxes[i++] = std::make_pair(mapping.get_bounding_box(cell), cell);

  auto tree = pack_rtree<bgi::rstar<max_elem_per_node>>(boxes);
  Assert(n_levels(tree) >= 2, ExcMessage("At least two levels are needed."));
  pcout << "Total number of available levels: " << n_levels(tree) << std::endl;

  agglomeration_handler_coarse =
    std::make_unique<AgglomerationHandler<dim>>(cached_tria);


  const unsigned int starting_level    = 3;
  const unsigned int total_tree_levels = n_levels(tree) - starting_level + 1;
  agglomeration_handlers.resize(total_tree_levels);

  // Loop through the available levels and set AgglomerationHandlers up.
  for (unsigned int extraction_level = starting_level;
       extraction_level <= n_levels(tree);
       ++extraction_level)
    {
      agglomeration_handlers[extraction_level - starting_level] =
        std::make_unique<AgglomerationHandler<dim>>(cached_tria);
      CellsAgglomerator<dim, decltype(tree)> agglomerator{tree,
                                                          extraction_level};
      const auto agglomerates = agglomerator.extract_agglomerates();
      agglomeration_handlers[extraction_level - starting_level]
        ->connect_hierarchy(agglomerator);

      // Flag elements for agglomeration
      unsigned int agglo_index = 0;
      for (unsigned int i = 0; i < agglomerates.size(); ++i)
        {
          const auto &agglo = agglomerates[i]; // i-th agglomerate
          for (const auto &el : agglo)
            {
              el->set_material_id(agglo_index);
            }
          ++agglo_index;
        }

      const unsigned int n_local_agglomerates = agglo_index;
      unsigned int       total_agglomerates =
        Utilities::MPI::sum(n_local_agglomerates, comm);
      pcout << "Total agglomerates per (tree) level: " << extraction_level
            << ": " << total_agglomerates << std::endl;

      // Now, perform agglomeration within each locally owned partition
      std::vector<
        std::vector<typename Triangulation<dim>::active_cell_iterator>>
        cells_per_subdomain(n_local_agglomerates);
      for (const auto &cell : tria_pft.active_cell_iterators())
        if (cell->is_locally_owned())
          cells_per_subdomain[cell->material_id()].push_back(cell);

      // For every subdomain, agglomerate elements together
      for (std::size_t i = 0; i < cells_per_subdomain.size(); ++i)
        agglomeration_handlers[extraction_level - starting_level]
          ->define_agglomerate(cells_per_subdomain[i]);

      agglomeration_handlers[extraction_level - starting_level]
        ->initialize_fe_values(QGauss<dim>(degree_finite_element + 1),
                               update_values | update_gradients |
                                 update_JxW_values | update_quadrature_points,
                               QGauss<dim - 1>(degree_finite_element + 1),
                               update_JxW_values);
      agglomeration_handlers[extraction_level - starting_level]
        ->distribute_agglomerated_dofs(fe_dg);
    }

  // Compute two-level transfers between agglomeration handlers
  pcout << "Fill injection matrices between agglomerated levels" << std::endl;
  injection_matrices_two_level.resize(total_tree_levels - 1);
  pcout << "Number of injection matrices: "
        << injection_matrices_two_level.size() << std::endl;
  for (unsigned int l = 1; l < total_tree_levels; ++l)
    {
      pcout << "from level " << l - 1 << " to level " << l << std::endl;
      SparsityPattern sparsity;
      Utils::fill_injection_matrix(*agglomeration_handlers[l - 1],
                                   *agglomeration_handlers[l],
                                   sparsity,
                                   injection_matrices_two_level[l - 1]);
    }
  pcout << "Computed two-level matrices between agglomerated levels"
        << std::endl;

  // Compute embedding matrices (last matrix from first agglo to finest real
  // mesh is computed hereafter)
  std::vector<TrilinosWrappers::SparseMatrix> embedding_matrices_value(
    total_tree_levels);
  for (unsigned int l = 0; l < total_tree_levels; ++l)
    {
      pcout << "Filling interpolation matrix number " << l << std::endl;
      fill_interpolation_matrix(*agglomeration_handlers[l],
                                embedding_matrices_value[l]);
    }

  std::vector<TrilinosWrappers::SparseMatrix *> embedding_matrices(
    total_tree_levels);
  for (unsigned int i = 0; i < embedding_matrices.size(); i++)
    embedding_matrices[i] = &embedding_matrices_value[i]; // TODO: avoid this


  pcout << "Last injection matrix has size: (" << embedding_matrices.back()->m()
        << "," << embedding_matrices.back()->n() << ")" << std::endl;



  // Define transfer between levels.
  std::vector<TrilinosWrappers::SparseMatrix *> transfer_matrices(
    total_tree_levels);
  for (unsigned int l = 0; l < total_tree_levels - 1; ++l)
    {
      transfer_matrices[l] = &injection_matrices_two_level[l];

      std::string filename{"P_" + std::to_string(l) + "_" +
                           std::to_string(l + 1)};
      std::string matrix_name{"P_" + std::to_string(l) + "_" +
                              std::to_string(l + 1)};
      write_to_matrix_market_format(filename,
                                    matrix_name,
                                    injection_matrices_two_level[l]);
    }

  // Last matrix, fill it by hand
  transfer_matrices[total_tree_levels - 1] = embedding_matrices.back();
  std::string filename_pro{"P_" + std::to_string(total_tree_levels - 1) + "_" +
                           std::to_string(total_tree_levels)};
  std::string matrix_name_pro{"P_" + std::to_string(total_tree_levels - 1) +
                              "_" + std::to_string(total_tree_levels)};
  write_to_matrix_market_format(filename_pro,
                                matrix_name_pro,
                                *transfer_matrices[total_tree_levels - 1]);



  {
    // Uncomment when PR 17029 is merged
    // dof_handler.distribute_dofs(fe_dg);
    // constexpr unsigned int n_qpoints = degree_finite_element + 1;
    // LaplaceOperatorDGSystemMatrix<dim, degree_finite_element, n_qpoints,
    // double>
    //   system_matrix_dg_check;
    // system_matrix_dg_check.reinit(mapping, dof_handler);
    pcout << "Get Trilinos system matrix" << std::endl;
    // const auto &system_matrix_sparse =
    //   system_matrix_dg_check.get_system_matrix();

    TrilinosWrappers::SparseMatrix system_matrix_sparse;
    AgglomerationHandler<dim>      dummy_ah(cached_tria);
    for (const auto &cell : tria_pft.active_cell_iterators())
      if (cell->is_locally_owned())
        dummy_ah.define_agglomerate({cell});

    dummy_ah.initialize_fe_values(QGauss<dim>(degree_finite_element + 1),
                                  update_values | update_gradients |
                                    update_JxW_values |
                                    update_quadrature_points,
                                  QGauss<dim - 1>(degree_finite_element + 1),
                                  update_JxW_values);

    dummy_ah.distribute_agglomerated_dofs(fe_dg);
    assemble_matrix(&dummy_ah, system_matrix_sparse, comm);

    // Dump matrix to file
    std::string filename{"unstructured_square_ord_" +
                         std::to_string(degree_finite_element)};
    std::string matrix_name{"unstructured_square"};
    write_to_matrix_market_format(filename, matrix_name, system_matrix_sparse);

    pcout << "Written system matrix" << std::endl;
  }
}


template <int dim>
void
ExportMGMatrices<dim>::run()
{
  const unsigned int n_vect_doubles = VectorizedArray<double>::size();
  const unsigned int n_vect_bits    = 8 * sizeof(double) * n_vect_doubles;

  pcout << "Vectorization over " << n_vect_doubles
        << " doubles = " << n_vect_bits << " bits ("
        << Utilities::System::get_current_vectorization_level() << ')'
        << std::endl;
  make_fine_grid(6); // 6 global refinements of unit cube
  agglomerate_and_compute_level_matrices();
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  const MPI_Comm                   comm = MPI_COMM_WORLD;
  static constexpr unsigned int    dim  = 2;



  if (Utilities::MPI::this_mpi_process(comm) == 0)
    std::cout << "Degree: " << degree_finite_element << std::endl;
  ExportMGMatrices<dim> problem(GridType::unstructured,
                                degree_finite_element,
                                comm);
  problem.run();
}
