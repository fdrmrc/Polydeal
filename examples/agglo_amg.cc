// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright ( ) XXXX - YYYY by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------


// Agglomerated multigrid. Verbose implementation, where agglomerated levels
// have been constructed explicitly.

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

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_integrate_difference.templates.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <agglomeration_handler.h>
#include <agglomerator.h>
#include <multigrid_amg.h>
#include <poly_utils.h>
#include <utils.h>

using namespace dealii;

static constexpr unsigned int degree_finite_element = 1;
static constexpr unsigned int n_components          = 1;
static constexpr bool         CHECK_AMG             = true;



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

    compute_inverse_diagonal();
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
    // Boilerplate for SIP-DG form. TODO: unify interface.
    //////////////////////////////////////////////////
    const auto cell_operation = [&](auto &phi) {
      phi.evaluate(EvaluationFlags::gradients);
      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_gradient(phi.get_gradient(q), q);
      phi.integrate(EvaluationFlags::gradients);
    };

    const auto face_operation = [&](auto &phi_m, auto &phi_p) {
      phi_m.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
      phi_p.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

      VectorizedArrayType sigmaF =
        (std::abs(
           (phi_m.normal_vector(0) * phi_m.inverse_jacobian(0))[dim - 1]) +
         std::abs(
           (phi_m.normal_vector(0) * phi_p.inverse_jacobian(0))[dim - 1])) *
        (number)(std::max(fe_degree, 1) * (fe_degree + 1.0));

      for (unsigned int q = 0; q < phi_m.n_q_points; ++q)
        {
          VectorizedArrayType average_value =
            (phi_m.get_value(q) - phi_p.get_value(q)) * 0.5;
          VectorizedArrayType average_valgrad =
            phi_m.get_normal_derivative(q) + phi_p.get_normal_derivative(q);
          average_valgrad = average_value * 2. * sigmaF - average_valgrad * 0.5;
          phi_m.submit_normal_derivative(-average_value, q);
          phi_p.submit_normal_derivative(-average_value, q);
          phi_m.submit_value(average_valgrad, q);
          phi_p.submit_value(-average_valgrad, q);
        }
      phi_m.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
      phi_p.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
    };

    const auto boundary_operation = [&](auto &phi_m) {
      phi_m.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
      VectorizedArrayType sigmaF =
        std::abs(
          (phi_m.normal_vector(0) * phi_m.inverse_jacobian(0))[dim - 1]) *
        number(std::max(fe_degree, 1) * (fe_degree + 1.0)) * 2.0;

      for (unsigned int q = 0; q < phi_m.n_q_points; ++q)
        {
          VectorizedArrayType average_value   = phi_m.get_value(q);
          VectorizedArrayType average_valgrad = -phi_m.get_normal_derivative(q);
          average_valgrad += average_value * sigmaF * 2.0;
          phi_m.submit_normal_derivative(-average_value, q);
          phi_m.submit_value(average_valgrad, q);
        }

      phi_m.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
    };


    //////////////////////////////////////////////////


    // Check if matrix has already been set up.
    if (system_matrix.m() == 0 && system_matrix.n() == 0)
      {
        // Set up sparsity pattern of system matrix.
        const auto &dof_handler = data.get_dof_handler();

        TrilinosWrappers::SparsityPattern dsp(
          data.get_mg_level() != numbers::invalid_unsigned_int ?
            dof_handler.locally_owned_mg_dofs(data.get_mg_level()) :
            dof_handler.locally_owned_dofs(),
          data.get_task_info().communicator);

        if (data.get_mg_level() != numbers::invalid_unsigned_int)
          MGTools::make_flux_sparsity_pattern(dof_handler,
                                              dsp,
                                              data.get_mg_level(),
                                              constraints);
        else
          DoFTools::make_flux_sparsity_pattern(dof_handler, dsp, constraints);

        dsp.compress();
        system_matrix.reinit(dsp);

        // Assemble system matrix. Notice that degree 1 has been hardcoded.

        MatrixFreeTools::compute_matrix<dim,
                                        degree,
                                        n_qpoints,
                                        n_components,
                                        number,
                                        VectorizedArrayType>(
          data,
          constraints,
          system_matrix,
          cell_operation,
          face_operation,
          boundary_operation);
      }

    return system_matrix;
  }



  void
  get_system_matrix(TrilinosWrappers::SparseMatrix &mg_matrix) const
  {
    // Boilerplate for SIP-DG form. TODO: unify interface.
    //////////////////////////////////////////////////
    const auto cell_operation = [&](auto &phi) {
      phi.evaluate(EvaluationFlags::gradients);
      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_gradient(phi.get_gradient(q), q);
      phi.integrate(EvaluationFlags::gradients);
    };

    const auto face_operation = [&](auto &phi_m, auto &phi_p) {
      phi_m.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
      phi_p.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

      VectorizedArrayType sigmaF =
        (std::abs(
           (phi_m.normal_vector(0) * phi_m.inverse_jacobian(0))[dim - 1]) +
         std::abs(
           (phi_m.normal_vector(0) * phi_p.inverse_jacobian(0))[dim - 1])) *
        (number)(std::max(fe_degree, 1) * (fe_degree + 1.0));

      for (unsigned int q = 0; q < phi_m.n_q_points; ++q)
        {
          VectorizedArrayType average_value =
            (phi_m.get_value(q) - phi_p.get_value(q)) * 0.5;
          VectorizedArrayType average_valgrad =
            phi_m.get_normal_derivative(q) + phi_p.get_normal_derivative(q);
          average_valgrad = average_value * 2. * sigmaF - average_valgrad * 0.5;
          phi_m.submit_normal_derivative(-average_value, q);
          phi_p.submit_normal_derivative(-average_value, q);
          phi_m.submit_value(average_valgrad, q);
          phi_p.submit_value(-average_valgrad, q);
        }
      phi_m.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
      phi_p.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
    };

    const auto boundary_operation = [&](auto &phi_m) {
      phi_m.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
      VectorizedArrayType sigmaF =
        std::abs(
          (phi_m.normal_vector(0) * phi_m.inverse_jacobian(0))[dim - 1]) *
        number(std::max(fe_degree, 1) * (fe_degree + 1.0)) * 2.0;

      for (unsigned int q = 0; q < phi_m.n_q_points; ++q)
        {
          VectorizedArrayType average_value   = phi_m.get_value(q);
          VectorizedArrayType average_valgrad = -phi_m.get_normal_derivative(q);
          average_valgrad += average_value * sigmaF * 2.0;
          phi_m.submit_normal_derivative(-average_value, q);
          phi_m.submit_value(average_valgrad, q);
        }

      phi_m.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
    };


    //////////////////////////////////////////////////


    // Check if matrix has already been set up.
    AssertThrow((mg_matrix.m() == 0 && mg_matrix.n() == 0), ExcInternalError());

    // Set up sparsity pattern of system matrix.
    const DoFHandler<dim> &dof_handler = data.get_dof_handler();

    const IndexSet &system_partitioning = dof_handler.locally_owned_dofs();
    const IndexSet  system_relevant_set =
      DoFTools::extract_locally_relevant_dofs(dof_handler);



    DynamicSparsityPattern dsp(dof_handler.n_dofs(),
                               dof_handler.n_dofs(),
                               system_relevant_set);
    DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);

    SparsityTools::distribute_sparsity_pattern(
      dsp,
      system_partitioning,
      data.get_task_info().communicator,
      system_relevant_set);
    mg_matrix.reinit(system_partitioning,
                     dsp,
                     data.get_task_info().communicator);


    MatrixFreeTools::compute_matrix<dim,
                                    degree,
                                    n_qpoints,
                                    n_components,
                                    number,
                                    VectorizedArrayType>(data,
                                                         constraints,
                                                         mg_matrix,
                                                         cell_operation,
                                                         face_operation,
                                                         boundary_operation);
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
  compute_inverse_diagonal()
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
                             locally_relevant_dofs);

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
      const MPI_Comm &communicator = tria.get_mpi_communicator();
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


// DG OperatorLaplace


enum class GridType
{
  grid_generator, // hyper_cube or hyper_ball
  unstructured    // square generated with gmsh, unstructured
};



template <typename VectorType, typename MatrixType, typename SolverType>
class MGCoarseDirect : public MGCoarseGridBase<VectorType>
{
public:
  // using  = typename
  // LinearAlgebra::distributed::Vector<MatrixType::value_type>;
  MGCoarseDirect(const MatrixType &matrix)
  {
    coarse_matrix = &matrix;
    direct_solver.initialize(*coarse_matrix);
  }

  void
  initialize(const MatrixType &matrix)
  {}

  virtual void
  operator()(const unsigned int, VectorType &dst, const VectorType &src) const
  {
    if constexpr (std::is_same_v<SolverType, SparseDirectUMFPACK>)
      direct_solver.vmult(dst, src);
    else if constexpr (std::is_same_v<SolverType,
                                      TrilinosWrappers::SolverDirect>)
      const_cast<SolverType *>(&direct_solver)->solve(*coarse_matrix, dst, src);
    else
      AssertThrow(false, ExcNotImplemented());
  }

  SolverType        direct_solver;
  const MatrixType *coarse_matrix;
};

template <int dim>
class TestMGMatrix
{
public:
  TestMGMatrix(const GridType    &grid_type,
               const unsigned int degree,
               const unsigned int starting_level,
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
  unsigned int                                   starting_level;


  std::vector<std::unique_ptr<AgglomerationHandler<dim>>>
    agglomeration_handlers;

  std::vector<TrilinosWrappers::SparseMatrix> injection_matrices_two_level;
  std::unique_ptr<AgglomerationHandler<dim>>  agglomeration_handler_coarse;
};



template <int dim>
TestMGMatrix<dim>::TestMGMatrix(const GridType    &grid_type_,
                                const unsigned int degree,
                                const unsigned int starting_tree_level,
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
  starting_level = starting_tree_level;
  grid_type == GridType::grid_generator ?
    pcout << " Structured mesh" << std::endl :
    pcout << " Unstructured mesh" << std::endl;
}



template <int dim>
void
TestMGMatrix<dim>::make_fine_grid(const unsigned int n_global_refinements)
{
  Triangulation<dim> tria;

  if (grid_type == GridType::unstructured)
    {
      GridIn<dim> grid_in;
      grid_in.attach_triangulation(tria);

      if constexpr (dim == 2)
        {
          std::ifstream gmsh_file(
            "../../meshes/t3.msh"); // unstructured square [0,1]^2
          grid_in.read_msh(gmsh_file);
          tria.refine_global(n_global_refinements + 2);
        }
      else
        {
          std::ifstream abaqus_file("../../meshes/piston_3.inp"); // piston
          // mesh
          // std::ifstream abaqus_file(
          //   "../../meshes/idealized_lv.msh"); // idealized mesh
          // std::ifstream abaqus_file(
          // "../../meshes/realistic_lv.msh"); // idealized mesh
          grid_in.read_abaqus(abaqus_file);
          // grid_in.read_abaqus(abaqus_file);
          // tria.refine_global(n_global_refinements - 3);
        }
    }
  else if (grid_type == GridType::grid_generator)
    {
      GridGenerator::hyper_cube(tria, 0., 1.);
      tria.refine_global(n_global_refinements);
    }
  else
    {
      Assert(false, ExcInternalError());
    }

  pcout << "Total number of fine cells: " << tria.n_active_cells() << std::endl;

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
TestMGMatrix<dim>::agglomerate_and_compute_level_matrices()
{
  using VectorType = LinearAlgebra::distributed::Vector<double>;
  GridTools::Cache<dim> cached_tria(tria_pft);

  // Partition with Rtree locally to each partition.
  MappingQ1<dim> mapping; // use standard mapping

  // Define matrix free operator
  AffineConstraints constraints;
  constraints.close();
  dof_handler.distribute_dofs(fe_dg);

  constexpr unsigned int n_qpoints = degree_finite_element + 1;
  LaplaceOperatorDGSystemMatrix<dim, degree_finite_element, n_qpoints, double>
    system_matrix_dg;
  system_matrix_dg.reinit(mapping, dof_handler);
  const TrilinosWrappers::SparseMatrix &fine_matrix =
    system_matrix_dg.get_system_matrix();
  pcout << "Built finest operator" << std::endl;


  // Start building R-tree
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


  pcout << "Starting level: " << starting_level << std::endl;
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
  injection_matrices_two_level.resize(total_tree_levels);
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


  // Define transfer between levels.
  std::vector<TrilinosWrappers::SparseMatrix *> transfer_matrices(
    total_tree_levels);
  for (unsigned int l = 0; l < total_tree_levels - 1; ++l)
    transfer_matrices[l] = &injection_matrices_two_level[l];


  // Last matrix, fill it by hand
  // add last two-level (which is an embedding)
  fill_interpolation_matrix(*agglomeration_handlers.back(),
                            injection_matrices_two_level.back());
  transfer_matrices[total_tree_levels - 1] =
    &injection_matrices_two_level.back();

  pcout << injection_matrices_two_level.back().m() << " and "
        << injection_matrices_two_level.back().n() << std::endl;

  AmgProjector<dim, TrilinosWrappers::SparseMatrix, double> amg_projector(
    injection_matrices_two_level);
  pcout << "Initialized projector" << std::endl;



  MGLevelObject<std::unique_ptr<TrilinosWrappers::SparseMatrix>>
    multigrid_matrices(0, total_tree_levels);

  multigrid_matrices[multigrid_matrices.max_level()] =
    std::make_unique<TrilinosWrappers::SparseMatrix>();

  Utilities::System::MemoryStats stats;
  Utilities::System::get_memory_stats(stats);
  const auto print = [this](const double value) {
    const auto min_max_avg =
      dealii::Utilities::MPI::min_max_avg(value / 1e6, MPI_COMM_WORLD);

    pcout << min_max_avg.min << " " << min_max_avg.max << " " << min_max_avg.avg
          << " " << min_max_avg.sum << " ";
  };

  print(stats.VmPeak);
  print(stats.VmSize);
  print(stats.VmHWM);
  print(stats.VmRSS);
  pcout << "Building finest operator" << std::endl;
  system_matrix_dg.get_system_matrix(
    *multigrid_matrices[multigrid_matrices.max_level()]);
  pcout << "Built finest operator" << std::endl;
  Utilities::System::get_memory_stats(stats);
  print(stats.VmPeak);
  print(stats.VmSize);
  print(stats.VmHWM);
  print(stats.VmRSS);

  amg_projector.compute_level_matrices(multigrid_matrices);

  pcout << "Projected using transfer_matrices:" << std::endl;

  pcout << "Check dimensions of level operators" << std::endl;
  for (unsigned int l = 0; l < total_tree_levels + 1; ++l)
    pcout << "Number of rows and cols operator " << l << ":("
          << multigrid_matrices[l]->m() << "," << multigrid_matrices[l]->n()
          << ")" << std::endl;


  // Setup multigrid


  // Multigrid matrices
  using LevelMatrixType = TrilinosWrappers::SparseMatrix;
  mg::Matrix<VectorType> mg_matrix(multigrid_matrices);

  using SmootherType = PreconditionChebyshev<LevelMatrixType, VectorType>;
  mg::SmootherRelaxation<SmootherType, VectorType>     mg_smoother;
  MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
  smoother_data.resize(0, total_tree_levels + 1);

  // system_matrix.compute_diagonal();
  const VectorType &fine_diag_inverse_vector =
    system_matrix_dg.get_matrix_diagonal_inverse();

  // Fine level
  smoother_data[total_tree_levels].preconditioner =
    std::make_shared<DiagonalMatrix<VectorType>>(fine_diag_inverse_vector);
  std::vector<LinearAlgebra::distributed::Vector<double>> diag_inverses(
    total_tree_levels + 1);
  diag_inverses[total_tree_levels] = fine_diag_inverse_vector;


  pcout << "Start defining smoothers data" << std::endl;

  for (unsigned int l = 0; l < total_tree_levels; ++l)
    {
      pcout << "l = " << l << std::endl;
      diag_inverses[l].reinit(
        agglomeration_handlers[l]->agglo_dh.locally_owned_dofs(), comm);

      // Set exact diagonal for each operator
      for (unsigned int i = multigrid_matrices[l]->local_range().first;
           i < multigrid_matrices[l]->local_range().second;
           ++i)
        diag_inverses[l][i] = 1. / multigrid_matrices[l]->diag_element(i);

      smoother_data[l].preconditioner =
        std::make_shared<DiagonalMatrix<VectorType>>(diag_inverses[l]);
    }

  pcout << "Smoothers data initialized" << std::endl;

  for (unsigned int level = 0; level < total_tree_levels + 1; ++level)
    {
      if (level > 0)
        {
          smoother_data[level].smoothing_range     = 20.; // 15.;
          smoother_data[level].degree              = 3;   // 5;
          smoother_data[level].eig_cg_n_iterations = 20;
        }
      else
        {
          smoother_data[0].smoothing_range = 1e-3;
          smoother_data[0].degree = 3; // numbers::invalid_unsigned_int;
          smoother_data[0].eig_cg_n_iterations = dof_handler.n_dofs();
          smoother_data[0].eig_cg_n_iterations = multigrid_matrices[0]->m();
        }
    }

  mg_smoother.set_steps(5);
  mg_smoother.initialize(multigrid_matrices, smoother_data);

  pcout << "Smoothers initialized" << std::endl;

  // Define coarse grid solver
  const unsigned int min_level = 0;
  Utils::MGCoarseDirect<VectorType,
                        TrilinosWrappers::SparseMatrix,
                        TrilinosWrappers::SolverDirect>
    mg_coarse(*multigrid_matrices[min_level]);

  // Transfers
  MGLevelObject<TrilinosWrappers::SparseMatrix *> mg_level_transfers(
    0, total_tree_levels);
  for (unsigned int l = 0; l < total_tree_levels; ++l)
    mg_level_transfers[l] = transfer_matrices[l];


  std::vector<DoFHandler<dim> *> dof_handlers(total_tree_levels + 1);
  for (unsigned int l = 0; l < dof_handlers.size() - 1; ++l)
    dof_handlers[l] = &agglomeration_handlers[l]->agglo_dh;
  dof_handlers[dof_handlers.size() - 1] = &dof_handler; // fine

  unsigned int lev = 0;
  for (const auto &dh : dof_handlers)
    pcout << "Number of DoFs in level " << lev++ << ": " << dh->n_dofs()
          << std::endl;

  MGTransferAgglomeration<dim, VectorType> mg_transfer(mg_level_transfers,
                                                       dof_handlers);
  pcout << "MG transfers initialized" << std::endl;

  // Define multigrid object and convert to preconditioner.
  Multigrid<VectorType> mg(mg_matrix,
                           mg_coarse,
                           mg_transfer,
                           mg_smoother,
                           mg_smoother,
                           min_level,
                           numbers::invalid_unsigned_int,
                           Multigrid<VectorType>::v_cycle);

  PreconditionMG<dim, VectorType, MGTransferAgglomeration<dim, VectorType>>
    preconditioner(dof_handler, mg, mg_transfer);



  // Assemble system rhs
  VectorType system_rhs;
  system_matrix_dg.initialize_dof_vector(system_rhs);

  system_rhs = 0;
  FEEvaluation<dim, degree_finite_element> phi(
    *system_matrix_dg.get_matrix_free());
  for (unsigned int cell = 0;
       cell < system_matrix_dg.get_matrix_free()->n_cell_batches();
       ++cell)
    {
      phi.reinit(cell);
      for (const unsigned int q : phi.quadrature_point_indices())
        phi.submit_value(make_vectorized_array<double>(1.0), q);
      phi.integrate(EvaluationFlags::values);
      phi.distribute_local_to_global(system_rhs);
    }
  system_rhs.compress(VectorOperation::add);


  // Finally, solve.

  VectorType solution;
  system_matrix_dg.initialize_dof_vector(solution);
  ReductionControl     solver_control(10000, 1e-9, 1e-6);
  SolverCG<VectorType> cg(solver_control);
  double               start, stop;
  pcout << "Start solver" << std::endl;
  start = MPI_Wtime();
  cg.solve(system_matrix_dg, solution, system_rhs, preconditioner);
  stop = MPI_Wtime();
  pcout << "Agglo AMG elapsed time: " << stop - start << "[s]" << std::endl;

  pcout << "Initial value: " << solver_control.initial_value() << std::endl;
  pcout << "Converged in " << solver_control.last_step()
        << " iterations with value " << solver_control.last_value()
        << std::endl;



  [[maybe_unused]] auto output_results = [&]() -> void {
    pcout << "Output results" << std::endl;
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution,
                             "interpolated_solution",
                             DataOut<dim>::type_dof_data);

    Vector<float> subdomain(tria_pft.n_active_cells());

    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = tria_pft.locally_owned_subdomain();

    data_out.add_data_vector(subdomain, "subdomain");

    Vector<float> agglo_idx(tria_pft.n_active_cells());
    for (const auto &cell : tria_pft.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          agglo_idx[cell->active_cell_index()] = cell->material_id();
      }
    data_out.add_data_vector(agglo_idx,
                             "agglo_idx",
                             DataOut<dim>::type_cell_data);

    data_out.build_patches(mapping);
    const std::string filename =
      ("agglo_mg." +
       Utilities::int_to_string(tria_pft.locally_owned_subdomain(), 4));
    std::ofstream output((filename + ".vtu").c_str());
    data_out.write_vtu(output);

    {
      std::vector<std::string> filenames;
      for (unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(comm); i++)
        {
          filenames.push_back("agglo_mg." + Utilities::int_to_string(i, 4) +
                              ".vtu");
        }
      std::ofstream master_output("agglo_mg.pvtu");
      data_out.write_pvtu_record(master_output, filenames);
    }
  };

  if (dof_handler.n_dofs() < 3e6)
    output_results();



  if constexpr (CHECK_AMG == true)
    {
      if (starting_level == 2)
        {
          constexpr unsigned int n_qpoints = degree_finite_element + 1;
          LaplaceOperatorDGSystemMatrix<dim,
                                        degree_finite_element,
                                        n_qpoints,
                                        double>
            system_matrix_dg_check;
          system_matrix_dg_check.reinit(mapping, dof_handler);
          pcout << "Classical way" << std::endl;
          TrilinosWrappers::PreconditionAMG                 prec_amg;
          TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
          amg_data.aggregation_threshold = 1e-3;
          amg_data.smoother_type         = "Chebyshev";
          amg_data.smoother_sweeps       = 5;
          amg_data.output_details        = true;
          if (degree_finite_element > 1)
            amg_data.higher_order_elements = true;
          pcout << "Initialized AMG prec matrix" << std::endl;
          prec_amg.initialize(fine_matrix, amg_data);

          solution = 0.;
          SolverCG<VectorType> cg_check(solver_control);
          double               start_cg, stop_cg;
          start_cg = MPI_Wtime();
          cg_check.solve(fine_matrix,
                         solution,
                         system_rhs,
                         prec_amg); // with id
          stop_cg = MPI_Wtime();

          pcout << "CG+AMG elapsed time: " << stop_cg - start_cg << "[s]"
                << std::endl;

          pcout << "Initial value: " << solver_control.initial_value()
                << std::endl;
          pcout << "Converged (CG+AMG) in " << solver_control.last_step()
                << " iterations with value " << solver_control.last_value()
                << std::endl;


          if (dof_handler.n_dofs() < 3e6)
            {
              DataOut<dim> data_out;
              data_out.attach_dof_handler(dof_handler);
              data_out.add_data_vector(solution,
                                       "interpolated_solution",
                                       DataOut<dim>::type_dof_data);

              const std::string filename = "check_multigrid_mf_amg.vtu";
              std::ofstream     output(filename);
              data_out.build_patches(mapping);
              data_out.write_vtu(output);
            }
        }
    }
}



template <int dim>
void
TestMGMatrix<dim>::run()
{
  const unsigned int n_vect_doubles = VectorizedArray<double>::size();
  const unsigned int n_vect_bits    = 8 * sizeof(double) * n_vect_doubles;

  pcout << "Vectorization over " << n_vect_doubles
        << " doubles = " << n_vect_bits << " bits ("
        << Utilities::System::get_current_vectorization_level() << ')'
        << std::endl;
  make_fine_grid(3); // 6 global refinements of unit cube
  agglomerate_and_compute_level_matrices();
  pcout << std::endl;
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  const MPI_Comm                   comm = MPI_COMM_WORLD;
  static constexpr unsigned int    dim  = 2;

  if (Utilities::MPI::this_mpi_process(comm) == 0)
    std::cout << "Degree: " << degree_finite_element << std::endl;

  for (unsigned int starting_level = 0; starting_level < 3; ++starting_level)
    {
      TestMGMatrix<dim> problem(GridType::grid_generator,
                                degree_finite_element,
                                starting_level,
                                comm);
      problem.run();
    }
}
