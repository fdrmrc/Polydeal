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


#ifndef utils_h
#define utils_h

#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/multigrid/mg_tools.h>

#include <multigrid_amg.h>



template <int dim, typename RtreeType>
class Agglomerator;
template <int, int>
class AgglomerationHandler;



namespace Utils
{
  template <typename T>
  inline constexpr T
  constexpr_pow(T num, unsigned int pow)
  {
    return (pow >= sizeof(unsigned int) * 8) ? 0 :
           pow == 0                          ? 1 :
                                               num * constexpr_pow(num, pow - 1);
  }



  /**
   * We describe a undirect graph through a vector of nodes and its adjacency
  information
   */
  struct Graph
  {
    std::vector<types::global_cell_index>              nodes;
    std::vector<std::vector<types::global_cell_index>> adjacency;

    void
    print_graph() const
    {
      std::cout << "Graph information" << std::endl;
      for (const auto &node : nodes)
        {
          std::cout << "Neighbors for node " << node << std::endl;
          for (const auto &neigh_node : adjacency[node])
            std::cout << neigh_node << std::endl;
        }
    }
  };



  /**
   * Given a coarse AgglomerationHandler @p coarse_ah and a fine
   * AgglomerationHandler @p fine_ah, this function fills the injection matrix
   * @p matrix and the associated SparsityPattern @p sp from the coarse space
   * to the finer one.
   *
   * The matrix @p matrix (as well as @p sp) are assumed to be only
   * default-constructed upon calling this function, i.e. the matrix should
   * just be empty.
   *
   * @note Supported types are SparseMatrix<double> or TrilinosWrappers::SparseMatrix.
   */
  template <int dim, int spacedim, typename MatrixType>
  void
  fill_injection_matrix(const AgglomerationHandler<dim, spacedim> &coarse_ah,
                        const AgglomerationHandler<dim, spacedim> &fine_ah,
                        SparsityPattern                           &sp,
                        MatrixType                                &matrix)
  {
    using NumberType = typename MatrixType::value_type;
    // First, check that we support the matrix types
    static constexpr bool is_trilinos_matrix =
      std::is_same_v<TrilinosWrappers::SparseMatrix, MatrixType>;
    static constexpr bool is_serial_matrix =
      std::is_same_v<SparseMatrix<NumberType>, MatrixType>;
    static constexpr bool is_supported_matrix =
      is_trilinos_matrix || is_serial_matrix;
    static_assert(is_supported_matrix);
    if constexpr (is_trilinos_matrix)
      Assert(matrix.m() == 0, ExcInternalError());
    if constexpr (is_serial_matrix)
      Assert(sp.empty() && matrix.empty(),
             ExcMessage(
               "The destination matrix and its sparsity pattern must the empty "
               "upon calling this function."));

    Assert(coarse_ah.n_dofs() < fine_ah.n_dofs(), ExcInternalError());
    AssertDimension(dim, spacedim);

    // Get information from the handlers
    const DoFHandler<dim, spacedim> &coarse_agglo_dh = coarse_ah.agglo_dh;
    const DoFHandler<dim, spacedim> &fine_agglo_dh   = fine_ah.agglo_dh;

    const Mapping<dim, spacedim>       &mapping = fine_ah.get_mapping();
    const FiniteElement<dim, spacedim> &fe      = coarse_ah.get_fe();
    const Triangulation<dim, spacedim> &tria    = coarse_ah.get_triangulation();
    const auto &fine_bboxes                     = fine_ah.get_local_bboxes();
    const auto &coarse_bboxes                   = coarse_ah.get_local_bboxes();

    const IndexSet &locally_owned_dofs_fine =
      fine_agglo_dh.locally_owned_dofs();
    const IndexSet locally_relevant_dofs_fine =
      DoFTools::extract_locally_relevant_dofs(fine_agglo_dh);

    const IndexSet &locally_owned_dofs_coarse =
      coarse_agglo_dh.locally_owned_dofs();

    std::conditional_t<is_trilinos_matrix,
                       TrilinosWrappers::SparsityPattern,
                       DynamicSparsityPattern>
      dsp;

    if constexpr (is_trilinos_matrix)
      dsp.reinit(locally_owned_dofs_fine,
                 locally_owned_dofs_coarse,
                 tria.get_mpi_communicator());
    else
      dsp.reinit(fine_agglo_dh.n_dofs(),
                 coarse_agglo_dh.n_dofs(),
                 locally_relevant_dofs_fine);

    const unsigned int                   dofs_per_cell = fe.dofs_per_cell;
    std::vector<types::global_dof_index> agglo_dof_indices(dofs_per_cell);
    std::vector<types::global_dof_index> standard_dof_indices(dofs_per_cell);
    std::vector<types::global_dof_index> output_dof_indices(dofs_per_cell);

    const std::vector<Point<dim>> &unit_support_points =
      fe.get_unit_support_points();
    Quadrature<dim>         quad(unit_support_points);
    FEValues<dim, spacedim> output_fe_values(mapping,
                                             fe,
                                             quad,
                                             update_quadrature_points);

    std::vector<types::global_dof_index> local_dof_indices_coarse(
      dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices_child(dofs_per_cell);

    for (const auto &polytope : coarse_ah.polytope_iterators())
      if (polytope->is_locally_owned())
        {
          polytope->get_dof_indices(local_dof_indices_coarse);

          // Get local children and their DoFs
          const auto &children_polytopes = polytope->children();
          for (const types::global_cell_index child_idx : children_polytopes)
            {
              const typename DoFHandler<dim>::active_cell_iterator &child_dh =
                fine_ah.polytope_to_dh_iterator(child_idx);
              child_dh->get_dof_indices(local_dof_indices_child);
              for (const auto row : local_dof_indices_child)
                dsp.add_entries(row,
                                local_dof_indices_coarse.begin(),
                                local_dof_indices_coarse.end());
            }
        }

    const auto assemble_injection_matrix = [&]() {
      FullMatrix<NumberType>  local_matrix(dofs_per_cell, dofs_per_cell);
      std::vector<Point<dim>> reference_q_points(dofs_per_cell);

      // Dummy AffineConstraints, only needed for loc2glb
      AffineConstraints<NumberType> c;
      c.close();

      for (const auto &polytope : coarse_ah.polytope_iterators())
        if (polytope->is_locally_owned())
          {
            polytope->get_dof_indices(local_dof_indices_coarse);
            const BoundingBox<dim> &coarse_bbox =
              coarse_bboxes[polytope->index()];

            // Get local children of the present polytope
            const auto &children_polytopes = polytope->children();
            for (const types::global_cell_index child_idx : children_polytopes)
              {
                const BoundingBox<dim> &fine_bbox = fine_bboxes[child_idx];
                const typename DoFHandler<dim>::active_cell_iterator &child_dh =
                  fine_ah.polytope_to_dh_iterator(child_idx);
                child_dh->get_dof_indices(local_dof_indices_child);

                local_matrix = 0.;

                // compute real location of support points
                std::vector<Point<dim>> real_qpoints;
                real_qpoints.reserve(unit_support_points.size());
                for (const Point<dim> &p : unit_support_points)
                  real_qpoints.push_back(fine_bbox.unit_to_real(p));

                for (unsigned int i = 0; i < local_dof_indices_coarse.size();
                     ++i)
                  {
                    const auto &p = coarse_bbox.real_to_unit(real_qpoints[i]);
                    for (unsigned int j = 0; j < local_dof_indices_child.size();
                         ++j)
                      {
                        local_matrix(i, j) = fe.shape_value(j, p);
                      }
                  }

                c.distribute_local_to_global(local_matrix,
                                             local_dof_indices_child,
                                             local_dof_indices_coarse,
                                             matrix);
              }
          }
    };


    if constexpr (is_trilinos_matrix)
      {
        dsp.compress();
        matrix.reinit(dsp);
        assemble_injection_matrix();
      }
    else if constexpr (is_serial_matrix)
      {
        sp.copy_from(dsp);
        matrix.reinit(sp);
        assemble_injection_matrix();
      }
    else
      {
        // PETSc types not implemented.
        (void)coarse_ah;
        (void)fine_ah;
        (void)sp;
        (void)matrix;
        AssertThrow(false,
                    ExcNotImplemented(
                      "This injection does not support PETSc types."));
      }

    // If tria is distributed
    if (dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
          &tria) != nullptr)
      matrix.compress(VectorOperation::add);
  }



  /**
   * This class implements the interface required by MGCoarseGridBase and
   * allows a direct to be used as coarse grid solver in the Multigrid
   * framework. The SolverType is provided as template parameter.
   *
   * @note Supported direct solver types are: SparseDirectUMFPACK and
   * TrilinosWrappers::SolverDirect. Available vector types are the ones
   * supported by the respective solve() (or vmult()) interfaces by such types.
   * This means that SparseDirectUMFPACK supports only serial deal.II vectors,
   * while TrilinosWrappers::SolverDirect supports as parallel vector types:
   * LinearAlgebra::distributed::Vector and TrilinosWrappers::MPI::Vector.
   */
  template <typename VectorType, typename MatrixType, typename SolverType>
  class MGCoarseDirect : public MGCoarseGridBase<VectorType>
  {
  public:
    explicit MGCoarseDirect(const MatrixType &matrix)
      : direct_solver(TrilinosWrappers::SolverDirect::AdditionalData())
      , coarse_matrix(matrix)
    {
      // Check if matrix types are supported
      static constexpr bool is_serial_matrix =
        std::is_same_v<SolverType, SparseDirectUMFPACK>;
      static constexpr bool is_trilinos_matrix =
        std::is_same_v<SolverType, TrilinosWrappers::SolverDirect>;
      [[maybe_unused]] static constexpr bool is_matrix_type_supported =
        is_serial_matrix || is_trilinos_matrix;
      Assert(is_matrix_type_supported, ExcNotImplemented());

      // Check on the vector types: standard deal.II vectors, LA::d::V, or
      // Trilinos vectors.
      if constexpr (is_serial_matrix)
        {
          [[maybe_unused]] static constexpr bool is_serial_vector =
            std::is_same_v<VectorType,
                           dealii::Vector<typename MatrixType::value_type>>;
          Assert(is_serial_vector, ExcNotImplemented());
        }
      else if constexpr (is_trilinos_matrix)
        {
          [[maybe_unused]] static constexpr bool is_supported_parallel_vector =
            std::is_same_v<VectorType,
                           LinearAlgebra::distributed::Vector<
                             typename MatrixType::value_type>> ||
            std::is_same_v<VectorType, TrilinosWrappers::MPI::Vector>;
          Assert(is_supported_parallel_vector, ExcNotImplemented());
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
          // DEAL_II_NOT_IMPLEMENTED(); //not available in older releases
        }


      // regardless of UMFPACK or Trilinos, both direct solvers need a call
      // to `initialize()`.
      direct_solver.initialize(coarse_matrix);
    }


    void
    operator()(const unsigned int,
               VectorType       &dst,
               const VectorType &src) const override
    {
      AssertDimension(coarse_matrix.n(), src.size());
      AssertDimension(coarse_matrix.m(), dst.size());
      const_cast<SolverType &>(direct_solver).solve(dst, src);
    }


    ~MGCoarseDirect() = default;

  private:
    SolverType        direct_solver;
    const MatrixType &coarse_matrix;
  };



  namespace Physics
  {
    struct BilinearFormParameters
    {
      double dt               = 1e-4;
      double penalty_constant = 10.;
      double chi              = 1e5;
      double Cm               = 1e-2;
      double sigma            = 12e-2;
    };

  } // namespace Physics

  namespace MatrixFreeOperators
  {
    /**
     * Class that implements the action of symmetric interior penalty operator
     * on a vector.
     * @note The member function rhs() assume the source term is 1. Different source terms
     * can be defined by suitably changing that function.
     */
    template <int dim,
              int degree,
              int n_q_points,
              int n_components,
              typename number = double>
    class LaplaceOperatorDG : public Subscriptor
    {
    public:
      using value_type          = number;
      using VectorizedArrayType = VectorizedArray<number>;
      using VectorType          = LinearAlgebra::distributed::Vector<number>;


      LaplaceOperatorDG(){};

      void
      reinit(const Mapping<dim>    &mapping,
             const DoFHandler<dim> &dof_handler,
             const unsigned int     level = numbers::invalid_unsigned_int)
      {
        fe_degree = dof_handler.get_fe().degree;

        const QGauss<1>                                  quad(n_q_points);
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
        data.loop(&LaplaceOperatorDG::local_apply,
                  &LaplaceOperatorDG::local_apply_face,
                  &LaplaceOperatorDG::local_apply_boundary,
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
              VectorizedArrayType jump_value =
                (phi_m.get_value(q) - phi_p.get_value(q)) * 0.5;
              VectorizedArrayType average_valgrad =
                phi_m.get_normal_derivative(q) + phi_p.get_normal_derivative(q);
              average_valgrad =
                jump_value * 2. * sigmaF - average_valgrad * 0.5;
              phi_m.submit_normal_derivative(-jump_value, q);
              phi_p.submit_normal_derivative(-jump_value, q);
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
              VectorizedArrayType jump_value = phi_m.get_value(q);
              VectorizedArrayType average_valgrad =
                -phi_m.get_normal_derivative(q);
              average_valgrad += jump_value * sigmaF * 2.0;
              phi_m.submit_normal_derivative(-jump_value, q);
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
              DoFTools::make_flux_sparsity_pattern(dof_handler,
                                                   dsp,
                                                   constraints);

            dsp.compress();
            system_matrix.reinit(dsp);

            // Assemble system matrix. Notice that degree 1 has been hardcoded.

            MatrixFreeTools::compute_matrix<dim,
                                            degree,
                                            n_q_points,
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
              VectorizedArrayType jump_value =
                (phi_m.get_value(q) - phi_p.get_value(q)) * 0.5;
              VectorizedArrayType average_valgrad =
                phi_m.get_normal_derivative(q) + phi_p.get_normal_derivative(q);
              average_valgrad =
                jump_value * 2. * sigmaF - average_valgrad * 0.5;
              phi_m.submit_normal_derivative(-jump_value, q);
              phi_p.submit_normal_derivative(-jump_value, q);
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
              VectorizedArrayType jump_value = phi_m.get_value(q);
              VectorizedArrayType average_valgrad =
                -phi_m.get_normal_derivative(q);
              average_valgrad += jump_value * sigmaF * 2.0;
              phi_m.submit_normal_derivative(-jump_value, q);
              phi_m.submit_value(average_valgrad, q);
            }

          phi_m.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        };


        //////////////////////////////////////////////////

        // Check if matrix has already been set up.
        AssertThrow((mg_matrix.m() == 0 && mg_matrix.n() == 0),
                    ExcInternalError());

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

        // Assemble system matrix.
        MatrixFreeTools::compute_matrix<dim,
                                        degree,
                                        n_q_points,
                                        n_components,
                                        number,
                                        VectorizedArrayType>(
          data,
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


        data
          .template cell_loop<LinearAlgebra::distributed::Vector<number>, int>(
            [](const auto &matrix_free,
               auto       &dst,
               const auto &,
               const auto cells) {
              FEEvaluation<dim, -1, 0, n_components, number> phi(matrix_free,
                                                                 cells);
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
        data.loop(&LaplaceOperatorDG::local_diagonal_cell,
                  &LaplaceOperatorDG::local_diagonal_face,
                  &LaplaceOperatorDG::local_diagonal_boundary,
                  this,
                  inverse_diagonal_entries,
                  dummy);

        for (unsigned int i = 0;
             i < inverse_diagonal_entries.locally_owned_size();
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

        for (unsigned int cell = cell_range.first; cell < cell_range.second;
             ++cell)
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

        for (unsigned int face = face_range.first; face < face_range.second;
             ++face)
          {
            fe_eval.reinit(face);
            fe_eval_neighbor.reinit(face);

            fe_eval.read_dof_values(src);
            fe_eval.evaluate(EvaluationFlags::values |
                             EvaluationFlags::gradients);
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
                VectorizedArray<number> jump_value =
                  (fe_eval.get_value(q) - fe_eval_neighbor.get_value(q)) * 0.5;
                VectorizedArray<number> average_valgrad =
                  fe_eval.get_normal_derivative(q) +
                  fe_eval_neighbor.get_normal_derivative(q);
                average_valgrad =
                  jump_value * 2. * sigmaF - average_valgrad * 0.5;
                fe_eval.submit_normal_derivative(-jump_value, q);
                fe_eval_neighbor.submit_normal_derivative(-jump_value, q);
                fe_eval.submit_value(average_valgrad, q);
                fe_eval_neighbor.submit_value(-average_valgrad, q);
              }
            fe_eval.integrate(EvaluationFlags::values |
                              EvaluationFlags::gradients);
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
        for (unsigned int face = face_range.first; face < face_range.second;
             ++face)
          {
            fe_eval.reinit(face);
            fe_eval.read_dof_values(src);
            fe_eval.evaluate(EvaluationFlags::values |
                             EvaluationFlags::gradients);
            VectorizedArray<number> sigmaF =
              std::abs((fe_eval.normal_vector(0) *
                        fe_eval.inverse_jacobian(0))[dim - 1]) *
              number(std::max(fe_degree, 1) * (fe_degree + 1.0)) * 2.0;

            for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
              {
                VectorizedArray<number> jump_value = fe_eval.get_value(q);
                VectorizedArray<number> average_valgrad =
                  -fe_eval.get_normal_derivative(q);
                average_valgrad += jump_value * sigmaF * 2.0;
                fe_eval.submit_normal_derivative(-jump_value, q);
                fe_eval.submit_value(average_valgrad, q);
              }

            fe_eval.integrate(EvaluationFlags::values |
                              EvaluationFlags::gradients);
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

        for (unsigned int cell = cell_range.first; cell < cell_range.second;
             ++cell)
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

        for (unsigned int face = face_range.first; face < face_range.second;
             ++face)
          {
            phi.reinit(face);
            phi_outer.reinit(face);

            VectorizedArray<number> sigmaF =
              (std::abs(
                 (phi.normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]) +
               std::abs((phi.normal_vector(0) *
                         phi_outer.inverse_jacobian(0))[dim - 1])) *
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
                phi.evaluate(EvaluationFlags::values |
                             EvaluationFlags::gradients);

                for (unsigned int q = 0; q < phi.n_q_points; ++q)
                  {
                    VectorizedArray<number> jump_value =
                      (phi.get_value(q) - phi_outer.get_value(q)) * 0.5;
                    VectorizedArray<number> average_valgrad =
                      phi.get_normal_derivative(q) +
                      phi_outer.get_normal_derivative(q);
                    average_valgrad =
                      jump_value * 2. * sigmaF - average_valgrad * 0.5;
                    phi.submit_normal_derivative(-jump_value, q);
                    phi.submit_value(average_valgrad, q);
                  }
                phi.integrate(EvaluationFlags::values |
                              EvaluationFlags::gradients);
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
                    VectorizedArray<number> jump_value =
                      (phi.get_value(q) - phi_outer.get_value(q)) * 0.5;
                    VectorizedArray<number> average_valgrad =
                      phi.get_normal_derivative(q) +
                      phi_outer.get_normal_derivative(q);
                    average_valgrad =
                      jump_value * 2. * sigmaF - average_valgrad * 0.5;
                    phi_outer.submit_normal_derivative(-jump_value, q);
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

        for (unsigned int face = face_range.first; face < face_range.second;
             ++face)
          {
            phi.reinit(face);

            VectorizedArray<number> sigmaF =
              std::abs(
                (phi.normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]) *
              number(std::max(fe_degree, 1) * (fe_degree + 1.0)) * 2.0;

            for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
                  phi.begin_dof_values()[j] = VectorizedArray<number>();
                phi.begin_dof_values()[i] = 1.;
                phi.evaluate(EvaluationFlags::values |
                             EvaluationFlags::gradients);

                for (unsigned int q = 0; q < phi.n_q_points; ++q)
                  {
                    VectorizedArray<number> jump_value = phi.get_value(q);
                    VectorizedArray<number> average_valgrad =
                      -phi.get_normal_derivative(q);
                    average_valgrad += jump_value * sigmaF * 2.0;
                    phi.submit_normal_derivative(-jump_value, q);
                    phi.submit_value(average_valgrad, q);
                  }

                phi.integrate(EvaluationFlags::values |
                              EvaluationFlags::gradients);
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



    /**
     * Action of SIPDG form on a vector for the monodomain problem.
     */
    template <int dim,
              int degree,
              int n_q_points,
              int n_components,
              typename number = double>
    class MonodomainOperatorDG : public Subscriptor
    {
    public:
      using value_type          = number;
      using VectorizedArrayType = VectorizedArray<number>;
      using VectorType          = LinearAlgebra::distributed::Vector<number>;

      MonodomainOperatorDG(const Physics::BilinearFormParameters &parameters_)
      {
        parameters = parameters_;
      };

      void
      reinit(const Mapping<dim>    &mapping,
             const DoFHandler<dim> &dof_handler,
             const unsigned int     level = numbers::invalid_unsigned_int)
      {
        Assert(
          degree == dof_handler.get_fe().degree,
          ExcMessage(
            "Degree of the operator must match the degree of the DoFHandler"));
        fe_degree = dof_handler.get_fe().degree;

        const QGauss<1>                                  quad(n_q_points);
        typename MatrixFree<dim, number>::AdditionalData addit_data;
        addit_data.tasks_parallel_scheme =
          MatrixFree<dim, number>::AdditionalData::none;
        addit_data.tasks_block_size = 3;
        addit_data.mg_level         = level;
        addit_data.mapping_update_flags =
          (update_values | update_gradients | update_quadrature_points);
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
        data.loop(&MonodomainOperatorDG::local_apply,
                  &MonodomainOperatorDG::local_apply_face,
                  &MonodomainOperatorDG::local_apply_boundary,
                  this,
                  dst,
                  src,
                  /*zero_dst =*/false,
                  MatrixFree<dim, number>::DataAccessOnFaces::gradients,
                  MatrixFree<dim, number>::DataAccessOnFaces::gradients);
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
        const double factor = (parameters.chi * parameters.Cm) / parameters.dt;

        const auto cell_operation = [&](auto &phi) {
          phi.evaluate(EvaluationFlags::gradients | EvaluationFlags::values);
          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            {
              phi.submit_value(phi.get_value(q) * factor, q);
              phi.submit_gradient(phi.get_gradient(q) * parameters.sigma, q);
            }
          phi.integrate(EvaluationFlags::gradients | EvaluationFlags::values);
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
              VectorizedArrayType jump_value =
                (phi_m.get_value(q) - phi_p.get_value(q)) * 0.5;
              VectorizedArrayType average_valgrad =
                phi_m.get_normal_derivative(q) + phi_p.get_normal_derivative(q);
              average_valgrad =
                jump_value * 2. * sigmaF - average_valgrad * 0.5;
              phi_m.submit_normal_derivative(-parameters.sigma * jump_value, q);
              phi_p.submit_normal_derivative(-parameters.sigma * jump_value, q);
              phi_m.submit_value(parameters.sigma * average_valgrad, q);
              phi_p.submit_value(-parameters.sigma * average_valgrad, q);
            }
          phi_m.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
          phi_p.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        };

        // const auto boundary_operation = [&](auto &phi_m) {
        //   (void)phi_m;
        //   AssertThrow(
        //     false,
        //     ExcMessage(
        //       "We have Neumann BCs only, so this should never be called."));
        //   // Do nothing since we have homogeneous Neumann BCs
        // };


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
              DoFTools::make_flux_sparsity_pattern(dof_handler,
                                                   dsp,
                                                   constraints);

            dsp.compress();
            system_matrix.reinit(dsp);

            // Assemble system matrix. Notice that degree 1 has been hardcoded.

            MatrixFreeTools::compute_matrix<dim,
                                            degree,
                                            n_q_points,
                                            n_components,
                                            number,
                                            VectorizedArrayType>(
              data,
              constraints,
              system_matrix,
              cell_operation,
              face_operation,
              nullptr /*boundary_operation*/);
          }

        return system_matrix;
      }



      void
      get_system_matrix(TrilinosWrappers::SparseMatrix &mg_matrix) const
      {
        // Boilerplate for SIP-DG form. TODO: unify interface.
        //////////////////////////////////////////////////
        const double factor = (parameters.chi * parameters.Cm) / parameters.dt;

        const auto cell_operation = [&](auto &phi) {
          phi.evaluate(EvaluationFlags::gradients | EvaluationFlags::values);
          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            {
              phi.submit_value(phi.get_value(q) * factor, q);
              phi.submit_gradient(phi.get_gradient(q) * parameters.sigma, q);
            }
          phi.integrate(EvaluationFlags::gradients | EvaluationFlags::values);
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
              VectorizedArrayType jump_value =
                (phi_m.get_value(q) - phi_p.get_value(q)) * 0.5;
              VectorizedArrayType average_valgrad =
                phi_m.get_normal_derivative(q) + phi_p.get_normal_derivative(q);
              average_valgrad =
                jump_value * 2. * sigmaF - average_valgrad * 0.5;
              phi_m.submit_normal_derivative(-parameters.sigma * jump_value, q);
              phi_p.submit_normal_derivative(-parameters.sigma * jump_value, q);
              phi_m.submit_value(parameters.sigma * average_valgrad, q);
              phi_p.submit_value(-parameters.sigma * average_valgrad, q);
            }
          phi_m.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
          phi_p.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        };


        //////////////////////////////////////////////////

        // Check if matrix has already been set up.
        AssertThrow((mg_matrix.m() == 0 && mg_matrix.n() == 0),
                    ExcInternalError());

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

        // Assemble system matrix.
        MatrixFreeTools::compute_matrix<dim,
                                        degree,
                                        n_q_points,
                                        n_components,
                                        number,
                                        VectorizedArrayType>(
          data,
          constraints,
          mg_matrix,
          cell_operation,
          face_operation,
          nullptr /*boundary_operation*/);
      }


      void
      compute_inverse_diagonal()
      {
        data.initialize_dof_vector(inverse_diagonal_entries);
        unsigned int dummy = 0;
        data.loop(&MonodomainOperatorDG::local_diagonal_cell,
                  &MonodomainOperatorDG::local_diagonal_face,
                  &MonodomainOperatorDG::local_diagonal_boundary,
                  this,
                  inverse_diagonal_entries,
                  dummy);

        for (unsigned int i = 0;
             i < inverse_diagonal_entries.locally_owned_size();
             ++i)
          if (std::abs(inverse_diagonal_entries.local_element(i)) > 1e-10)
            inverse_diagonal_entries.local_element(i) =
              1. / inverse_diagonal_entries.local_element(i);
      }



      void
      apply_mass_term(
        LinearAlgebra::distributed::Vector<number>       &dst,
        const LinearAlgebra::distributed::Vector<number> &src) const
      {
        FEEvaluation<dim, degree, n_q_points, n_components, number> phi(data);
        const double factor = (parameters.chi * parameters.Cm) / parameters.dt;

        for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
          {
            phi.reinit(cell);
            phi.read_dof_values(src);
            phi.evaluate(EvaluationFlags::values);
            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              phi.submit_value(phi.get_value(q) * factor, q);
            phi.integrate(EvaluationFlags::values);
            phi.distribute_local_to_global(dst);
          }
        dst.compress(VectorOperation::add);
      }



      void
      rhs(LinearAlgebra::distributed::Vector<number>       &rhs,
          const LinearAlgebra::distributed::Vector<number> &solution_minus_ion,
          const Function<dim>                              &Iext) const
      {
        FEEvaluation<dim, degree, n_q_points, n_components, number> phi(data);

        for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
          {
            phi.reinit(cell);
            phi.read_dof_values(solution_minus_ion);
            phi.evaluate(EvaluationFlags::values);
            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              {
                const Point<dim, VectorizedArray<number>> p_vect =
                  phi.quadrature_point(q);
                // evaluate the external current for each component in
                // VectorizedArray
                VectorizedArray<number> applied_current_value = 0.0;
                for (unsigned int v = 0; v < VectorizedArray<number>::size();
                     ++v)
                  {
                    Point<dim> p;
                    for (unsigned int d = 0; d < dim; ++d)
                      p[d] = p_vect[d][v];
                    applied_current_value[v] = Iext.value(p);
                  }

                //  reaction_term =  (parameters.chi * phi.get_value(q)) +
                //  applied_current_value;
                phi.submit_value((parameters.chi * phi.get_value(q)) +
                                   applied_current_value,
                                 q);
              }
            phi.integrate(EvaluationFlags::values);
            phi.distribute_local_to_global(rhs);
          }
        rhs.compress(VectorOperation::add);
      }



    private:
      void
      local_apply(const MatrixFree<dim, number>                    &data,
                  LinearAlgebra::distributed::Vector<number>       &dst,
                  const LinearAlgebra::distributed::Vector<number> &src,
                  const std::pair<unsigned int, unsigned int> &cell_range) const
      {
        FEEvaluation<dim, degree, n_q_points, n_components, number> phi(data);
        const double factor = (parameters.chi * parameters.Cm) / parameters.dt;

        for (unsigned int cell = cell_range.first; cell < cell_range.second;
             ++cell)
          {
            phi.reinit(cell);
            phi.read_dof_values(src);
            phi.evaluate(EvaluationFlags::gradients | EvaluationFlags::values);
            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              {
                phi.submit_value(phi.get_value(q) * factor, q);
                phi.submit_gradient(phi.get_gradient(q) * parameters.sigma, q);
              }
            phi.integrate(EvaluationFlags::gradients | EvaluationFlags::values);
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
        FEFaceEvaluation<dim, degree, n_q_points, n_components, number> fe_eval(
          data, true);
        FEFaceEvaluation<dim, degree, n_q_points, n_components, number>
          fe_eval_neighbor(data, false);

        for (unsigned int face = face_range.first; face < face_range.second;
             ++face)
          {
            fe_eval.reinit(face);
            fe_eval_neighbor.reinit(face);

            fe_eval.read_dof_values(src);
            fe_eval.evaluate(EvaluationFlags::values |
                             EvaluationFlags::gradients);
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
                VectorizedArray<number> jump_value =
                  (fe_eval.get_value(q) - fe_eval_neighbor.get_value(q)) * 0.5;
                VectorizedArray<number> average_valgrad =
                  fe_eval.get_normal_derivative(q) +
                  fe_eval_neighbor.get_normal_derivative(q);
                average_valgrad =
                  jump_value * 2. * sigmaF - average_valgrad * 0.5;
                fe_eval.submit_normal_derivative(-parameters.sigma * jump_value,
                                                 q);
                fe_eval_neighbor.submit_normal_derivative(-parameters.sigma *
                                                            jump_value,
                                                          q);
                fe_eval.submit_value(parameters.sigma * average_valgrad, q);
                fe_eval_neighbor.submit_value(-parameters.sigma *
                                                average_valgrad,
                                              q);
              }
            fe_eval.integrate(EvaluationFlags::values |
                              EvaluationFlags::gradients);
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
        (void)data;
        (void)dst;
        (void)src;
        (void)face_range;
        // Do nothing since we have homogeneous Neumann BCs
      }


      void
      local_diagonal_cell(
        const MatrixFree<dim, number>              &data,
        LinearAlgebra::distributed::Vector<number> &dst,
        const unsigned int &,
        const std::pair<unsigned int, unsigned int> &cell_range) const
      {
        FEEvaluation<dim, degree, n_q_points, n_components, number> phi(data);
        AlignedVector<VectorizedArray<number>> local_diagonal_vector(
          phi.dofs_per_cell);
        const double factor = (parameters.chi * parameters.Cm) / parameters.dt;

        for (unsigned int cell = cell_range.first; cell < cell_range.second;
             ++cell)
          {
            phi.reinit(cell);

            for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
                  phi.begin_dof_values()[j] = VectorizedArray<number>();
                phi.begin_dof_values()[i] = 1.;
                phi.evaluate(EvaluationFlags::gradients |
                             EvaluationFlags::values);
                for (unsigned int q = 0; q < phi.n_q_points; ++q)
                  {
                    phi.submit_value(phi.get_value(q) * factor, q);
                    phi.submit_gradient(phi.get_gradient(q) * parameters.sigma,
                                        q);
                  }
                phi.integrate(EvaluationFlags::gradients |
                              EvaluationFlags::values);
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
        FEFaceEvaluation<dim, degree, n_q_points, n_components, number> phi(
          data, true);
        FEFaceEvaluation<dim, degree, n_q_points, n_components, number>
                                               phi_outer(data, false);
        AlignedVector<VectorizedArray<number>> local_diagonal_vector(
          phi.dofs_per_cell);

        for (unsigned int face = face_range.first; face < face_range.second;
             ++face)
          {
            phi.reinit(face);
            phi_outer.reinit(face);

            VectorizedArray<number> sigmaF =
              (std::abs(
                 (phi.normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]) +
               std::abs((phi.normal_vector(0) *
                         phi_outer.inverse_jacobian(0))[dim - 1])) *
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
                phi.evaluate(EvaluationFlags::values |
                             EvaluationFlags::gradients);

                for (unsigned int q = 0; q < phi.n_q_points; ++q)
                  {
                    VectorizedArray<number> jump_value =
                      (phi.get_value(q) - phi_outer.get_value(q)) * 0.5;
                    VectorizedArray<number> average_valgrad =
                      phi.get_normal_derivative(q) +
                      phi_outer.get_normal_derivative(q);
                    average_valgrad =
                      jump_value * 2. * sigmaF - average_valgrad * 0.5;
                    phi.submit_normal_derivative(-parameters.sigma * jump_value,
                                                 q);
                    phi.submit_value(parameters.sigma * average_valgrad, q);
                  }
                phi.integrate(EvaluationFlags::values |
                              EvaluationFlags::gradients);
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
                    VectorizedArray<number> jump_value =
                      (phi.get_value(q) - phi_outer.get_value(q)) * 0.5;
                    VectorizedArray<number> average_valgrad =
                      phi.get_normal_derivative(q) +
                      phi_outer.get_normal_derivative(q);
                    average_valgrad =
                      jump_value * 2. * sigmaF - average_valgrad * 0.5;
                    phi_outer.submit_normal_derivative(-parameters.sigma *
                                                         jump_value,
                                                       q);
                    phi_outer.submit_value(-parameters.sigma * average_valgrad,
                                           q);
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
        (void)data;
        (void)dst;
        (void)face_range;
        // Do nothing since we have homogeneous Neumann BCs
      }



      MatrixFree<dim, number>                    data;
      LinearAlgebra::distributed::Vector<number> inverse_diagonal_entries;
      int                                        fe_degree;
      mutable TrilinosWrappers::SparseMatrix     system_matrix;
      AffineConstraints<number>                  constraints;
      Physics::BilinearFormParameters            parameters;
    };



  } // namespace MatrixFreeOperators



  /**
   * Helper function to compute the position of index @p index in vector @p v.
   */
  inline types::global_cell_index
  get_index(const std::vector<types::global_cell_index> &v,
            const types::global_cell_index               index)
  {
    return std::distance(v.begin(), std::find(v.begin(), v.end(), index));
  }



  template <int dim>
  void
  create_graph_from_agglomerate(
    const std::vector<typename Triangulation<dim>::active_cell_iterator> &cells,
    Graph                                                                &g)
  {
    Assert(cells.size() > 0, ExcMessage("No cells to be agglomerated."));
    const unsigned int n_faces = cells[0]->n_faces();

    std::vector<types::global_cell_index> vec_cells(cells.size());
    for (size_t i = 0; i < cells.size(); i++)
      vec_cells[i] = cells[i]->active_cell_index();

    g.adjacency.resize(cells.size());
    for (const auto &cell : cells)
      {
        // std::cout << "Cell idx: " << cell->active_cell_index() <<
        // std::endl; std::cout << "new idx: "
        //           << get_index(vec_cells, cell->active_cell_index())
        //           << std::endl;
        g.nodes.push_back(get_index(vec_cells, cell->active_cell_index()));
        for (unsigned int f = 0; f < n_faces; ++f)
          {
            const auto &neigh = cell->neighbor(f);
            if (neigh.state() == IteratorState::IteratorStates::valid &&
                std::find(cells.begin(), cells.end(), neigh) != std::end(cells))
              g.adjacency[get_index(vec_cells, cell->active_cell_index())]
                .push_back(get_index(vec_cells, neigh->active_cell_index()));
          }
      }
  }



  inline void
  dfs(std::vector<types::global_cell_index> &comp,
      std::vector<bool>                     &visited,
      const Graph                           &g,
      const types::global_cell_index         v)
  {
    visited[v] = true;
    comp.push_back(v);
    for (const types::global_cell_index u : g.adjacency[v])
      {
        if (!visited[u])
          dfs(comp, visited, g, u);
      }
  }



  void
  compute_connected_components(
    Graph                                              &g,
    std::vector<std::vector<types::global_cell_index>> &connected_components)
  {
    Assert(g.nodes.size() > 0, ExcMessage("No nodes in this graph."));
    Assert(
      connected_components.size() == 0,
      ExcMessage(
        "Connected components have to be computed by the present function."));

    std::vector<bool> visited(g.nodes.size()); // register visited node
    std::fill(visited.begin(), visited.end(), 0);


    for (types::global_cell_index v : g.nodes)
      {
        if (!visited[v])
          {
            connected_components.emplace_back();
            dfs(connected_components.back(), visited, g, v);
          }
      }
  }


} // namespace Utils



#endif
