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
    else if constexpr (is_serial_matrix)
      Assert(sp.empty() && matrix.empty(),
             ExcMessage(
               "The destination matrix and its sparsity pattern must the empty "
               "upon calling this function."));

    Assert(coarse_ah.n_dofs() < fine_ah.n_dofs(), ExcInternalError());
    AssertDimension(dim, spacedim);

    // Get information from the handlers
    const DoFHandler<dim, spacedim> &coarse_agglo_dh = coarse_ah.agglo_dh;
    const DoFHandler<dim, spacedim> &fine_agglo_dh   = fine_ah.agglo_dh;

    const FiniteElement<dim, spacedim> &fe   = coarse_ah.get_fe();
    const Triangulation<dim, spacedim> &tria = coarse_ah.get_triangulation();
    const auto &fine_bboxes                  = fine_ah.get_local_bboxes();
    const auto &coarse_bboxes                = coarse_ah.get_local_bboxes();

    const IndexSet &locally_owned_dofs_fine =
      fine_agglo_dh.locally_owned_dofs();
    const IndexSet locally_relevant_dofs_fine =
      DoFTools::extract_locally_relevant_dofs(fine_agglo_dh);

    const IndexSet &locally_owned_dofs_coarse =
      coarse_agglo_dh.locally_owned_dofs();

    DynamicSparsityPattern               dsp(fine_agglo_dh.n_dofs(),
                               coarse_agglo_dh.n_dofs());
    const unsigned int                   dofs_per_cell = fe.dofs_per_cell;
    std::vector<types::global_dof_index> agglo_dof_indices(dofs_per_cell);
    std::vector<types::global_dof_index> standard_dof_indices(dofs_per_cell);
    std::vector<types::global_dof_index> output_dof_indices(dofs_per_cell);

    const std::vector<Point<dim>> &unit_support_points =
      fe.get_unit_support_points();
    Quadrature<dim>         quad(unit_support_points);
    FEValues<dim, spacedim> output_fe_values(fe,
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
        const MPI_Comm &communicator = tria.get_communicator();
        SparsityTools::distribute_sparsity_pattern(dsp,
                                                   locally_owned_dofs_fine,
                                                   communicator,
                                                   locally_relevant_dofs_fine);

        matrix.reinit(locally_owned_dofs_fine,
                      locally_owned_dofs_coarse,
                      dsp,
                      communicator);
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
   * supported by the respective vmult() interfaces by such types. This means
   * that SparseDirectUMFPACK supports only serial deal.II vectors, while
   * TrilinosWrappers::SolverDirect supports as parallel vector types:
   * LinearAlgebra::distributed::Vector and TrilinosWrappers::MPI::Vector.
   */
  template <typename VectorType, typename MatrixType, typename SolverType>
  class MGCoarseDirect : public MGCoarseGridBase<VectorType>
  {
  public:
    explicit MGCoarseDirect(const MatrixType &matrix)
      : coarse_matrix(matrix)
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
          DEAL_II_NOT_IMPLEMENTED();
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
      double start, stop;
      start = MPI_Wtime();
      direct_solver.vmult(dst, src);
      stop          = MPI_Wtime();
      MPI_Comm comm = dst.get_mpi_communicator();
      if (Utilities::MPI::this_mpi_process(comm) == 0)
        std::cout << "Direct solver elapsed time: " << stop - start << "[s]"
                  << std::endl;
    }


    ~MGCoarseDirect() = default;

  private:
    SolverType        direct_solver;
    const MatrixType &coarse_matrix;
  };



  template <int dim,
            typename VectorType,
            typename MatrixType,
            typename SolverType = SolverCG<VectorType>>
  class MGCoarsePolytopalSolver : public MGCoarseGridBase<VectorType>
  {
  public:
    MGCoarsePolytopalSolver() = delete;

    MGCoarsePolytopalSolver(
      const MatrixType                               &coarse_level_operator_,
      const DoFHandler<dim>                          &coarse_dof_handler,
      const mg::Matrix<VectorType>                   &mg_matrix,
      const TrilinosWrappers::SparseMatrix           &coarse_trilinos_matrix,
      const MGTransferAgglomeration<dim, VectorType> &transfer,
      const MGSmootherBase<VectorType>               &pre_smooth,
      const MGSmootherBase<VectorType>               &post_smooth,
      SolverControl                                  &solver_control)
      : comm(coarse_level_operator_.get_dof_handler().get_communicator())
      , solver(solver_control)
    {
      control = &solver_control;
      // extract communicator from operator in the coarsest level.
      coarse_level_matrix = &coarse_level_operator_;
      coarse_dh           = &coarse_dof_handler;
      Assert((coarse_dh->n_dofs() == coarse_level_operator_.m()),
             ExcInternalError());

      mg_coarse_level_polytopal_hierarchy =
        std::make_unique<Utils::MGCoarseDirect<VectorType,
                                               TrilinosWrappers::SparseMatrix,
                                               TrilinosWrappers::SolverDirect>>(
          coarse_trilinos_matrix); // coarse polytopal level trilinos matrix

      //  Create a Multigrid object and a PreconditionMG object to be fed to the
      //  CG solver constructed in this class. The coarse grid solver is
      //  performed through a direct solver, see the
      //  mg_coarse_level_polytopal_hierarchy member of this class.

      mg = std::make_unique<Multigrid<VectorType>>(
        mg_matrix,
        *mg_coarse_level_polytopal_hierarchy,
        transfer,
        pre_smooth,
        post_smooth);
      preconditioner_mg = std::make_unique<
        PreconditionMG<dim,
                       VectorType,
                       MGTransferAgglomeration<dim, VectorType>>>(*coarse_dh,
                                                                  *mg,
                                                                  transfer);
    }


    void
    operator()(const unsigned int,
               VectorType       &dst,
               const VectorType &src) const override
    {
      double start, stop;
      start = MPI_Wtime();
      const_cast<SolverType &>(solver).solve(*coarse_level_matrix,
                                             dst,
                                             src,
                                             *preconditioner_mg);
      stop = MPI_Wtime();

      if (Utilities::MPI::this_mpi_process(comm) == 0)
        std::cout << "Coarse polytopal solver elapsed time: " << stop - start
                  << "[s] with " << control->last_step() << "iterations. "
                  << std::endl;
    }


    ~MGCoarsePolytopalSolver() = default;

  private:
    /**
     * MPI communicator.
     */
    const MPI_Comm comm;

    /**
     * Multigrid object to construct the preconditioner.
     */
    std::unique_ptr<Multigrid<VectorType>> mg;

    /**
     * Precondtioner to be given to the conjugate-gradient solver.
     */
    std::unique_ptr<
      PreconditionMG<dim, VectorType, MGTransferAgglomeration<dim, VectorType>>>
      preconditioner_mg;

    /**
     * Coarsest level matrix within the polynomial hierarchy.
     */
    const MatrixType *coarse_level_matrix;

    /**
     * Coarsest DoFHandler in the polynomial hierarchy.
     */
    const DoFHandler<dim> *coarse_dh;

    /**
     * Iterative solver to be invoked by operator().
     */
    const SolverType solver;

    const SolverControl *control;


    /**
     * Coarse direct solver to be used in the coarsest agglomerate level.
     */
    std::unique_ptr<Utils::MGCoarseDirect<VectorType,
                                          TrilinosWrappers::SparseMatrix,
                                          TrilinosWrappers::SolverDirect>>
      mg_coarse_level_polytopal_hierarchy;
  };



} // namespace Utils



#endif