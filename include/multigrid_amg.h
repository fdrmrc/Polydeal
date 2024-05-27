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


#ifndef multigrid_amg_h
#define multigrid_amg_h


#include <deal.II/base/config.h>

#include <deal.II/base/mg_level_object.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/linear_operator.h>

#include <deal.II/matrix_free/operators.h>

#include <deal.II/multigrid/mg_base.h>

namespace dealii
{
  /**
   * This class is responsible for the setup of level matrices for a given
   * (matrix-free) operator evaluation. Such level matrices are the "level
   * matrices" to be used in a multigrid method. The difference compared to
   * standard multilevel methods is that we construct such matrices using a
   * Galerkin projection.
   */
  template <int dim, typename Number = double>
  class MatrixFreeProjector
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;
    /**
     * Constructor. It takes the matrix-free operator evaluation on the finest
     * level, and a series of transfers from levels.
     */
    MatrixFreeProjector(
      const MatrixFreeOperators::
        Base<dim, LinearAlgebra::distributed::Vector<Number>> &mf_operator,
      const std::vector<TrilinosWrappers::SparseMatrix *>      transfers);

    /**
     * Initialize level matrices using the operator evaluation and the transfer
     * matrices provided in the constructor.
     *
     * In particular, matrix[0]= A0, while for the other levels it holds that:
     * matrix[l] = P_l^T A0 P_l, being P_l the injection from the fine level
     * (indexed by 0) and level l.
     */
    void
    compute_level_matrices(
      MGLevelObject<LinearOperator<VectorType, VectorType>> &mg_matrices);

  private:
    /**
     * MPI communicator used by Trilinos objects.
     */
    MPI_Comm communicator;

    /**
     * Matrix-free operator evaluation.
     */
    const MatrixFreeOperators::Base<dim, VectorType> *mf_operator;

    /**
     * Vector of (pointers of) Trilinos Matrices storing two-level projections.
     */
    std::vector<TrilinosWrappers::SparseMatrix *> transfer_matrices;

    /**
     * LinearOperator for each level, storing Galerkin projections.
     */
    std::vector<LinearOperator<VectorType, VectorType>> level_operators;

    /**
     * For matrix-free operator evaluation
     */
    LinearOperator<VectorType, VectorType> mf_linear_operator;
  };



  /**
   * This class implements the transfer across possibly agglomerated
   * multigrid levels. To perform this, it needs a sequence of transfer matrices
   * and DoFHandlers which identify the degrees of freedom on each level.
   *
   */
  template <int dim, typename VectorType>
  class MGTransferAgglomeration : public MGTransferBase<VectorType>
  {
  public:
    /**
     * Constructor. It takes a sequence of transfer matrices from possibly
     * agglomerated meshes.
     */
    MGTransferAgglomeration(
      const MGLevelObject<TrilinosWrappers::SparseMatrix *> &transfer_matrices,
      const std::vector<DoFHandler<dim> *>                  &dof_handlers);


    /**
     * Perform prolongation from a coarse level vector @p src to a fine one
     * @p dst. The previous content of dst is overwritten.
     */
    void
    prolongate(const unsigned int to_level,
               VectorType        &dst,
               const VectorType  &src) const override;


    /**
     * Perform prolongation, summing into the previous content of dst.
     */
    void
    prolongate_and_add(const unsigned int to_level,
                       VectorType        &dst,
                       const VectorType  &src) const override;


    /**
     * Perform restriction.
     */
    void
    restrict_and_add(const unsigned int from_level,
                     VectorType        &dst,
                     const VectorType  &src) const override;


    /**
     * Transfer from a vector on the global grid to vectors defined on each of
     * the levels separately for the active degrees of freedom. In particular,
     * for a globally refined mesh only the finest level in dst is filled as a
     * plain copy of src. All the other level objects are left untouched.
     */
    void
    copy_to_mg(const DoFHandler<dim>     &dof_handler,
               MGLevelObject<VectorType> &dst,
               const VectorType          &src) const;


    /**
     * Transfer from multi-level vector to normal vector.
     */
    void
    copy_from_mg(const DoFHandler<dim>           &dof_handler,
                 VectorType                      &dst,
                 const MGLevelObject<VectorType> &src) const;


  private:
    /**
     * Sequence of transfer operators, stored as pointers to Trilinos matrices.
     */
    MGLevelObject<SmartPointer<TrilinosWrappers::SparseMatrix>>
      transfer_matrices;

    /**
     * Pointers to DoFHandler employe on the levels.
     */
    std::vector<const DoFHandler<dim> *> dof_handlers;
  };

} // namespace dealii


#endif