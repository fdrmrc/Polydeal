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
  template <int dim, typename OperatorType, typename Number = double>
  class MatrixFreeProjector
  {
  private:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;
    /**
     * MPI communicator used by Trilinos objects.
     */
    MPI_Comm communicator;

    /**
     * Matrix-free operator evaluation.
     */
    const OperatorType *mf_operator;

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

  public:
    /**
     * Constructor. It takes the matrix-free operator evaluation on the finest
     * level, and a series of transfers from levels.
     */
    MatrixFreeProjector(
      const OperatorType                                 &mf_operator_,
      const std::vector<TrilinosWrappers::SparseMatrix *> transfers_)
    {
      // Only DGQ discretizations are supported.
      Assert(dynamic_cast<const FE_DGQ<dim> *>(
               &mf_operator_.get_matrix_free()->get_dof_handler().get_fe()) !=
               nullptr,
             ExcNotImplemented());
      using VectorType = LinearAlgebra::distributed::Vector<Number>;

      // Check parallel layout is identical on every level
      // for (unsigned int l = 0; l < transfers_.size(); ++l)
      Assert((mf_operator_.get_matrix_free()->get_locally_owned_set() ==
              transfers_[transfers_.size() - 1]->locally_owned_range_indices()),
             ExcInternalError());

      transfer_matrices.resize(transfers_.size());
      level_operators.resize(transfers_.size());
      // get communicator from first Trilinos matrix
      communicator = transfers_[0]->get_mpi_communicator();

      for (unsigned int l = 0; l < transfers_.size(); ++l)
        {
          // Set the pointer to the correct matrix
          transfer_matrices[l] = transfers_[l];

          level_operators[l].n_rows = transfers_[l]->m();
          level_operators[l].n_cols = transfers_[l]->n();

          // Define vmult-type lambdas for each linear operator.
          level_operators[l].vmult = [this, l](VectorType       &dst,
                                               const VectorType &src) {
            transfer_matrices[l]->vmult(dst, src);
          };
          level_operators[l].vmult_add = [this, l](VectorType       &dst,
                                                   const VectorType &src) {
            transfer_matrices[l]->vmult_add(dst, src);
          };
          level_operators[l].Tvmult = [this, l](VectorType       &dst,
                                                const VectorType &src) {
            transfer_matrices[l]->Tvmult(dst, src);
          };
          level_operators[l].Tvmult_add = [this, l](VectorType       &dst,
                                                    const VectorType &src) {
            transfer_matrices[l]->Tvmult_add(dst, src);
          };

          // Inform each linear operator about the parallel layout. Use the
          // given trilinos matrices.
          level_operators[l].reinit_domain_vector =
            [this, l](VectorType &v, bool) { (void)v; };

          level_operators[l].reinit_range_vector = [this, l](VectorType &v,
                                                             bool) {
            v.reinit(transfer_matrices[l]->locally_owned_range_indices(),
                     communicator);
          };
        }

      // Do the same for the matrix-free object.
      // First, set the pointer
      mf_operator = &mf_operator_;

      mf_linear_operator.n_rows = mf_operator_.m();
      mf_linear_operator.n_cols = mf_operator_.n();

      // Then, populate the corresponding lambdas (std::functions)
      mf_linear_operator.vmult = [this](VectorType       &dst,
                                        const VectorType &src) {
        // std::cout << "vmult LO called" << std::endl;
        mf_operator->vmult(dst, src);
      };
      mf_linear_operator.vmult_add = [this](VectorType       &dst,
                                            const VectorType &src) {
        // std::cout << "vmult_add LO called" << std::endl;
        mf_operator->vmult_add(dst, src);
      };
      mf_linear_operator.Tvmult = [this](VectorType       &dst,
                                         const VectorType &src) {
        mf_operator->Tvmult(dst, src);
      };
      mf_linear_operator.Tvmult_add = [this](VectorType       &dst,
                                             const VectorType &src) {
        mf_operator->Tvmult_add(dst, src);
      };

      mf_linear_operator.reinit_domain_vector = [&](VectorType &v, bool) {
        (void)v;
      };
      mf_linear_operator.reinit_range_vector = [&](VectorType &v, bool) {
        v.reinit(mf_operator->get_matrix_free()
                   ->get_dof_handler()
                   .locally_owned_dofs(),
                 communicator);
      };
    }

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
      MGLevelObject<LinearOperator<VectorType, VectorType>> &mg_matrices)
    {
      Assert(mg_matrices.n_levels() > 1,
             ExcMessage("Vector of matrices set to invalid size."));
      using VectorType = LinearAlgebra::distributed::Vector<Number>;
      Assert(!mf_linear_operator.is_null_operator, ExcInternalError());
      const unsigned int min_level = mg_matrices.min_level();
      const unsigned int max_level = mg_matrices.max_level();

      mg_matrices[max_level] = mf_linear_operator; // finest level

      // do the same, but using transfers to define level matrices
      // std::cout << "min level = " << min_level << std::endl;
      // std::cout << "max level = " << max_level << std::endl;
      for (unsigned int l = max_level; l-- > min_level;)
        {
          // std::cout << "l= " << l << std::endl;
          mg_matrices[l] = transpose_operator(level_operators[l]) *
                           mf_linear_operator * level_operators[l];
          mg_matrices[l].n_rows = level_operators[l].n();
          mg_matrices[l].n_cols = mg_matrices[l].n_rows;
        }
    }
  };



  /**
   * Same as above, but matrix-based.
   */
  template <int dim, typename MatrixType, typename Number = double>
  class AmgProjector
  {
  private:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;
    /**
     * MPI communicator used by Trilinos objects.
     */
    MPI_Comm communicator;

    /**
     * Fine operator evaluation.
     */
    // const MatrixType &fine_operator;

    /**
     * Vector of (pointers of) Trilinos Matrices storing two-level projections.
     */
    std::vector<const MatrixType *> transfer_matrices;



  public:
    /**
     * Constructor. It takes the matrix-free operator evaluation on the finest
     * level, and a series of transfers from levels.
     */
    AmgProjector(const std::vector<TrilinosWrappers::SparseMatrix> &transfers_)
    {
      // Only DGQ discretizations are supported.

      using VectorType = LinearAlgebra::distributed::Vector<Number>;

      // Check parallel layout is identical on every level
      // for (unsigned int l = 0; l < transfers_.size(); ++l)
      // Assert((fine_operator.locally_owned_range_indices() ==
      //         transfers_[transfers_.size() -
      //         1].locally_owned_range_indices()),
      //        ExcInternalError());

      // get communicator from first Trilinos matrix
      communicator = transfers_[0].get_mpi_communicator();
      transfer_matrices.resize(transfers_.size());

      // Store pointers to fine operator and transfers
      for (unsigned int l = 0; l < transfers_.size(); ++l)
        {
          // std::cout << "transferring matrix l= " << l << std::endl;
          transfer_matrices[l] = &transfers_[l];
        }
    }



    /**
     * Initialize level matrices using the operator evaluation and the transfer
     * matrices provided in the constructor.
     *
     * In particular, matrix[0]= A0, while for the other levels it holds that:
     * matrix[l] = P_l^T A_l-1 P_l, being P_l the two-level injection.
     */
    void
    compute_level_matrices(
      MGLevelObject<std::unique_ptr<MatrixType>> &mg_matrices)
    {
      Assert(mg_matrices.n_levels() > 1,
             ExcMessage("Vector of matrices set to invalid size."));
      using VectorType = LinearAlgebra::distributed::Vector<Number>;

      const unsigned int min_level = mg_matrices.min_level();
      const unsigned int max_level = mg_matrices.max_level();

      // mg_matrices[max_level].copy_from(fine_operator); // finest level
      // MPI_Barrier(communicator);
      // std::cout << "copied finest operator " << max_level << std::endl;

      // do the same, but using transfers to define level matrices
      // std::cout << "min level = " << min_level << std::endl;
      for (unsigned int l = max_level; l-- > min_level;)
        {
          // Set parallel layout of intermediate operators AP
          MatrixType level_operator(
            transfer_matrices[l]->locally_owned_range_indices(),
            transfer_matrices[l]->locally_owned_domain_indices(),
            communicator);

          mg_matrices[l] = std::make_unique<MatrixType>();

          // First, compute AP
          mg_matrices[l + 1]->mmult(
            level_operator,
            *transfer_matrices[l]); // result stored in level_operators[l]
                                    // Multiply by the transpose
          transfer_matrices[l]->Tmmult(*mg_matrices[l], level_operator);
        }
    }



    void
    compute_level_matrices(MGLevelObject<MatrixType> &mg_matrices)
    {
      Assert(mg_matrices.n_levels() > 1,
             ExcMessage("Vector of matrices set to invalid size."));
      using VectorType = LinearAlgebra::distributed::Vector<Number>;

      const unsigned int min_level = mg_matrices.min_level();
      const unsigned int max_level = mg_matrices.max_level();

      // mg_matrices[max_level].copy_from(fine_operator); // finest level
      // MPI_Barrier(communicator);
      // std::cout << "copied finest operator " << max_level << std::endl;

      // do the same, but using transfers to define level matrices
      // std::cout << "min level = " << min_level << std::endl;
      for (unsigned int l = max_level; l-- > min_level;)
        {
          // Set parallel layout of intermediate operators AP
          MatrixType level_operator(
            transfer_matrices[l]->locally_owned_range_indices(),
            transfer_matrices[l]->locally_owned_domain_indices(),
            communicator);

          // First, compute AP
          mg_matrices[l + 1].mmult(
            level_operator,
            *transfer_matrices[l]); // result stored in level_operators[l]
                                    // Multiply by the transpose
          transfer_matrices[l]->Tmmult(mg_matrices[l], level_operator);
        }
    }
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