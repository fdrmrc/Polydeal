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


#include <agglomeration_handler.h>
#include <multigrid_amg.h>

namespace dealii
{
  template <int dim, typename Number>
  MatrixFreeProjector<dim, Number>::MatrixFreeProjector(
    const MatrixFreeOperators::Base<dim,
                                    LinearAlgebra::distributed::Vector<Number>>
                                                       &mf_operator_,
    const std::vector<TrilinosWrappers::SparseMatrix *> transfers_)
  {
    // Only DGQ discretizations are supported.
    Assert(dynamic_cast<const FE_DGQ<dim> *>(
             &mf_operator_.get_matrix_free()->get_dof_handler().get_fe()) !=
             nullptr,
           ExcNotImplemented());
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

    // Check parallel layout is identical on every level
    for (unsigned int l = 0; l < transfers_.size(); ++l)
      Assert((mf_operator_.get_matrix_free()->get_locally_owned_set() ==
              transfers_[l]->locally_owned_range_indices()),
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

        // Inform each linear operator about the parallel layout. Use the given
        // trilinos matrices.
        level_operators[l].reinit_domain_vector = [this, l](VectorType &v,
                                                            bool) {
          v.reinit(transfer_matrices[l]->locally_owned_domain_indices(),
                   communicator);
        };

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
    mf_linear_operator.vmult = [this](VectorType &dst, const VectorType &src) {
      mf_operator->vmult(dst, src);
    };
    mf_linear_operator.vmult_add = [this](VectorType       &dst,
                                          const VectorType &src) {
      mf_operator->vmult_add(dst, src);
    };
    mf_linear_operator.Tvmult = [this](VectorType &dst, const VectorType &src) {
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
      v.reinit(
        mf_operator->get_matrix_free()->get_dof_handler().locally_owned_dofs(),
        communicator);
    };
  }



  template <int dim, typename Number>
  void
  MatrixFreeProjector<dim, Number>::compute_level_matrices(
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
    for (unsigned int l = max_level - 1; l == min_level; --l)
      {
        mg_matrices[l] = transpose_operator(level_operators[l]) *
                         mf_linear_operator * level_operators[l];
        mg_matrices[l].n_rows = level_operators[l].n();
        mg_matrices[l].n_cols = mg_matrices[l].n_rows;
      }
  }



  template <int dim, typename VectorType>
  MGTransferAgglomeration<dim, VectorType>::MGTransferAgglomeration(
    const MGLevelObject<TrilinosWrappers::SparseMatrix *> &transfer_matrices_,
    const std::vector<DoFHandler<dim> *>                  &dof_handlers_)
  {
    Assert(transfer_matrices_.n_levels() > 0, ExcInternalError());
    transfer_matrices.resize(0, dof_handlers_.size());
    dof_handlers.resize(dof_handlers_.size());

    for (unsigned int l = transfer_matrices_.min_level();
         l < transfer_matrices_.max_level();
         ++l)
      {
        transfer_matrices[l] = transfer_matrices_[l];
        dof_handlers[l]      = dof_handlers_[l];
      }

    transfer_matrices[dof_handlers_.size() - 1] =
      transfer_matrices_[dof_handlers_.size() - 1];
    dof_handlers[dof_handlers_.size() - 1] =
      dof_handlers_[dof_handlers_.size() - 1];
  }



  template <int dim, typename VectorType>
  void
  MGTransferAgglomeration<dim, VectorType>::prolongate(
    const unsigned int to_level,
    VectorType        &dst,
    const VectorType  &src) const
  {
    dst = typename VectorType::value_type(0.0);
    prolongate_and_add(to_level, dst, src);
  }



  template <int dim, typename VectorType>
  void
  MGTransferAgglomeration<dim, VectorType>::prolongate_and_add(
    const unsigned int to_level,
    VectorType        &dst,
    const VectorType  &src) const
  {
    Assert(transfer_matrices[to_level] != nullptr,
           ExcMessage("Transfer matrix has not been initialized."));
    transfer_matrices[to_level]->vmult_add(dst, src);
  }



  template <int dim, typename VectorType>
  void
  MGTransferAgglomeration<dim, VectorType>::restrict_and_add(
    const unsigned int from_level,
    VectorType        &dst,
    const VectorType  &src) const
  {
    Assert(transfer_matrices[from_level] != nullptr,
           ExcMessage("Matrix has not been initialized."));
    transfer_matrices[from_level]->Tvmult_add(dst, src);
  }



  template <int dim, typename VectorType>
  void
  MGTransferAgglomeration<dim, VectorType>::copy_to_mg(
    const DoFHandler<dim>     &dof_handler,
    MGLevelObject<VectorType> &dst,
    const VectorType          &src) const
  {
    (void)dof_handler; // required by interface, but not needed.
    for (unsigned int level = dst.min_level(); level <= dst.max_level();
         ++level)
      {
        dst[level].reinit(dof_handlers[level]->locally_owned_dofs(),
                          dof_handlers[level]->get_communicator());
      }
    dst[dst.max_level()].copy_locally_owned_data_from(src);
  }



  template <int dim, typename VectorType>
  void
  MGTransferAgglomeration<dim, VectorType>::copy_from_mg(
    const DoFHandler<dim>           &dof_handler,
    VectorType                      &dst,
    const MGLevelObject<VectorType> &src) const
  {
    (void)dof_handler;
    dst.copy_locally_owned_data_from(src[src.max_level()]);
  }



  // explicit instantiations for doubles and floats
  template class MatrixFreeProjector<1, double>;
  template class MatrixFreeProjector<2, double>;
  template class MatrixFreeProjector<3, double>;

  template class MGTransferAgglomeration<
    1,
    LinearAlgebra::distributed::Vector<double>>;
  template class MGTransferAgglomeration<
    2,
    LinearAlgebra::distributed::Vector<double>>;
  template class MGTransferAgglomeration<
    3,
    LinearAlgebra::distributed::Vector<double>>;
} // namespace dealii
