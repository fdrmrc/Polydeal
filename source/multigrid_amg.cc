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

    mf_linear_operator.reinit_domain_vector = [&](VectorType &v, bool) {};
    mf_linear_operator.reinit_range_vector  = [&](VectorType &v, bool) {
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
    using VectorType            = LinearAlgebra::distributed::Vector<Number>;
    const unsigned int n_levels = mg_matrices.n_levels();
    Assert(n_levels > 1, ExcMessage("Vector of matrices set to invalid size."));
    Assert(!mf_linear_operator.is_null_operator, ExcInternalError());
    const unsigned int min_level = mg_matrices.min_level();
    const unsigned int max_level = mg_matrices.max_level();

    mg_matrices[min_level] = mf_linear_operator; // finest level

    // do the same, but using transfers to define level matrices
    for (unsigned int l = min_level + 1; l < max_level; l++)
      mg_matrices[l] = transpose_operator(level_operators[l - 1]) *
                       mf_linear_operator * level_operators[l - 1];
  }

  // explicit instantiations
  template class MatrixFreeProjector<1, double>;
  template class MatrixFreeProjector<2, double>;
  template class MatrixFreeProjector<3, double>;


} // namespace dealii
