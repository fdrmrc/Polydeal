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

#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <multigrid_amg.h>

namespace dealii
{



  template <int dim, typename VectorType>
  MGTransferAgglomeration<dim, VectorType>::MGTransferAgglomeration(
    const MGLevelObject<TrilinosWrappers::SparseMatrix *> &transfer_matrices_,
    const std::vector<DoFHandler<dim> *>                  &dof_handlers_)
  {
    Assert(transfer_matrices_.n_levels() > 0, ExcInternalError());
    transfer_matrices.resize(0, dof_handlers_.size());
    dof_handlers.resize(dof_handlers_.size());

    for (unsigned int l = transfer_matrices_.min_level();
         l <= transfer_matrices_.max_level();
         ++l)
      {
        // std::cout << "l in build transfers: " << l << std::endl;
        transfer_matrices[l] = transfer_matrices_[l];
        dof_handlers[l]      = dof_handlers_[l];
      }
  }



  template <int dim, typename VectorType>
  MGTransferAgglomeration<dim, VectorType>::MGTransferAgglomeration(
    const MGLevelObject<TrilinosWrappers::SparseMatrix> &transfer_matrices_,
    const std::vector<DoFHandler<dim> *>                &dof_handlers_)
  {
    Assert(transfer_matrices_.n_levels() > 0, ExcInternalError());
    transfer_matrices.resize(0, dof_handlers_.size());
    dof_handlers.resize(dof_handlers_.size());

    for (unsigned int l = transfer_matrices_.min_level();
         l <= transfer_matrices_.max_level();
         ++l)
      {
        // std::cout << "l in build transfers: " << l << std::endl;
        transfer_matrices[l] =
          const_cast<dealii::TrilinosWrappers::SparseMatrix *>(
            &transfer_matrices_[l]);
        dof_handlers[l] = dof_handlers_[l];
      }
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
    Assert(transfer_matrices[to_level - 1] != nullptr,
           ExcMessage("Transfer matrix has not been initialized."));
    transfer_matrices[to_level - 1]->vmult_add(dst, src);
  }



  template <int dim, typename VectorType>
  void
  MGTransferAgglomeration<dim, VectorType>::restrict_and_add(
    const unsigned int from_level,
    VectorType        &dst,
    const VectorType  &src) const
  {
    // std::cout << "from_level " << from_level << std::endl;
    // std::cout << "Rows: " << transfer_matrices[from_level - 1]->m()
    //           << std::endl;
    // std::cout << "Cols: " << transfer_matrices[from_level - 1]->n()
    //           << std::endl;
    Assert(transfer_matrices[from_level - 1] != nullptr,
           ExcMessage("Matrix has not been initialized."));
    transfer_matrices[from_level - 1]->Tvmult_add(dst, src);
  }



  template <int dim, typename VectorType>
  void
  MGTransferAgglomeration<dim, VectorType>::copy_to_mg(
    const DoFHandler<dim>     &dof_handler,
    MGLevelObject<VectorType> &dst,
    const VectorType          &src) const
  {
    (void)dof_handler; // required by interface, but not needed.
    // std::cout << "Before copy_to_mg() " << std::endl;
    for (unsigned int level = dst.min_level(); level <= dst.max_level();
         ++level)
      {
        // std::cout << "level = " << level << std::endl;
        dst[level].reinit(dof_handlers[level]->locally_owned_dofs(),
                          dof_handlers[level]->get_communicator());
      }

    if constexpr (std::is_same_v<VectorType, TrilinosWrappers::MPI::Vector>)
      dst[dst.max_level()] = src;
    else if constexpr (std::is_same_v<
                         VectorType,
                         LinearAlgebra::distributed::Vector<double>>)
      dst[dst.max_level()].copy_locally_owned_data_from(src);
    else
      DEAL_II_NOT_IMPLEMENTED();
  }



  template <int dim, typename VectorType>
  void
  MGTransferAgglomeration<dim, VectorType>::copy_from_mg(
    const DoFHandler<dim>           &dof_handler,
    VectorType                      &dst,
    const MGLevelObject<VectorType> &src) const
  {
    (void)dof_handler;
    if constexpr (std::is_same_v<VectorType, TrilinosWrappers::MPI::Vector>)
      dst = src[src.max_level()];
    else if constexpr (std::is_same_v<
                         VectorType,
                         LinearAlgebra::distributed::Vector<double>>)
      dst.copy_locally_owned_data_from(src[src.max_level()]);
    else
      DEAL_II_NOT_IMPLEMENTED();
  }



  // explicit instantiations for doubles and floats
  // template class MatrixFreeProjector<1, double>;
  // template class MatrixFreeProjector<2, double>;
  // template class MatrixFreeProjector<3, double>;

  template class MGTransferAgglomeration<
    1,
    LinearAlgebra::distributed::Vector<double>>;
  template class MGTransferAgglomeration<
    2,
    LinearAlgebra::distributed::Vector<double>>;
  template class MGTransferAgglomeration<
    3,
    LinearAlgebra::distributed::Vector<double>>;

  // Trilinos vectors instantiations
  template class MGTransferAgglomeration<1, TrilinosWrappers::MPI::Vector>;
  template class MGTransferAgglomeration<2, TrilinosWrappers::MPI::Vector>;
  template class MGTransferAgglomeration<3, TrilinosWrappers::MPI::Vector>;
} // namespace dealii
