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
  MGTransferAgglomeration<dim, VectorType>::MGTransferAgglomeration(
    const DoFHandler<dim> &fine_dof_handler_,
    const std::vector<std::unique_ptr<AgglomerationHandler<dim>>>
      &coarse_handlers_)
  {
    Assert(coarse_handlers_.size() > 0, ExcInternalError());
    fine_dof_handler = &fine_dof_handler_;
    dof_handlers.resize(coarse_handlers_.size() + 1);
    dof_handlers.back() = fine_dof_handler;

    agglomeration_handlers.resize(coarse_handlers_.size());
    for (unsigned int l = 0; l < coarse_handlers_.size(); ++l)
      {
        Assert(coarse_handlers_[l]->n_agglomerates() > 0,
               ExcMessage("Invalid number of agglomerates."));
        agglomeration_handlers[l] = coarse_handlers_[l].get();
        dof_handlers[l]           = &agglomeration_handlers[l]->agglo_dh;
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
    // Assert(transfer_matrices[to_level - 1] != nullptr,
    //        ExcMessage("Transfer matrix has not been initialized."));
    // std::cout << "to_level - 1= " << to_level - 1 << std::endl;
    prolongate_mf(to_level, dst, src);

    // transfer_matrices[to_level]->vmult_add(dst, src);
  }



  template <int dim, typename VectorType>
  void
  MGTransferAgglomeration<dim, VectorType>::prolongate_mf(
    const unsigned int to_level,
    VectorType        &dst,
    const VectorType  &src) const
  {
    // Assert(transfer_matrices[to_level - 1] != nullptr,
    //        ExcMessage("Transfer matrix has not been initialized."));
    // transfer_matrices[to_level - 1]->vmult_add(dst, src);
    // otherwise, do not create any matrix
    std::cout << "to_level inside = " << to_level << std::endl;


    if (to_level < agglomeration_handlers.size())
      {
        std::cout << "Coarse DoFs:"
                  << agglomeration_handlers[to_level - 1]->n_dofs()
                  << std::endl;
        std::cout << "Fine DoFs:" << agglomeration_handlers[to_level]->n_dofs()
                  << std::endl;

        const Triangulation<dim, dim> &tria =
          agglomeration_handlers[to_level]->get_triangulation();
        const Mapping<dim> &mapping =
          agglomeration_handlers[to_level]->get_mapping();
        const FiniteElement<dim, dim> &original_fe =
          agglomeration_handlers[to_level]->get_fe();

        const std::vector<BoundingBox<dim>> &coarse_bboxes =
          agglomeration_handlers[to_level - 1]->get_local_bboxes();
        const std::vector<BoundingBox<dim>> &fine_bboxes =
          agglomeration_handlers[to_level]->get_local_bboxes();

        // We use DGQ (on tensor-product meshes) or DGP (on simplex meshes)
        // nodal elements of the same degree as the ones in the agglomeration
        // handler to interpolate the solution onto the finer grid.
        std::unique_ptr<FiniteElement<dim>> output_fe;
        if (tria.all_reference_cells_are_hyper_cube())
          output_fe = std::make_unique<FE_DGQ<dim>>(original_fe.degree);
        // else if (tria.all_reference_cells_are_simplex())
        //   output_fe =
        //   std::make_unique<FE_SimplexDGP<dim>>(original_fe.degree);
        else
          AssertThrow(false, ExcNotImplemented());


        if constexpr (std::is_same_v<
                        VectorType,
                        LinearAlgebra::distributed::Vector<double>>)
          {
            // const IndexSet &locally_owned_dofs =
            //   agglomeration_handlers[to_level]->agglo_dh.locally_owned_dofs();
            // dst.reinit(locally_owned_dofs, MPI_COMM_WORLD);
          }
        else if constexpr (std::is_same_v<
                             VectorType,
                             Vector<typename VectorType::value_type>>)
          {
            dst.reinit(agglomeration_handlers[to_level]->n_dofs());
          }
        else
          {
            // PETSc, LA::d::v options not implemented.
            (void)agglomeration_handlers[to_level];
            (void)dst;
            (void)src;
            AssertThrow(false, ExcNotImplemented());
          }



        const unsigned int dofs_per_cell =
          agglomeration_handlers[to_level]->n_dofs_per_cell();
        const unsigned int fine_dofs_per_cell = dofs_per_cell;

        std::vector<types::global_dof_index> local_dof_indices_coarse(
          dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices_child(
          fine_dofs_per_cell);

        const auto &bboxes =
          agglomeration_handlers[to_level - 1]->get_local_bboxes();

        const std::vector<Point<dim>> &unit_support_points =
          original_fe.get_unit_support_points();

        // loop over coarse polytopes
        for (const auto &polytope :
             agglomeration_handlers[to_level - 1]->polytope_iterators())
          if (polytope->is_locally_owned())
            {
              polytope->get_dof_indices(local_dof_indices_coarse);
              const BoundingBox<dim> &coarse_bbox =
                coarse_bboxes[polytope->index()];

              // Get local children of the present polytope
              const auto &children_polytopes = polytope->children();
              for (const types::global_cell_index child_idx :
                   children_polytopes)
                {
                  const BoundingBox<dim> &fine_bbox = fine_bboxes[child_idx];
                  const typename DoFHandler<dim>::active_cell_iterator
                    &child_dh =
                      agglomeration_handlers[to_level]->polytope_to_dh_iterator(
                        child_idx);
                  child_dh->get_dof_indices(local_dof_indices_child);

                  // compute real location of support points
                  std::vector<Point<dim>> real_qpoints;
                  real_qpoints.reserve(unit_support_points.size());
                  for (const Point<dim> &p : unit_support_points)
                    real_qpoints.push_back(fine_bbox.unit_to_real(p));

                  for (unsigned int j = 0; j < local_dof_indices_child.size();
                       ++j)
                    {
                      const auto &ref_qpoint =
                        coarse_bbox.real_to_unit(real_qpoints[j]);
                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        dst(local_dof_indices_child[j]) +=
                          src(local_dof_indices_coarse[i]) *
                          original_fe.shape_value(i, ref_qpoint);
                    }
                }
            }
      }
    else
      {
        std::cout << "We are at dofhandler level" << std::endl;
        std::cout
          << "AH coarse has: "
          << agglomeration_handlers.back()->agglo_dh.locally_owned_dofs().size()
          << std::endl;

        // otherwise, do not create any matrix
        const Triangulation<dim, dim> &tria =
          agglomeration_handlers.back()->get_triangulation();
        const Mapping<dim> &mapping =
          agglomeration_handlers.back()->get_mapping();
        const FiniteElement<dim, dim> &original_fe =
          agglomeration_handlers.back()->get_fe();

        // We use DGQ (on tensor-product meshes) or DGP (on simplex meshes)
        // nodal elements of the same degree as the ones in the agglomeration
        // handler to interpolate the solution onto the finer grid.
        std::unique_ptr<FiniteElement<dim>> output_fe;
        if (tria.all_reference_cells_are_hyper_cube())
          output_fe = std::make_unique<FE_DGQ<dim>>(original_fe.degree);

        DoFHandler<dim> &output_dh = const_cast<DoFHandler<dim> &>(
          agglomeration_handlers.back()->output_dh);
        output_dh.reinit(tria);
        output_dh.distribute_dofs(*output_fe);


        if constexpr (std::is_same_v<
                        VectorType,
                        LinearAlgebra::distributed::Vector<double>>)
          {
            // const IndexSet &locally_owned_dofs =
            // output_dh.locally_owned_dofs(); dst.reinit(locally_owned_dofs,
            // MPI_COMM_WORLD); std::cout << locally_owned_dofs.size() <<
            // std::endl;
          }
        else if constexpr (std::is_same_v<
                             VectorType,
                             Vector<typename VectorType::value_type>>)
          {
            // dst.reinit(agglomeration_handlers.back()->n_dofs());
          }
        else
          {
            // PETSc, LA::d::v options not implemented.
            (void)agglomeration_handlers;
            (void)dst;
            (void)src;
            AssertThrow(false, ExcNotImplemented());
          }



        const unsigned int dofs_per_cell =
          agglomeration_handlers.back()->n_dofs_per_cell();
        const unsigned int output_dofs_per_cell = output_fe->n_dofs_per_cell();
        Quadrature<dim>    quad(output_fe->get_unit_support_points());
        FEValues<dim>      output_fe_values(mapping,
                                       *output_fe,
                                       quad,
                                       update_quadrature_points);

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices_output(
          output_dofs_per_cell);

        const auto &bboxes = agglomeration_handlers.back()->get_local_bboxes();
        for (const auto &polytope :
             agglomeration_handlers.back()->polytope_iterators())
          {
            if (polytope->is_locally_owned())
              {
                polytope->get_dof_indices(local_dof_indices);
                const BoundingBox<dim> &box = bboxes[polytope->index()];

                const auto &deal_cells =
                  polytope->get_agglomerate(); // fine deal.II cells
                for (const auto &cell : deal_cells)
                  {
                    const auto slave_output = cell->as_dof_handler_iterator(
                      agglomeration_handlers.back()->output_dh);
                    slave_output->get_dof_indices(local_dof_indices_output);
                    output_fe_values.reinit(slave_output);

                    const auto &qpoints =
                      output_fe_values.get_quadrature_points();

                    for (unsigned int j = 0; j < output_dofs_per_cell; ++j)
                      {
                        const auto &ref_qpoint = box.real_to_unit(qpoints[j]);
                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                          dst(local_dof_indices_output[j]) +=
                            src(local_dof_indices[i]) *
                            original_fe.shape_value(i, ref_qpoint);
                      }
                  }
              }
          }
      }
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
    // Assert(transfer_matrices[from_level - 1] != nullptr,
    //        ExcMessage("Matrix has not been initialized."));
    // transfer_matrices[from_level - 1]->Tvmult_add(dst, src);
    restrict_mf(from_level, dst, src);
  }



  template <int dim, typename VectorType>
  void
  MGTransferAgglomeration<dim, VectorType>::restrict_mf(
    const unsigned int from_level,
    VectorType        &dst,
    const VectorType  &src) const
  {
    // Assert(transfer_matrices[to_level - 1] != nullptr,
    //        ExcMessage("Transfer matrix has not been initialized."));
    // transfer_matrices[to_level - 1]->vmult_add(dst, src);
    // otherwise, do not create any matrix
    std::cout << "from_level (restriction) = " << from_level << std::endl;


    if (from_level < agglomeration_handlers.size())
      {
        const AgglomerationHandler<dim> *coarse_handler =
          agglomeration_handlers[from_level - 1];
        const AgglomerationHandler<dim> *fine_handler =
          agglomeration_handlers[from_level];
        std::cout << "Fine DoFs:" << fine_handler->n_dofs() << std::endl;
        std::cout << "Coarse DoFs:" << coarse_handler->n_dofs() << std::endl;

        const Triangulation<dim, dim> &tria = fine_handler->get_triangulation();
        const FiniteElement<dim, dim> &original_fe = fine_handler->get_fe();

        const std::vector<BoundingBox<dim>> &coarse_bboxes =
          coarse_handler->get_local_bboxes();
        const std::vector<BoundingBox<dim>> &fine_bboxes =
          fine_handler->get_local_bboxes();

        // We use DGQ (on tensor-product meshes)
        std::unique_ptr<FiniteElement<dim>> output_fe;
        if (tria.all_reference_cells_are_hyper_cube())
          output_fe = std::make_unique<FE_DGQ<dim>>(original_fe.degree);
        else
          AssertThrow(false, ExcNotImplemented());


        const unsigned int dofs_per_cell      = fine_handler->n_dofs_per_cell();
        const unsigned int fine_dofs_per_cell = dofs_per_cell;

        std::vector<types::global_dof_index> local_dof_indices_fine(
          dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices_parent(
          fine_dofs_per_cell);


        const std::vector<Point<dim>> &unit_support_points =
          original_fe.get_unit_support_points();

        // loop over fine polytopes
        for (const auto &polytope : fine_handler->polytope_iterators())
          if (polytope->is_locally_owned())
            {
              const types::global_cell_index fine_idx = polytope->index();
              std::cout << "fine idx = " << fine_idx << std::endl;

              polytope->get_dof_indices(local_dof_indices_fine);
              const BoundingBox<dim> &fine_bbox = fine_bboxes[fine_idx];

              // Get local parent of the present polytope
              const auto &parent_idx = polytope->parent();
              std::cout << "parent_idx = " << parent_idx << std::endl;
              const BoundingBox<dim> &parent_bbox = coarse_bboxes[parent_idx];

              const typename DoFHandler<dim>::active_cell_iterator &parent_dh =
                coarse_handler->polytope_to_dh_iterator(parent_idx);

              parent_dh->get_dof_indices(local_dof_indices_parent);

              // compute real location of support points
              std::vector<Point<dim>> real_qpoints;
              real_qpoints.reserve(unit_support_points.size());
              for (const Point<dim> &p : unit_support_points)
                real_qpoints.push_back(fine_bbox.unit_to_real(p));


              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    const auto &ref_qpoint =
                      parent_bbox.real_to_unit(real_qpoints[j]);

                    dst(local_dof_indices_parent[i]) +=
                      src(local_dof_indices_fine[j]) *
                      original_fe.shape_value(i, ref_qpoint);
                  }
            }
      }
    else
      {
        std::cout << "We are at dofhandler level (restriction)" << std::endl;
        std::cout << "With DoFs (fine):" << fine_dof_handler->n_dofs()
                  << std::endl;

        const AgglomerationHandler<dim> *handler =
          agglomeration_handlers[from_level - 1];
        std::cout << "With DoFs (coarse):" << handler->n_dofs() << std::endl;


        const Triangulation<dim, dim> &tria    = handler->get_triangulation();
        const Mapping<dim, dim>       &mapping = handler->get_mapping();
        const FiniteElement<dim, dim> &original_fe = handler->get_fe();

        const std::vector<BoundingBox<dim>> &coarse_bboxes =
          handler->get_local_bboxes();

        // We use DGQ (on tensor-product meshes)
        std::unique_ptr<FiniteElement<dim>> output_fe;
        if (tria.all_reference_cells_are_hyper_cube())
          output_fe = std::make_unique<FE_DGQ<dim>>(original_fe.degree);
        else
          AssertThrow(false, ExcNotImplemented());


        const unsigned int dofs_per_cell      = handler->n_dofs_per_cell();
        const unsigned int fine_dofs_per_cell = dofs_per_cell;

        std::vector<types::global_dof_index> local_dof_indices_fine(
          dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices_parent(
          fine_dofs_per_cell);


        const std::vector<Point<dim>> &unit_support_points =
          original_fe.get_unit_support_points();

        // loop over deal.II cells
        for (const auto &cell : fine_dof_handler->active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(local_dof_indices_fine);

              // Get local parent (polytope) of the present cell
              const types::global_cell_index parent_idx =
                handler->master2polygon.at(
                  handler->get_master_idx_of_cell_local(cell));

              const BoundingBox<dim> &parent_bbox = coarse_bboxes[parent_idx];

              const typename DoFHandler<dim>::active_cell_iterator &parent_dh =
                handler->polytope_to_dh_iterator(parent_idx);

              parent_dh->get_dof_indices(local_dof_indices_parent);

              // compute real location of support points
              std::vector<Point<dim>> real_qpoints;
              real_qpoints.reserve(unit_support_points.size());
              for (const Point<dim> &p : unit_support_points)
                real_qpoints.push_back(
                  mapping.transform_unit_to_real_cell(cell, p));

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    const auto &ref_qpoint =
                      parent_bbox.real_to_unit(real_qpoints[j]);

                    dst(local_dof_indices_parent[i]) +=
                      src(local_dof_indices_fine[j]) *
                      original_fe.shape_value(i, ref_qpoint);
                  }
            }
      }
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
