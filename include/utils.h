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
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
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
    // First, check that we support the matrix types
    static constexpr bool is_trilinos_matrix =
      std::is_same_v<TrilinosWrappers::SparseMatrix, MatrixType>;
    static constexpr bool is_serial_matrix =
      std::is_same_v<SparseMatrix<double>, MatrixType>;
    static constexpr bool is_supported_matrix =
      is_trilinos_matrix || is_serial_matrix;
    static_assert(is_supported_matrix);
    Assert(matrix.empty() && sp.empty(),
           ExcMessage(
             "The destination matrix and its sparsity pattern must the empty "
             "upon calling this function."));
    Assert(coarse_ah.n_dofs() < fine_ah.n_dofs(), ExcInternalError());
    AssertDimension(dim, spacedim);

    using NumberType = typename MatrixType::value_type;

    // Get information from the handlers
    const DoFHandler<dim, spacedim> &coarse_agglo_dh = coarse_ah.agglo_dh;
    const DoFHandler<dim, spacedim> &fine_agglo_dh   = fine_ah.agglo_dh;

    const FiniteElement<dim, spacedim> &fe   = coarse_ah.get_fe();
    const Triangulation<dim, spacedim> &tria = coarse_ah.get_triangulation();
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
            const typename DoFHandler<dim>::active_cell_iterator
              &coarse_polytope_dh =
                coarse_ah.polytope_to_dh_iterator(polytope->index());

            polytope->get_dof_indices(local_dof_indices_coarse);

            // Get local children of the present polytope
            const auto &children_polytopes = polytope->children();
            for (const types::global_cell_index child_idx : children_polytopes)
              {
                const typename DoFHandler<dim>::active_cell_iterator &child_dh =
                  fine_ah.polytope_to_dh_iterator(child_idx);
                child_dh->get_dof_indices(local_dof_indices_child);

                local_matrix = 0.;

                // compute real location of support points
                std::vector<Point<dim>> real_qpoints;
                real_qpoints.reserve(unit_support_points.size());
                for (const Point<dim> &p : unit_support_points)
                  real_qpoints.push_back(
                    fine_ah.euler_mapping->transform_unit_to_real_cell(child_dh,
                                                                       p));
                // real_qpoints.push_back(box_fine.unit_to_real(p));

                for (unsigned int i = 0; i < local_dof_indices_coarse.size();
                     ++i)
                  {
                    const auto &p =
                      coarse_ah.euler_mapping->transform_real_to_unit_cell(
                        coarse_polytope_dh, real_qpoints[i]);
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
        // PETSc, LA::d::v options not implemented.
        (void)coarse_ah;
        (void)fine_ah;
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

} // namespace Utils



#endif