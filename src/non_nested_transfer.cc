/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2022 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 */


#include <non_nested_transfer.h>
namespace bgi = boost::geometry::index;

template <int dim0, int dim1, int spacedim, typename VectorType>
void
non_nested_prolongation(const GridTools::Cache<dim0, spacedim> &coarse_cache,
                        const GridTools::Cache<dim1, spacedim> &fine_cache,
                        const DoFHandler<dim0, spacedim>       &coarse_dh,
                        const DoFHandler<dim1, spacedim>       &fine_dh,
                        const FiniteElement<dim0, spacedim>    &fe_space,
                        const VectorType                       &src,
                        VectorType                             &dst)
{
  Assert(fe_space.has_support_points(),
         ExcMessage("The FiniteElement need to have support points."))

    const auto      &unit_pts        = fe_space.get_unit_support_points();
  const unsigned int n_dofs_per_cell = fe_space.n_dofs_per_cell();
  const auto        &coarse_tree = coarse_cache.get_cell_bounding_boxes_rtree();
  const auto        &tree        = fine_cache.get_cell_bounding_boxes_rtree();
  std::vector<types::global_dof_index> coarse_dofs(n_dofs_per_cell);
  std::vector<types::global_dof_index> space_dofs(n_dofs_per_cell);
  const auto &coarse_mapping = coarse_cache.get_mapping();

  FullMatrix<double> prolongation_matrix(
    fine_dh.n_dofs(), coarse_dh.n_dofs()); // use Sparse, of course...

  Quadrature<dim1>         quadrature(unit_pts);
  FEValues<dim1, spacedim> fe_values(fine_cache.get_mapping(),
                                     fe_space,
                                     quadrature,
                                     update_quadrature_points);
  std::vector<std::pair<types::global_dof_index, Point<spacedim>>> dofs_and_pts;

  for (const auto &[coarse_box, coarse_cell] : coarse_tree)
    {
      typename DoFHandler<dim0, spacedim>::active_cell_iterator coarse_cell_dh(
        *coarse_cell, &coarse_dh);
      coarse_cell_dh->get_dof_indices(coarse_dofs);

      for (const auto &[space_box, fine_cell] :
           tree | bgi::adaptors::queried(bgi::intersects(coarse_box)))
        {
          // Collect all the DoFs and points that are falling in this fine cell
          typename DoFHandler<dim1, spacedim>::active_cell_iterator
            fine_cell_dh(*fine_cell, &fine_dh);
          fine_cell_dh->get_dof_indices(space_dofs);
          fe_values.reinit(fine_cell_dh);

          for (unsigned int i = 0; i < space_dofs.size(); ++i)
            {
              const auto &ref_p = coarse_mapping.transform_real_to_unit_cell(
                coarse_cell, fe_values.quadrature_point(i));
              if (GeometryInfo<dim0>::is_inside_unit_cell(ref_p))
                dofs_and_pts.emplace_back(space_dofs[i], ref_p);
            }
        }

      // Remove duplicate points in case of a continuous element.
      std::sort(dofs_and_pts.begin(),
                dofs_and_pts.end(),
                [](const auto &a, const auto &b) { return a.first < b.first; });
      dofs_and_pts.erase(std::unique(dofs_and_pts.begin(),
                                     dofs_and_pts.end(),
                                     [](const auto &a, const auto &b) {
                                       return a.first == b.first;
                                     }),
                         dofs_and_pts.end());

      // Distribute
      for (unsigned int i = 0; i < coarse_dofs.size(); ++i)
        for (unsigned int q = 0; q < dofs_and_pts.size(); ++q)
          prolongation_matrix(dofs_and_pts[q].first, coarse_dofs[i]) +=
            fe_space.shape_value(i, dofs_and_pts[q].second);

      dofs_and_pts.clear();
    }
  // Prolongate
  prolongation_matrix.vmult(dst, src);
}


template void
non_nested_prolongation<2, 2, 2, Vector<double>>(
  const GridTools::Cache<2> &coarse_cache,
  const GridTools::Cache<2> &fine_cache,
  const DoFHandler<2>       &coarse_dh,
  const DoFHandler<2>       &fine_dh,
  const FiniteElement<2>    &fe_space,
  const Vector<double>      &src,
  Vector<double>            &dst);
