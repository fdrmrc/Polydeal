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
#ifndef non_nested_transfer_h
#define non_nested_transfer_h

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>

/**
 * Prolongate the vector `src` living on the coarse grid cached in `cache0` onto
 * the second grid cached onto `cache1` and store it in `dst`.
 *
 */
using namespace dealii;
template <int dim0, int dim1, int spacedim = dim0, typename VectorType>
void
non_nested_prolongation(const GridTools::Cache<dim0, spacedim> &coarse_cache,
                        const GridTools::Cache<dim1, spacedim> &fine_cache,
                        const DoFHandler<dim0, spacedim>       &coarse_dh,
                        const DoFHandler<dim1, spacedim>       &fine_dh,
                        const FiniteElement<dim0, spacedim>    &fe_space,
                        const VectorType                       &src,
                        VectorType                             &dst);


#endif
