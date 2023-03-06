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


#include "agglomerator.h"

template <int dim, int spacedim>
Agglomerator<dim, spacedim>::Agglomerator(
  const GridTools::Cache<dim, spacedim> &cache_tria,
  const Function<dim>                   &level_set,
  const Quadrature<1>                   &quad_1D)
  : cached_tria(std::make_unique<GridTools::Cache<dim, spacedim>>(cache_tria))
  , level_set_function(std::make_unique<Function<dim>>(level_set))
  , quadrature_1D(std::make_unique<Quadrature<1>>(quad_1D))
  , level_set_dof_handler(cache_tria->get_triangulation())
  , level_set_fe(1)
  , mesh_classifier(level_set_dof_handler, level_set)
{
  // Set up the discrete level set function by interpolating onto a Lagrangian
  // finite element space.
  level_set_dof_handler.distribute_dofs(level_set_fe);
  level_set_vector.reinit(level_set_dof_handler.n_dofs());
  VectorTools::interpolate(level_set_dof_handler,
                           level_set_function,
                           level_set_vector);
}



template <int dim, int spacedim>
void
Agglomerator<dim, spacedim>::identify_cut_cells()
{
  Assert(level_set_vector.size() > 0,
         ExcMessage(
           "The level set describing the interface has not been set up."));

  NonMatching::RegionUpdateFlags region_update_flags;
  region_update_flags.inside  = update_JxW_values | update_quadrature_points;
  region_update_flags.outside = update_JxW_values | update_quadrature_points;

  // Loop over all cells of the tria and use MeshClassifier to get the proper
  // location.
  NonMatching::FEValues<dim> non_matching_fe_values(level_set_fe,
                                                    quadrature_1D,
                                                    region_update_flags,
                                                    mesh_classifier,
                                                    level_set_dof_handler,
                                                    level_set_vector);

  for (const auto &cell :
       cached_tria->get_triangulation().active_cell_iterators())
    {
      const NonMatching::LocationToLevelSet cell_location =
        mesh_classifier.location_to_level_set(cell);
      if (cell_location == NonMatching::LocationToLevelSet::intersected)
        {
          non_matching_fe_values.reinit(cell);
          // Push back su diversi (sets) a seconda dell'area delle subcelle
        }
    }
}
