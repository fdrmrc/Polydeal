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


#include <agglomeration_handler.h>



template <int dim, int spacedim>
AgglomerationHandler<dim, spacedim>::AgglomerationHandler(
  const std::unique_ptr<GridTools::Cache<dim, spacedim>> &cached_tria)
  : agglo_dh(cached_tria->get_triangulation())
  , euler_fe(
      std::make_unique<FESystem<dim, spacedim>>(FE_DGQ<spacedim>(1), spacedim))
  , euler_dh(cached_tria->get_triangulation())
{
  Assert(dim == spacedim, ExcMessage("Not tested with different dimensions"));
  Assert(dim == 2 || dim == 3, ExcMessage("Not available in 1D."));
  Assert(cached_tria->get_triangulation().n_active_cells() > 0,
         ExcMessage(
           "The triangulation must not be empty upon calling this function."));
  tria    = &cached_tria->get_triangulation();
  mapping = &cached_tria->get_mapping();
  fe_collection.push_back(FE_DGQ<dim, spacedim>(1));
  fe_collection.push_back(FE_Nothing<dim, spacedim>());
  // All cells are initially marked with -2, while -1 is reserved for master
  // cells.
  master_slave_relationships.resize(
    cached_tria->get_triangulation().n_active_cells(), -2);
  bboxes.resize(tria->n_active_cells());
  euler_dh.distribute_dofs(*euler_fe);
  euler_vector.reinit(euler_dh.n_dofs());
}



template <int dim, int spacedim>
void
AgglomerationHandler<dim, spacedim>::agglomerate_cells(
  const std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>
    &vec_of_cells)
{
  Assert(
    master_slave_relationships.size() > 0,
    ExcMessage(
      "Before calling this function, be sure that the constructor of this object has been called."));
  Assert(vec_of_cells.size() >= 1, ExcMessage("No cells to be agglomerated."));

  // Get global index for each cell
  std::vector<unsigned int> global_indices;
  for (const auto &cell : vec_of_cells)
    global_indices.push_back(cell->active_cell_index());

  // Maximum index drives the selection of the master cell
  unsigned int master_idx =
    *std::max_element(global_indices.begin(), global_indices.end());

  for (const unsigned int idx : global_indices)
    master_slave_relationships[idx] = master_idx; // mark each slave
  master_slave_relationships[master_idx] = -1;

  ++n_agglomerations; // agglomeration has been performed, record it
  create_bounding_box(vec_of_cells, master_idx); // fill the vector of bboxes
}


template <int dim, int spacedim>
std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>
AgglomerationHandler<dim, spacedim>::get_slaves_of_idx(const int idx) const
{
  std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>
    slaves;
  // Loop over the tria, and check if a each cell is a slave of master cell idx
  // If no slave is found, return an empty vector.
  for (const auto &cell : tria->active_cell_iterators())
    {
      if (master_slave_relationships[cell->active_cell_index()] == idx)
        {
          slaves.push_back(cell);
        }
    }
  return slaves;
}



template <int dim, int spacedim>
std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>
AgglomerationHandler<dim, spacedim>::get_agglomerated_cells(
  const typename Triangulation<dim, spacedim>::active_cell_iterator &cell) const
{
  const int current_idx = cell->active_cell_index();
  return get_slaves_of_idx(current_idx);
}



template <int dim, int spacedim>
Quadrature<dim>
AgglomerationHandler<dim, spacedim>::agglomerated_quadrature(
  const std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>
    &                    cells,
  const Quadrature<dim> &quadrature_type) const
{
  Assert(quadrature_type.size() > 0,
         ExcMessage("Invalid size for the given Quadrature object"));
  FE_Nothing<dim, spacedim> dummy_fe;
  DoFHandler<dim, spacedim> dummy_dh(*tria);
  dummy_dh.distribute_dofs(dummy_fe);
  MappingQ<dim, spacedim> mapping_generic(1);

  FEValues<dim, spacedim> no_values(mapping_generic,
                                    dummy_fe,
                                    quadrature_type,
                                    update_quadrature_points |
                                      update_JxW_values); // only for quadrature
  std::vector<Point<dim>> vec_pts;
  std::vector<double>     vec_JxWs;
  for (const auto &dummy_cell : cells)
    {
      no_values.reinit(dummy_cell);
      auto        q_points = no_values.get_quadrature_points();
      const auto &JxWs     = no_values.get_JxW_values();

      typename DoFHandler<dim, spacedim>::cell_iterator cell(*dummy_cell,
                                                             &euler_dh);
      mapping_generic.transform_points_real_to_unit_cell(cell,
                                                         q_points,
                                                         q_points);

      std::transform(q_points.begin(),
                     q_points.end(),
                     std::back_inserter(vec_pts),
                     [&](const Point<spacedim> &p) { return p; });
      std::transform(JxWs.begin(),
                     JxWs.end(),
                     std::back_inserter(vec_JxWs),
                     [&](const double w) { return w; });
    }

  return Quadrature<dim>(vec_pts, vec_JxWs);
}



template <int dim, int spacedim>
void
AgglomerationHandler<dim, spacedim>::initialize_hp_structure()
{
  Assert(n_agglomerations > 0,
         ExcMessage("No agglomeration has been performed."));
  Assert(agglo_dh.get_triangulation().n_cells() > 0,
         ExcMessage(
           "Triangulation must not be empty upon calling this function."));
  for (const auto &cell : agglo_dh.active_cell_iterators())
    if (is_master_cell(cell))
      cell->set_active_fe_index(AggloIndex::master);
    else
      cell->set_active_fe_index(AggloIndex::slave);

  agglo_dh.distribute_dofs(fe_collection);
  euler_mapping =
    std::make_unique<MappingFEField<dim, spacedim>>(euler_dh, euler_vector);
}



template <int dim, int spacedim>
const FEValues<dim, spacedim> &
AgglomerationHandler<dim, spacedim>::reinit(
  const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell) const
{
  Assert(
    euler_mapping,
    ExcMessage(
      "The mapping describing the physical element stemming from agglomeration has not been set up."));
  Assert(master_slave_relationships[cell->active_cell_index()] == -1,
         ExcInternalError("The present cell must be a master cell."));

  std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>
    agglo_cells;
  // Push back that master and slaves
  agglo_cells.push_back(cell);
  const auto &slaves = get_slaves_of_idx(cell->active_cell_index());
  std::transform(
    slaves.begin(),
    slaves.end(),
    std::back_inserter(agglo_cells),
    [&](const typename Triangulation<dim, spacedim>::active_cell_iterator &c) {
      return c;
    });

  Quadrature<dim> agglo_quad =
    agglomerated_quadrature(agglo_cells,
                            QGauss<dim>(2 * agglo_dh.get_fe().degree + 1));

  agglomerated_scratch =
    std::make_unique<ScratchData>(*euler_mapping,
                                  agglo_dh.get_fe(),
                                  agglo_quad,
                                  update_values | update_quadrature_points);
  //@todo Give flags in proper way, without hardcoding
  return agglomerated_scratch->reinit(cell);
}

template class AgglomerationHandler<1>;
template class AgglomerationHandler<2>;
template class AgglomerationHandler<3>;
