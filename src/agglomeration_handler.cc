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
  const GridTools::Cache<dim, spacedim> &cache_tria)
  : cached_tria(std::make_unique<GridTools::Cache<dim, spacedim>>(
      cache_tria.get_triangulation(),
      cache_tria.get_mapping()))
{
  Assert(dim == spacedim, ExcNotImplemented("Not available with codim > 0"));
  Assert(dim == 2 || dim == 3, ExcMessage("Not available in 1D."));
  Assert(cached_tria->get_triangulation().n_active_cells() > 0,
         ExcMessage(
           "The triangulation must not be empty upon calling this function."));
  initialize_agglomeration_data(cached_tria);
}



template <int dim, int spacedim>
void
AgglomerationHandler<dim, spacedim>::agglomerate_cells(
  const std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>
    &vec_of_cells)
{
  Assert(master_slave_relationships.size() > 0,
         ExcMessage("Before calling this function, be sure that the "
                    "constructor of this object has been called."));
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

  for (const auto &cell : vec_of_cells)
    {
      if (cell->active_cell_index() == master_idx)
        master_slave_relationships_iterators[cell->active_cell_index()] =
          cell; // set iterator to master cell
    }

  for (const auto &cell : vec_of_cells)
    {
      if (cell->active_cell_index() != master_idx)
        master_slave_relationships_iterators[cell->active_cell_index()] =
          master_slave_relationships_iterators[master_idx];
    }

  for (const unsigned int idx : global_indices)
    {
      master_slave_relationships[idx] = master_idx; // mark each slave
    }

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

  // Loop over the tria, and check if a each cell is a slave of master cell
  // idx If no slave is found, return an empty vector.
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
                        &cells,
  const Quadrature<dim> &quadrature_type) const
{
  Assert(quadrature_type.size() > 0,
         ExcMessage("Invalid size for the given Quadrature object"));
  FE_Nothing<dim, spacedim> dummy_fe;
  DoFHandler<dim, spacedim> dummy_dh(*tria);
  dummy_dh.distribute_dofs(dummy_fe);
  MappingQ<dim, spacedim> mapping_generic(1);

  FEValues<dim, spacedim> no_values(
    mapping_generic,
    dummy_fe,
    quadrature_type,
    update_quadrature_points |
      update_JxW_values); // only for quadrature, see related issue.
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
  Assert(agglo_dh.get_triangulation().n_cells() > 0,
         ExcMessage(
           "Triangulation must not be empty upon calling this function."));
  Assert(n_agglomerations > 0,
         ExcMessage("No agglomeration has been performed."));
  for (const auto &cell : agglo_dh.active_cell_iterators())
    if (is_master_cell(cell))
      cell->set_active_fe_index(AggloIndex::master);
    else if (is_slave_cell(cell))
      cell->set_active_fe_index(AggloIndex::slave); // slave cell
    else
      cell->set_active_fe_index(AggloIndex::standard); // standard

  agglo_dh.distribute_dofs(fe_collection);
  euler_mapping =
    std::make_unique<MappingFEField<dim, spacedim>>(euler_dh, euler_vector);
}



template <int dim, int spacedim>
const FEValues<dim, spacedim> &
AgglomerationHandler<dim, spacedim>::reinit(
  const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell) const
{
  Assert(euler_mapping,
         ExcMessage("The mapping describing the physical element stemming from "
                    "agglomeration has not been set up."));

  if (is_master_cell(cell))
    {
      std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>
        agglo_cells;
      // Push back that master and slaves
      agglo_cells.push_back(cell);
      const auto &slaves = get_slaves_of_idx(cell->active_cell_index());
      std::transform(
        slaves.begin(),
        slaves.end(),
        std::back_inserter(agglo_cells),
        [&](const typename Triangulation<dim, spacedim>::active_cell_iterator
              &c) { return c; });

      Quadrature<dim> agglo_quad =
        agglomerated_quadrature(agglo_cells,
                                QGauss<dim>(agglomeration_quadrature_degree));

      const double bbox_measure = bboxes[cell->active_cell_index()].volume();

      // Scale weights with the volume of the BBox. This way, the euler_mapping
      // defining the BBOx doesn't alter them.
      std::vector<double> scaled_weights;
      std::transform(agglo_quad.get_weights().begin(),
                     agglo_quad.get_weights().end(),
                     std::back_inserter(scaled_weights),
                     [&bbox_measure](const double w) {
                       return w / bbox_measure;
                     });

      Quadrature<dim> scaled_quad(agglo_quad.get_points(), scaled_weights);

      agglomerated_scratch = std::make_unique<ScratchData>(*euler_mapping,
                                                           fe_collection[0],
                                                           scaled_quad,
                                                           agglomeration_flags);
      return agglomerated_scratch->reinit(cell);
    }
  else if (is_standard_cell(cell))
    {
      // ensure the DG space is the same we have from the other DoFHandler(s)
      standard_scratch =
        std::make_unique<ScratchData>(*mapping,
                                      fe_collection[2],
                                      QGauss<dim>(
                                        agglomeration_quadrature_degree),
                                      agglomeration_flags);
      return standard_scratch->reinit(cell);
    }
  else
    {
      std::vector<Point<dim>> pts{{}};
      std::vector<double>     wgts{0.};
      Quadrature<dim>         dummy_quad(pts, wgts);
      standard_scratch = std::make_unique<ScratchData>(*mapping,
                                                       fe_collection[1],
                                                       dummy_quad,
                                                       agglomeration_flags);
      return standard_scratch->reinit(cell);
    }
}



template <int dim, int spacedim>
const NonMatching::FEImmersedSurfaceValues<dim> &
AgglomerationHandler<dim, spacedim>::reinit(
  const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
  const unsigned int agglomeration_face_number) const
{
  // For now, this must be called for master cells only.
  Assert(master_slave_relationships[cell->active_cell_index()] == -1,
         ExcMessage(
           "This function is supposed to be called for master cells."));
  FE_Nothing<dim, spacedim> dummy_fe;
  DoFHandler<dim, spacedim> dummy_dh(*tria);
  dummy_dh.distribute_dofs(dummy_fe);
  MappingQ<dim, spacedim> mapping_generic(1);

  FEFaceValues<dim, spacedim> no_values(
    mapping_generic,
    dummy_fe,
    QGauss<dim - 1>(agglomeration_quadrature_degree),
    update_quadrature_points | update_JxW_values |
      update_normal_vectors); // only for quadrature


  const auto &info_neighbors =
    master_neighbors[{cell, agglomeration_face_number}];
  const auto &local_face_idx     = std::get<0>(info_neighbors);
  const auto &neighboring_cell   = std::get<1>(info_neighbors);
  const auto &local_face_idx_out = std::get<2>(info_neighbors);

  no_values.reinit(neighboring_cell->neighbor(local_face_idx_out),
                   local_face_idx);

  auto        q_points = no_values.get_quadrature_points();
  const auto &JxWs     = no_values.get_JxW_values();
  const auto &normals  = no_values.get_normal_vectors();

  typename DoFHandler<dim, spacedim>::cell_iterator cell_dh(
    *neighboring_cell->neighbor(local_face_idx_out), &euler_dh);
  mapping_generic.transform_points_real_to_unit_cell(cell_dh,
                                                     q_points,
                                                     q_points);
  NonMatching::ImmersedSurfaceQuadrature<dim, spacedim> surface_quad(q_points,
                                                                     JxWs,
                                                                     normals);

  agglomerated_isv =
    std::make_unique<NonMatching::FEImmersedSurfaceValues<spacedim>>(
      *euler_mapping, *fe, surface_quad, agglomeration_face_flags);

  agglomerated_isv->reinit(cell);

  // Weights must be scaled with det(J)*J^-t n for each quadrature point now.
  const double        bbox_measure = bboxes[cell->active_cell_index()].volume();
  std::vector<double> scale_factors;
  for (unsigned int i = 0; i < q_points.size(); ++i)
    {
      // J^-t*n
      const auto &JmT = (agglomerated_isv->inverse_jacobian(i)).transpose();
      scale_factors.push_back(bbox_measure *
                              apply_transformation(JmT, normals[i]).norm());
    }

  // Scale original weights properly
  std::vector<double> scaled_weights(JxWs.size());
  for (unsigned int i = 0; i < JxWs.size(); ++i)
    {
      scaled_weights[i] = JxWs[i] / scale_factors[i];
    }

  // update the Quadrature rule
  surface_quad.initialize(q_points, scaled_weights);

  // update the ptr to FEIsv and finally call reinit
  agglomerated_isv.reset(new NonMatching::FEImmersedSurfaceValues<spacedim>(
    *euler_mapping, *fe, surface_quad, agglomeration_face_flags));
  agglomerated_isv->reinit(cell);

  return *agglomerated_isv;
}



template <int dim, int spacedim>
void
AgglomerationHandler<dim, spacedim>::create_agglomeration_sparsity_pattern(
  SparsityPattern &sparsity_pattern)
{
  Assert(n_agglomerations > 0,
         ExcMessage("The agglomeration has not been set up correctly."));
  Assert(sparsity_pattern.empty(),
         ExcMessage(
           "The Sparsity pattern must be empty upon calling this function."));

  DynamicSparsityPattern    dsp(agglo_dh.n_dofs(), agglo_dh.n_dofs());
  AffineConstraints<double> constraints;
  const bool                keep_constrained_dofs = true;
  // The following lambda is used to teach to `make_flux_sparsity_pattern()` to
  // couple only cells that are standard, not also slaves and master cells, for
  // which we need to compute DoFs separately later.

  const auto face_has_flux_coupling = [&](const auto        &cell,
                                          const unsigned int face_index) {
    return master_slave_relationships[cell->active_cell_index()] *
             master_slave_relationships[cell->neighbor(face_index)
                                          ->active_cell_index()] ==
           +4;
  };
  const unsigned int           n_components = fe_collection.n_components();
  Table<2, DoFTools::Coupling> cell_coupling(n_components, n_components);
  Table<2, DoFTools::Coupling> face_coupling(n_components, n_components);
  cell_coupling[0][0] = DoFTools::always;
  face_coupling[0][0] = DoFTools::always;
  DoFTools::make_flux_sparsity_pattern(agglo_dh,
                                       dsp,
                                       constraints,
                                       keep_constrained_dofs,
                                       cell_coupling,
                                       face_coupling,
                                       numbers::invalid_subdomain_id,
                                       face_has_flux_coupling);


  std::vector<types::global_dof_index> dof_indices_master(
    agglo_dh.get_fe(0).n_dofs_per_cell());
  std::vector<types::global_dof_index> dof_indices_neighbor(
    agglo_dh.get_fe(2).n_dofs_per_cell());

  // Get the information about the neighboring
  // cells and add the corresponding entries to the sparsity pattern.

  for (const auto &value : master_neighbors)
    {
      // value.first is a pair storing master_cell and agglo_face index.
      // Another `first` is necessary to get the cell.
      const auto &master_cell = (value.first).first;
      typename DoFHandler<dim, spacedim>::cell_iterator master_cell_dh(
        *master_cell, &agglo_dh);
      master_cell_dh->get_dof_indices(dof_indices_master);

      const auto &neigh = std::get<1>(value.second);
      typename DoFHandler<dim, spacedim>::cell_iterator cell_slave(*neigh,
                                                                   &agglo_dh);
      cell_slave->get_dof_indices(dof_indices_neighbor);
      for (const unsigned int row_idx : dof_indices_master)
        dsp.add_entries(row_idx,
                        dof_indices_neighbor.begin(),
                        dof_indices_neighbor.end());
      for (const unsigned int col_idx : dof_indices_neighbor)
        dsp.add_entries(col_idx,
                        dof_indices_master.begin(),
                        dof_indices_master.end());
    }


  sparsity_pattern.copy_from(dsp);
}

template class AgglomerationHandler<1>;
template class AgglomerationHandler<2>;
template class AgglomerationHandler<3>;
