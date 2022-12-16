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

#include <deal.II/grid/grid_generator.h>

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
  n_agglomerations = 0;
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
  std::vector<types::global_cell_index> global_indices;
  for (const auto &cell : vec_of_cells)
    global_indices.push_back(cell->active_cell_index());

  // Maximum index drives the selection of the master cell
  types::global_cell_index master_idx =
    *std::max_element(global_indices.begin(), global_indices.end());

  for (const types::global_cell_index idx : global_indices)
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

  for (const types::global_cell_index idx : global_indices)
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


  FEValues<dim, spacedim> no_values(*mapping,
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
      mapping->transform_points_real_to_unit_cell(cell, q_points, q_points);

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
  // Assert(n_agglomerations > 0,
  //        ExcMessage("No agglomeration has been performed."));
  for (const auto &cell : agglo_dh.active_cell_iterators())
    if (is_master_cell(cell))
      cell->set_active_fe_index(CellAgglomerationType::master);
    else if (is_slave_cell(cell))
      cell->set_active_fe_index(CellAgglomerationType::slave); // slave cell
    else
      cell->set_active_fe_index(CellAgglomerationType::standard); // standard

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
        agglomerated_quadrature(agglo_cells, agglomeration_quad);

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
      standard_scratch = std::make_unique<ScratchData>(*mapping,
                                                       fe_collection[2],
                                                       agglomeration_quad,
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
const FEValuesBase<dim, spacedim> &
AgglomerationHandler<dim, spacedim>::reinit_master(
  const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
  const unsigned int                                              face_number,
  std::unique_ptr<NonMatching::FEImmersedSurfaceValues<spacedim>>
    &agglo_isv_ptr) const
{
  Assert(is_master_cell(cell), ExcMessage("This cell must be a master one."));
  const auto &info_neighbors   = master_neighbors[{cell, face_number}];
  const auto &local_face_idx   = std::get<0>(info_neighbors);
  const auto &neighboring_cell = std::get<3>(info_neighbors);

  FE_Nothing<dim, spacedim> dummy_fe;
  DoFHandler<dim, spacedim> dummy_dh(*tria);
  dummy_dh.distribute_dofs(dummy_fe);

  FEFaceValues<dim, spacedim> no_values(
    *mapping,
    dummy_fe,
    agglomeration_face_quad,
    update_quadrature_points | update_JxW_values |
      update_normal_vectors); // only for quadrature

  if (neighboring_cell.state() == IteratorState::valid)
    {
      no_values.reinit(neighboring_cell, local_face_idx);
      std::cout << "reinited inside" << std::endl;
      auto q_points = no_values.get_quadrature_points();

      const auto &JxWs    = no_values.get_JxW_values();
      const auto &normals = no_values.get_normal_vectors();

      const auto &bbox = bboxes[cell->active_cell_index()];

      const double bbox_measure = bbox.volume();
      std::cout << "Got the volume = " << bbox_measure << std::endl;
      std::vector<Point<spacedim>> unit_q_points;

      for (const auto &p : q_points)
        {
          std::cout << "Point: " << p << "inside?" << bbox.point_inside(p)
                    << std::endl;
          std::cout << bbox.real_to_unit(p) << std::endl;
        }

      std::cout << "Done" << std::endl;
      std::transform(q_points.begin(),
                     q_points.end(),
                     std::back_inserter(unit_q_points),
                     [&](const Point<spacedim> &p) {
                       return /*euler_mapping->transform_real_to_unit_cell(cell,
                                                                         p);*/
                         bbox.real_to_unit(p);
                     });
      std::cout << "Got the points" << std::endl;

      // Weights must be scaled with det(J)*|J^-t n| for each quadrature point.
      // Use the fact that we are using a AABBox, so the jacobi entries are the
      // side_length in each direction and normals are already available at this
      // point.
      std::vector<double> scale_factors(q_points.size());
      std::vector<double> scaled_weights(q_points.size());
      Tensor<1, spacedim> scale;

      for (unsigned int q = 0; q < q_points.size(); ++q)
        {
          for (unsigned int direction = 0; direction < spacedim; ++direction)
            {
              scale[direction] =
                normals[q][direction] / (bbox.side_length(direction));
            }

          // scale_factors[q]  = bbox_measure * scale.norm();
          scaled_weights[q] = JxWs[q] / (bbox_measure * scale.norm());
        }
      std::cout << "Scaled the weights" << std::endl;

      NonMatching::ImmersedSurfaceQuadrature<dim, spacedim> surface_quad(
        unit_q_points, scaled_weights, normals);

      agglo_isv_ptr =
        std::make_unique<NonMatching::FEImmersedSurfaceValues<spacedim>>(
          *euler_mapping, *fe, surface_quad, agglomeration_face_flags);

      agglo_isv_ptr->reinit(cell);

      return *agglo_isv_ptr;
    }
  else
    {
      // Then it's a boundary face of an agglomeration living on the
      // boundary of the tria. You need to return an FEFaceValues on the
      // boundary face of a boundary cell.
      no_values.reinit(neighboring_cell, local_face_idx);

      standard_scratch_face_bdary =
        std::make_unique<ScratchData>(*mapping,
                                      fe_collection[2],
                                      agglomeration_quad,
                                      agglomeration_flags,
                                      agglomeration_face_quad,
                                      agglomeration_face_flags);

      return standard_scratch_face_bdary->reinit(cell);
    }
}



template <int dim, int spacedim>
const FEValuesBase<dim, spacedim> &
AgglomerationHandler<dim, spacedim>::reinit(
  const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
  const unsigned int face_number) const
{
  Assert(euler_mapping,
         ExcMessage("The mapping describing the physical element stemming from "
                    "agglomeration has not been set up."));

  if ((is_standard_cell(cell) && at_boundary(cell, face_number)) ||
      (is_standard_cell(cell) &&
       is_master_cell(agglomerated_neighbor(cell, face_number))))
    {
      standard_scratch_face_any =
        std::make_unique<ScratchData>(*mapping,
                                      fe_collection[2],
                                      agglomeration_quad,
                                      agglomeration_flags,
                                      agglomeration_face_quad,
                                      agglomeration_face_flags);
      return standard_scratch_face_any->reinit(cell, face_number);
    }
  else
    {
      Assert(is_master_cell(cell), ExcMessage("This should be true."));
      return reinit_master(cell, face_number, agglomerated_isv_bdary);
    }
}



template <int dim, int spacedim>
std::pair<const FEValuesBase<dim, spacedim> &,
          const FEValuesBase<dim, spacedim> &>
AgglomerationHandler<dim, spacedim>::reinit_interface(
  const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell_in,
  const typename DoFHandler<dim, spacedim>::active_cell_iterator &neigh_cell,
  const unsigned int                                              local_in,
  const unsigned int local_neigh) const
{
  Assert(
    !is_slave_cell(cell_in) && !is_slave_cell(neigh_cell),
    ExcMessage(
      "At least of the two cells sharing a face is a slave cell. This should never happen if you want to agglomerate some cells together. "));

  if (is_standard_cell(cell_in) && is_standard_cell(neigh_cell))
    {
      standard_scratch_face_std =
        std::make_unique<ScratchData>(*mapping,
                                      fe_collection[2],
                                      agglomeration_quad,
                                      agglomeration_flags,
                                      agglomeration_face_quad,
                                      agglomeration_face_flags);

      standard_scratch_face_std_neigh =
        std::make_unique<ScratchData>(*mapping,
                                      fe_collection[2],
                                      agglomeration_quad,
                                      agglomeration_flags,
                                      agglomeration_face_quad,
                                      agglomeration_face_flags);

      std::pair<const FEValuesBase<dim, spacedim> &,
                const FEValuesBase<dim, spacedim> &>
        my_p(standard_scratch_face_std->reinit(cell_in, local_in),
             standard_scratch_face_std_neigh->reinit(neigh_cell, local_neigh));

      return my_p;
    }
  else if (is_standard_cell(neigh_cell) && is_master_cell(cell_in))
    {
      const auto &fe_in = reinit(cell_in, local_in);
      standard_scratch_face_std_another =
        std::make_unique<ScratchData>(*mapping,
                                      fe_collection[2],
                                      agglomeration_quad,
                                      agglomeration_flags,
                                      agglomeration_face_quad,
                                      agglomeration_face_flags);

      std::pair<const FEValuesBase<dim, spacedim> &,
                const FEValuesBase<dim, spacedim> &>
        my_p(fe_in,
             standard_scratch_face_std_another->reinit(neigh_cell,
                                                       local_neigh));
      return my_p;
    }
  else if (is_standard_cell(cell_in) && is_master_cell(neigh_cell))
    {
      const auto &fe_out = reinit(neigh_cell, local_neigh);
      standard_scratch_face_std_another =
        std::make_unique<ScratchData>(*mapping,
                                      fe_collection[2],
                                      agglomeration_quad,
                                      agglomeration_flags,
                                      agglomeration_face_quad,
                                      agglomeration_face_flags);

      std::pair<const FEValuesBase<dim, spacedim> &,
                const FEValuesBase<dim, spacedim> &>
        my_p(standard_scratch_face_std_another->reinit(cell_in, local_in),
             fe_out);

      return my_p;
    }
  else
    {
      Assert(is_master_cell(cell_in) && is_master_cell(neigh_cell),
             ExcMessage("Both cells should be masters."));
      // both are masters. That means you want to compute the jumps or
      // averages between a face shared by two neighboring agglomerations.

      const auto &fe_in = reinit_master(cell_in, local_in, agglomerated_isv);
      std::cout << "First master" << std::endl;
      const auto &fe_out =
        reinit_master(neigh_cell, local_neigh, agglomerated_isv_neigh);
      std::cout << "Second master" << std::endl;
      std::pair<const FEValuesBase<dim, spacedim> &,
                const FEValuesBase<dim, spacedim> &>
        my_p(fe_in, fe_out);

      return my_p;
    }
}



template <int dim, int spacedim>
void
AgglomerationHandler<dim, spacedim>::create_agglomeration_sparsity_pattern(
  SparsityPattern &sparsity_pattern)
{
  // Assert(n_agglomerations > 0,
  //        ExcMessage("The agglomeration has not been set up correctly."));
  Assert(sparsity_pattern.empty(),
         ExcMessage(
           "The Sparsity pattern must be empty upon calling this function."));

  DynamicSparsityPattern    dsp(agglo_dh.n_dofs(), agglo_dh.n_dofs());
  AffineConstraints<double> constraints;
  const bool                keep_constrained_dofs = true;
  // The following lambda is used to teach to `make_flux_sparsity_pattern()`
  // to couple only cells that are standard, not also slaves and master cells.
  // Indeed, for them we need to compute DoFs separately later.

  const auto face_has_flux_coupling =
    [&](const auto &cell, const types::global_cell_index face_index) {
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
      if (neigh.state() == IteratorState::valid)
        {
          typename DoFHandler<dim, spacedim>::cell_iterator cell_slave(
            *neigh, &agglo_dh);
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
    }


  sparsity_pattern.copy_from(dsp);
}

template class AgglomerationHandler<1>;
template class AgglomerationHandler<2>;
template class AgglomerationHandler<3>;