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

  std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>
    slaves;
  slaves.reserve(vec_of_cells.size() - 1);
  for (const auto &cell : vec_of_cells)
    {
      if (cell->active_cell_index() != master_idx)
        {
          master_slave_relationships_iterators[cell->active_cell_index()] =
            master_slave_relationships_iterators[master_idx];
          slaves.push_back(cell);
        }
    }
  master2slaves[master_idx] = slaves;

  for (const types::global_cell_index idx : global_indices)
    {
      master_slave_relationships[idx] = master_idx; // mark each slave
    }

  master_slave_relationships[master_idx] = -1;

  master2polygon[master_idx] = n_agglomerations;
  ++n_agglomerations; // agglomeration has been performed, record it
  create_bounding_box(vec_of_cells, master_idx); // fill the vector of bboxes
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
  const Quadrature<dim> &quadrature_type,
  const typename Triangulation<dim, spacedim>::active_cell_iterator
    &master_cell) const
{
  Assert(is_master_cell(master_cell),
         ExcMessage("This must be a master cell."));
  Assert(quadrature_type.size() > 0,
         ExcMessage("Invalid size for the given Quadrature object"));
  FE_Nothing<dim, spacedim> dummy_fe;
  FEValues<dim, spacedim>   no_values(*mapping,
                                    dummy_fe,
                                    quadrature_type,
                                    update_quadrature_points |
                                      update_JxW_values); // only for quadrature
  std::vector<Point<dim>>   vec_pts;
  std::vector<double>       vec_JxWs;
  for (const auto &dummy_cell : cells)
    {
      no_values.reinit(dummy_cell);
      auto        q_points = no_values.get_quadrature_points(); // real qpoints
      const auto &JxWs     = no_values.get_JxW_values();

      std::transform(q_points.begin(),
                     q_points.end(),
                     std::back_inserter(vec_pts),
                     [&](const Point<spacedim> &p) { return p; });
      std::transform(JxWs.begin(),
                     JxWs.end(),
                     std::back_inserter(vec_JxWs),
                     [&](const double w) { return w; });
    }

  // Map back each point in real space by using the map associated to the
  // bounding box.
  std::vector<Point<dim>> unit_points(vec_pts.size());
  euler_mapping->transform_points_real_to_unit_cell(master_cell,
                                                    vec_pts,
                                                    unit_points);


  return Quadrature<dim>(unit_points, vec_JxWs);
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
        agglomerated_quadrature(agglo_cells, agglomeration_quad, cell);

      const double bbox_measure =
        bboxes[master2polygon.at(cell->active_cell_index())].volume();

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

  FEFaceValues<dim, spacedim> no_values(
    *mapping,
    dummy_fe,
    agglomeration_face_quad,
    update_quadrature_points | update_JxW_values |
      update_normal_vectors); // only for quadrature

  if (neighboring_cell.state() == IteratorState::valid)
    {
      no_values.reinit(neighboring_cell, local_face_idx);
      auto q_points = no_values.get_quadrature_points();

      const auto &JxWs    = no_values.get_JxW_values();
      const auto &normals = no_values.get_normal_vectors();

      const auto & bbox = bboxes[master2polygon.at(cell->active_cell_index())];
      const double bbox_measure = bbox.volume();
      std::vector<Point<spacedim>> unit_q_points;

      std::transform(q_points.begin(),
                     q_points.end(),
                     std::back_inserter(unit_q_points),
                     [&](const Point<spacedim> &p) {
                       return bbox.real_to_unit(p);
                     });

      // Weights must be scaled with det(J)*|J^-t n| for each quadrature point.
      // Use the fact that we are using a BBox, so the jacobi entries are the
      // side_length in each direction.
      // Unfortunately, normal vectors will be scaled internally by deal.II by
      // using a covariant transformation. In order not to change the normals,
      // we multiply by the correct factors in order to obtain the original
      // normal after the call to `reinit(cell)`.
      std::vector<double>         scale_factors(q_points.size());
      std::vector<double>         scaled_weights(q_points.size());
      std::vector<Tensor<1, dim>> scaled_normals(q_points.size());

      for (unsigned int q = 0; q < q_points.size(); ++q)
        {
          for (unsigned int direction = 0; direction < spacedim; ++direction)
            scaled_normals[q][direction] =
              normals[q][direction] * (bbox.side_length(direction));

          scaled_weights[q] =
            (JxWs[q] * scaled_normals[q].norm()) / bbox_measure;
          scaled_normals[q] /= scaled_normals[q].norm();
        }

      NonMatching::ImmersedSurfaceQuadrature<dim, spacedim> surface_quad(
        unit_q_points, scaled_weights, scaled_normals);

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

      // TODO: check if *mapping or *euler_mapping
      standard_scratch_face_bdary =
        std::make_unique<ScratchData>(*euler_mapping,
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
      // TODO: check if euler or mapping
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
      // This feature is not implemented yet

      const auto &fe_in = reinit_master(cell_in, local_in, agglomerated_isv);
      const auto &fe_out =
        reinit_master(neigh_cell, local_neigh, agglomerated_isv_neigh);
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
  // to couple only cells that are standard, not also slaves and master cells,
  // for which we need to compute DoFs separately later.

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


template <int dim, int spacedim>

void
AgglomerationHandler<dim, spacedim>::setup_output_interpolation_matrix()
{
  Assert(fe->has_generalized_support_points(),
         ExcMessage("The present FiniteElement is not interpolatory."));

  // Setup an auxiliary DoFHandler for output purposes
  output_dh.reinit(*tria);
  output_dh.distribute_dofs(*fe);

  DynamicSparsityPattern dsp(output_dh.n_dofs(), agglo_dh.n_dofs());

  std::vector<types::global_dof_index> agglo_dof_indices(fe->dofs_per_cell);
  std::vector<types::global_dof_index> standard_dof_indices(fe->dofs_per_cell);
  std::vector<types::global_dof_index> output_dof_indices(fe->dofs_per_cell);

  Quadrature<dim>         quad(fe->get_unit_support_points());
  FEValues<dim, spacedim> output_fe_values(*fe, quad, update_quadrature_points);

  for (const auto &cell : agglo_dh.active_cell_iterators())
    if (is_master_cell(cell))
      {
        auto slaves = get_slaves_of_idx(cell->active_cell_index());
        slaves.emplace_back(cell);

        cell->get_dof_indices(agglo_dof_indices);

        for (const auto &slave : slaves)
          {
            // addd master-slave relationship
            const auto slave_output = slave->as_dof_handler_iterator(output_dh);
            slave_output->get_dof_indices(output_dof_indices);
            for (const auto row : output_dof_indices)
              dsp.add_entries(row,
                              agglo_dof_indices.begin(),
                              agglo_dof_indices.end());
          }
      }
    else if (is_standard_cell(cell))
      {
        cell->get_dof_indices(agglo_dof_indices);
        const auto standard_output = cell->as_dof_handler_iterator(output_dh);
        standard_output->get_dof_indices(standard_dof_indices);
        for (const auto row : standard_dof_indices)
          dsp.add_entries(row,
                          agglo_dof_indices.begin(),
                          agglo_dof_indices.end());
      }
  output_interpolation_sparsity.copy_from(dsp);
  output_interpolation_matrix.reinit(output_interpolation_sparsity);

  FullMatrix<double>      local_matrix(fe->dofs_per_cell, fe->dofs_per_cell);
  std::vector<Point<dim>> reference_q_points(fe->dofs_per_cell);

  // Dummy DoFHandler, only needed for loc2glb
  AffineConstraints<double> c;
  c.close();

  for (const auto &cell : agglo_dh.active_cell_iterators())
    {
      if (is_master_cell(cell))
        {
          auto slaves = get_slaves_of_idx(cell->active_cell_index());
          slaves.emplace_back(cell);

          cell->get_dof_indices(agglo_dof_indices);

          for (const auto &slave : slaves)
            {
              // addd master-slave relationship
              const auto slave_output =
                slave->as_dof_handler_iterator(output_dh);

              slave_output->get_dof_indices(output_dof_indices);
              output_fe_values.reinit(slave_output);

              local_matrix = 0.;

              const auto &q_points = output_fe_values.get_quadrature_points();
              for (const auto i : output_fe_values.dof_indices())
                {
                  const auto &p =
                    bboxes[master2polygon.at(cell->active_cell_index())]
                      .real_to_unit(q_points[i]);
                  for (const auto j : output_fe_values.dof_indices())
                    {
                      local_matrix(i, j) = fe->shape_value(j, p);
                    }
                }
              c.distribute_local_to_global(local_matrix,
                                           output_dof_indices,
                                           agglo_dof_indices,
                                           output_interpolation_matrix);
            }
        }
      else if (is_standard_cell(cell))
        {
          cell->get_dof_indices(agglo_dof_indices);

          const auto standard_output = cell->as_dof_handler_iterator(output_dh);

          standard_output->get_dof_indices(standard_dof_indices);
          output_fe_values.reinit(standard_output);

          local_matrix = 0.;
          for (const auto i : output_fe_values.dof_indices())
            local_matrix(i, i) = 1.;
          c.distribute_local_to_global(local_matrix,
                                       standard_dof_indices,
                                       agglo_dof_indices,
                                       output_interpolation_matrix);
        }
    }
}



template <int dim, int spacedim>
double
AgglomerationHandler<dim, spacedim>::volume(
  const typename Triangulation<dim>::active_cell_iterator &cell) const
{
  Assert(!is_slave_cell(cell),
         ExcMessage("The present function cannot be called for slave cells."));

  if (is_master_cell(cell))
    {
      // Get the agglomerate
      // std::vector<typename Triangulation<dim,
      // spacedim>::active_cell_iterator>
      //   agglo_cells = get_slaves_of_idx(cell->active_cell_index());
      // // Push back master cell
      // agglo_cells.push_back(cell);

      // Quadrature<dim> quad =
      //   agglomerated_quadrature(agglo_cells,
      //                           QGauss<dim>{2 * fe->degree + 1},
      //                           cell);

      return bboxes[master2polygon.at(cell->active_cell_index())].volume();
      // return std::accumulate(quad.get_weights().begin(),
      //                        quad.get_weights().end(),
      //                        0.);
    }
  else
    {
      // Standard deal.II way to get the measure of a cell.
      return cell->measure();
    }
}



template <int dim, int spacedim>
double
AgglomerationHandler<dim, spacedim>::diameter(
  const typename Triangulation<dim>::active_cell_iterator &cell) const
{
  Assert(!is_slave_cell(cell),
         ExcMessage("The present function cannot be called for slave cells."));

  if (is_master_cell(cell))
    {
      // Get the bounding box associated with the master cell
      const auto &bdary_pts =
        bboxes[master2polygon.at(cell->active_cell_index())]
          .get_boundary_points();
      return (bdary_pts.second - bdary_pts.first).norm();
    }
  else
    {
      // Standard deal.II way to get the measure of a cell.
      return cell->diameter();
    }
}



template class AgglomerationHandler<1>;
template class AgglomerationHandler<2>;
template class AgglomerationHandler<3>;