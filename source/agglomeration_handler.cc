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
  , communicator(cache_tria.get_triangulation().get_communicator())
{
  Assert(dim == spacedim, ExcNotImplemented("Not available with codim > 0"));
  Assert(dim == 2 || dim == 3, ExcImpossibleInDim(1));
  Assert((dynamic_cast<const parallel::shared::Triangulation<dim, spacedim> *>(
            &cached_tria->get_triangulation()) == nullptr),
         ExcNotImplemented());
  Assert(cached_tria->get_triangulation().n_active_cells() > 0,
         ExcMessage(
           "The triangulation must not be empty upon calling this function."));
  n_agglomerations = 0;
  hybrid_mesh      = false;
  initialize_agglomeration_data(cached_tria);
}



template <int dim, int spacedim>
typename AgglomerationHandler<dim, spacedim>::agglomeration_iterator
AgglomerationHandler<dim, spacedim>::define_agglomerate(
  const AgglomerationContainer &cells)
{
  Assert(master_slave_relationships.size() > 0,
         ExcMessage("Before calling this function, be sure that the "
                    "constructor of this object has been called."));
  Assert(cells.size() > 0, ExcMessage("No cells to be agglomerated."));

  if (cells.size() == 1)
    hybrid_mesh = true; // mesh is made also by classical cells

  // First index drives the selection of the master cell. After that, store the
  // master cell.
  const types::global_cell_index master_idx = cells[0]->active_cell_index();
  master_cells_container.push_back(cells[0]);
  master_slave_relationships[master_idx] = -1;

  // Store slave cells and save the relationship with the parent
  std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>
    slaves;
  slaves.reserve(cells.size() - 1);
  // exclude first cell since it's the master cell
  for (auto it = ++cells.begin(); it != cells.end(); ++it)
    {
      slaves.push_back(*it);
      master_slave_relationships[(*it)->active_cell_index()] =
        master_idx; // mark each slave
      master_slave_relationships_iterators[(*it)->active_cell_index()] =
        cells[0];

      // If we have a p::d::T, check that all cells are in the same subdomain.
      // If serial, just check that the subdomain_id is invalid.
      Assert(((*it)->subdomain_id() == tria->locally_owned_subdomain() ||
              tria->locally_owned_subdomain() == numbers::invalid_subdomain_id),
             ExcInternalError());
    }

  master_slave_relationships_iterators[master_idx] =
    cells[0]; // set iterator to master cell

  // Store the slaves of each master
  master2slaves[master_idx] = slaves;
  // Save to which polygon this agglomerate correspond
  master2polygon[master_idx] = n_agglomerations;

  ++n_agglomerations; // an agglomeration has been performed, record it

  create_bounding_box(cells, master_idx); // fill the vector of bboxes

  // Finally, return a polygonal iterator to the polytope just constructed.
  return {cells[0], this};
}



template <int dim, int spacedim>
void
AgglomerationHandler<dim, spacedim>::initialize_fe_values(
  const Quadrature<dim> &    cell_quadrature,
  const UpdateFlags &        flags,
  const Quadrature<dim - 1> &face_quadrature,
  const UpdateFlags &        face_flags)
{
  agglomeration_quad       = cell_quadrature;
  agglomeration_flags      = flags;
  agglomeration_face_quad  = face_quadrature;
  agglomeration_face_flags = face_flags | internal_agglomeration_face_flags;


  no_values =
    std::make_unique<FEValues<dim>>(*mapping,
                                    dummy_fe,
                                    agglomeration_quad,
                                    update_quadrature_points |
                                      update_JxW_values); // only for quadrature
  no_face_values = std::make_unique<FEFaceValues<dim>>(
    *mapping,
    dummy_fe,
    agglomeration_face_quad,
    update_quadrature_points | update_JxW_values |
      update_normal_vectors); // only for quadrature

  if (hybrid_mesh)
    {
      // the mesh is composed by standard and agglomerate cells. initialize
      // classes needed for standard cells in order to treat that finite element
      // space as defined on a standard shape and not on the BoundingBox.
      standard_scratch =
        std::make_unique<ScratchData>(*mapping,
                                      *fe,
                                      cell_quadrature,
                                      agglomeration_flags,
                                      face_quadrature,
                                      agglomeration_face_flags);
    }
}



template <int dim, int spacedim>
unsigned int
AgglomerationHandler<dim, spacedim>::n_agglomerated_faces_per_cell(
  const typename Triangulation<dim, spacedim>::active_cell_iterator &cell) const
{
  unsigned int n_neighbors = 0;
  for (const auto &f : cell->face_indices())
    {
      const auto &neighboring_cell = cell->neighbor(f);
      if ((cell->face(f)->at_boundary()) ||
          (neighboring_cell->is_active() &&
           !are_cells_agglomerated(cell, neighboring_cell)))
        {
          ++n_neighbors;
        }
    }
  return n_neighbors;
}



template <int dim, int spacedim>
void
AgglomerationHandler<dim, spacedim>::initialize_agglomeration_data(
  const std::unique_ptr<GridTools::Cache<dim, spacedim>> &cache_tria)
{
  tria    = &cache_tria->get_triangulation();
  mapping = &cache_tria->get_mapping();

  agglo_dh.reinit(*tria);
  FE_DGQ<dim, spacedim> dummy_dg(1);
  euler_fe = std::make_unique<FESystem<dim, spacedim>>(dummy_dg, spacedim);
  euler_dh.reinit(*tria);
  euler_dh.distribute_dofs(*euler_fe);
  euler_vector.reinit(euler_dh.n_dofs());

  master_slave_relationships.resize(tria->n_active_cells(), -2);
  master_slave_relationships_iterators.resize(tria->n_active_cells(), {});
  if (n_agglomerations > 0)
    std::fill(master_slave_relationships.begin(),
              master_slave_relationships.end(),
              -2); // identify all the tria with standard deal.II cells.

  polytope_cache.clear();
  bboxes.clear();

  // First, update the pointer
  cached_tria = std::make_unique<GridTools::Cache<dim, spacedim>>(
    cache_tria->get_triangulation(), cache_tria->get_mapping());

  connect_to_tria_signals();
  n_agglomerations = 0;
}



template <int dim, int spacedim>
void
AgglomerationHandler<dim, spacedim>::create_bounding_box(
  const AgglomerationContainer & polytope,
  const types::global_cell_index master_idx)
{
  Assert(n_agglomerations > 0,
         ExcMessage("No agglomeration has been performed."));
  Assert(dim > 1, ExcNotImplemented());

  std::vector<types::global_dof_index> dof_indices(euler_fe->dofs_per_cell);
  std::vector<Point<spacedim>>         pts; // store all the vertices
  for (const auto &cell : polytope)
    for (const auto i : cell->vertex_indices())
      pts.push_back(cell->vertex(i));

  bboxes.emplace_back(pts);

  typename DoFHandler<dim, spacedim>::cell_iterator polytope_dh(*polytope[0],
                                                                &euler_dh);
  polytope_dh->get_dof_indices(dof_indices);


  const auto &p0 =
    bboxes[master2polygon.at(master_idx)].get_boundary_points().first;
  const auto &p1 =
    bboxes[master2polygon.at(master_idx)].get_boundary_points().second;
  if constexpr (dim == 2)
    {
      euler_vector[dof_indices[0]] = p0[0];
      euler_vector[dof_indices[4]] = p0[1];
      // Lower right
      euler_vector[dof_indices[1]] = p1[0];
      euler_vector[dof_indices[5]] = p0[1];
      // Upper left
      euler_vector[dof_indices[2]] = p0[0];
      euler_vector[dof_indices[6]] = p1[1];
      // Upper right
      euler_vector[dof_indices[3]] = p1[0];
      euler_vector[dof_indices[7]] = p1[1];
    }
  else if constexpr (dim == 3)
    {
      // Lowers

      // left
      euler_vector[dof_indices[0]]  = p0[0];
      euler_vector[dof_indices[8]]  = p0[1];
      euler_vector[dof_indices[16]] = p0[2];

      // right
      euler_vector[dof_indices[1]]  = p1[0];
      euler_vector[dof_indices[9]]  = p0[1];
      euler_vector[dof_indices[17]] = p0[2];

      // left
      euler_vector[dof_indices[2]]  = p0[0];
      euler_vector[dof_indices[10]] = p1[1];
      euler_vector[dof_indices[18]] = p0[2];

      // right
      euler_vector[dof_indices[3]]  = p1[0];
      euler_vector[dof_indices[11]] = p1[1];
      euler_vector[dof_indices[19]] = p0[2];

      // Uppers

      // left
      euler_vector[dof_indices[4]]  = p0[0];
      euler_vector[dof_indices[12]] = p0[1];
      euler_vector[dof_indices[20]] = p1[2];

      // right
      euler_vector[dof_indices[5]]  = p1[0];
      euler_vector[dof_indices[13]] = p0[1];
      euler_vector[dof_indices[21]] = p1[2];

      // left
      euler_vector[dof_indices[6]]  = p0[0];
      euler_vector[dof_indices[14]] = p1[1];
      euler_vector[dof_indices[22]] = p1[2];

      // right
      euler_vector[dof_indices[7]]  = p1[0];
      euler_vector[dof_indices[15]] = p1[1];
      euler_vector[dof_indices[23]] = p1[2];
    }
  else
    {
      Assert(false, ExcNotImplemented());
    }
}



template <int dim, int spacedim>
void
AgglomerationHandler<dim, spacedim>::setup_connectivity_of_agglomeration()
{
  Assert(master_cells_container.size() > 0,
         ExcMessage("No agglomeration has been performed."));
  Assert(
    agglo_dh.n_dofs() > 0,
    ExcMessage(
      "The DoFHandler associated to the agglomeration has not been initialized. It's likely that you forgot to distribute the DoFs, i.e. you may want to check if a call to `initialize_hp_structure()` has been done."));

  number_of_agglomerated_faces.resize(master2polygon.size(), 0);
  for (const auto &cell : master_cells_container)
    {
      internal::AgglomerationHandlerImplementation<dim, spacedim>::
        setup_master_neighbor_connectivity(cell, *this);
    }

  if (Utilities::MPI::job_supports_mpi())
    {
      // communicate the number of faces
      recv_n_faces = Utilities::MPI::some_to_some(communicator, local_n_faces);

      // send information about boundaries and neighboring polytopes id
      recv_bdary_info =
        Utilities::MPI::some_to_some(communicator, local_bdary_info);

      recv_ghosted_master_id =
        Utilities::MPI::some_to_some(communicator, local_ghosted_master_id);
    }
}



template <int dim, int spacedim>
Quadrature<dim>
AgglomerationHandler<dim, spacedim>::agglomerated_quadrature(
  const typename AgglomerationHandler<dim, spacedim>::AgglomerationContainer
    &cells,
  const typename Triangulation<dim, spacedim>::active_cell_iterator
    &master_cell) const
{
  Assert(is_master_cell(master_cell),
         ExcMessage("This must be a master cell."));

  std::vector<Point<dim>> vec_pts;
  std::vector<double>     vec_JxWs;
  for (const auto &dummy_cell : cells)
    {
      no_values->reinit(dummy_cell);
      auto        q_points = no_values->get_quadrature_points(); // real qpoints
      const auto &JxWs     = no_values->get_JxW_values();

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
  const auto &            bbox =
    bboxes[master2polygon.at(master_cell->active_cell_index())];
  unit_points.reserve(vec_pts.size());

  for (unsigned int i = 0; i < vec_pts.size(); i++)
    unit_points[i] = bbox.real_to_unit(vec_pts[i]);

  return Quadrature<dim>(unit_points, vec_JxWs);
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
    if (cell->is_locally_owned())
      {
        if (is_master_cell(cell))
          cell->set_active_fe_index(CellAgglomerationType::master);
        else if (is_slave_cell(cell))
          cell->set_active_fe_index(CellAgglomerationType::slave); // slave cell
        else
          cell->set_active_fe_index(
            CellAgglomerationType::standard); // standard
      }

  agglo_dh.distribute_dofs(fe_collection);
  euler_mapping =
    std::make_unique<MappingFEField<dim, spacedim>>(euler_dh, euler_vector);
}



template <int dim, int spacedim>
const FEValues<dim, spacedim> &
AgglomerationHandler<dim, spacedim>::reinit(
  const AgglomerationIterator<dim, spacedim> &polytope) const
{
  Assert(euler_mapping,
         ExcMessage("The mapping describing the physical element stemming from "
                    "agglomeration has not been set up."));

  const auto &deal_cell = polytope->as_dof_handler_iterator(agglo_dh);

  // First check if the polytope is made just by a single cell. If so, use
  // classical FEValues
  if (polytope->n_background_cells() == 1)
    return standard_scratch->reinit(deal_cell);

  const auto &agglo_cells = polytope->get_agglomerate();

  Quadrature<dim> agglo_quad = agglomerated_quadrature(agglo_cells, deal_cell);

  const double bbox_measure =
    bboxes[master2polygon.at(deal_cell->active_cell_index())].volume();

  // Scale weights with the volume of the BBox. This way, the euler_mapping
  // defining the BBOx doesn't alter them.
  std::vector<double> scaled_weights;
  std::transform(agglo_quad.get_weights().begin(),
                 agglo_quad.get_weights().end(),
                 std::back_inserter(scaled_weights),
                 [&bbox_measure](const double w) { return w / bbox_measure; });

  Quadrature<dim> scaled_quad(agglo_quad.get_points(), scaled_weights);

  agglomerated_scratch = std::make_unique<ScratchData>(*euler_mapping,
                                                       fe_collection[0],
                                                       scaled_quad,
                                                       agglomeration_flags);
  return agglomerated_scratch->reinit(deal_cell);
}



template <int dim, int spacedim>
const FEValuesBase<dim, spacedim> &
AgglomerationHandler<dim, spacedim>::reinit_master(
  const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
  const unsigned int                                              face_index,
  std::unique_ptr<NonMatching::FEImmersedSurfaceValues<spacedim>>
    &agglo_isv_ptr) const
{
  return internal::AgglomerationHandlerImplementation<dim, spacedim>::
    reinit_master(cell, face_index, agglo_isv_ptr, *this);
}



template <int dim, int spacedim>
const FEValuesBase<dim, spacedim> &
AgglomerationHandler<dim, spacedim>::reinit(
  const AgglomerationIterator<dim, spacedim> &polytope,
  const unsigned int                          face_index) const
{
  Assert(euler_mapping,
         ExcMessage("The mapping describing the physical element stemming from "
                    "agglomeration has not been set up."));

  const auto &deal_cell = polytope->as_dof_handler_iterator(agglo_dh);
  Assert(is_master_cell(deal_cell), ExcMessage("This should be true."));

  return internal::AgglomerationHandlerImplementation<dim, spacedim>::
    reinit_master(deal_cell, face_index, agglomerated_isv_bdary, *this);
}



template <int dim, int spacedim>
std::pair<const FEValuesBase<dim, spacedim> &,
          const FEValuesBase<dim, spacedim> &>
AgglomerationHandler<dim, spacedim>::reinit_interface(
  const AgglomerationIterator<dim, spacedim> &polytope_in,
  const AgglomerationIterator<dim, spacedim> &neigh_polytope,
  const unsigned int                          local_in,
  const unsigned int                          local_neigh) const
{
  const auto &cell_in    = polytope_in->as_dof_handler_iterator(agglo_dh);
  const auto &neigh_cell = neigh_polytope->as_dof_handler_iterator(agglo_dh);
  Assert(is_master_cell(cell_in) && is_master_cell(neigh_cell),
         ExcMessage("Both cells should be masters."));

  const auto &fe_in =
    internal::AgglomerationHandlerImplementation<dim, spacedim>::reinit_master(
      cell_in, local_in, agglomerated_isv, *this);
  const auto &fe_out =
    internal::AgglomerationHandlerImplementation<dim, spacedim>::reinit_master(
      neigh_cell, local_neigh, agglomerated_isv_neigh, *this);
  std::pair<const FEValuesBase<dim, spacedim> &,
            const FEValuesBase<dim, spacedim> &>
    my_p(fe_in, fe_out);

  return my_p;
}



template <int dim, int spacedim>
template <typename Number>
void
AgglomerationHandler<dim, spacedim>::create_agglomeration_sparsity_pattern(
  SparsityPattern &               sparsity_pattern,
  const AffineConstraints<Number> constraints,
  const bool                      keep_constrained_dofs,
  const types::subdomain_id       subdomain_id)
{
  Assert(n_agglomerations > 0,
         ExcMessage("The agglomeration has not been set up correctly."));
  Assert(sparsity_pattern.empty(),
         ExcMessage(
           "The Sparsity pattern must be empty upon calling this function."));

  DynamicSparsityPattern dsp(agglo_dh.n_dofs(), agglo_dh.n_dofs());

  const unsigned int           n_components = fe_collection.n_components();
  Table<2, DoFTools::Coupling> cell_couplings{n_components, n_components};
  Table<2, DoFTools::Coupling> face_couplings{n_components, n_components};
  cell_couplings[0][0] = DoFTools::always;
  face_couplings[0][0] = DoFTools::always;
  DoFTools::make_flux_sparsity_pattern(agglo_dh,
                                       dsp,
                                       constraints,
                                       keep_constrained_dofs,
                                       cell_couplings,
                                       face_couplings,
                                       subdomain_id);


  const unsigned int dofs_per_cell = agglo_dh.get_fe(0).n_dofs_per_cell();
  std::vector<types::global_dof_index> current_dof_indices(dofs_per_cell);
  std::vector<types::global_dof_index> neighbor_dof_indices(dofs_per_cell);

  // Loop over all polytopes, find the neighbor and couple DoFs.

  for (const auto &polytope : polytope_iterators())
    {
      const unsigned int n_current_faces = polytope->n_faces();
      polytope->get_dof_indices(current_dof_indices);
      for (unsigned int f = 0; f < n_current_faces; ++f)
        {
          const auto &neigh_polytope = polytope->neighbor(f);
          if (neigh_polytope.state() == IteratorState::valid)
            {
              neigh_polytope->get_dof_indices(neighbor_dof_indices);
              constraints.add_entries_local_to_global(current_dof_indices,
                                                      neighbor_dof_indices,
                                                      dsp,
                                                      keep_constrained_dofs,
                                                      {});
            }
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
void
AgglomerationHandler<dim, spacedim>::setup_ghost_polytopes()
{
  for (const auto &cell : agglo_dh.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        std::cout << " cellindex = " << cell->active_cell_index() << std::endl;

        const types::global_cell_index master_idx =
          get_master_idx_of_cell(cell);

        std::vector<types::global_dof_index> global_dof_indices(
          fe->dofs_per_cell);

        const auto &master_cell =
          master_slave_relationships_iterators[master_idx];
        const auto master_cell_dh =
          master_cell->as_dof_handler_iterator(agglo_dh);
        master_cell_dh->get_dof_indices(global_dof_indices);

        // interior, locally owned, cell
        for (const auto &f : cell->face_indices())
          {
            if (!cell->at_boundary(f))
              {
                const auto &neighbor = cell->neighbor(f);
                if (neighbor->is_ghost())
                  {
                    // key of the map: the rank to which send the data
                    const types::subdomain_id neigh_rank =
                      neighbor->subdomain_id();
                    // send to neighboring process the ghosted (seen from the
                    // neighbor) cell index

                    // the present cell is a ghost cell for the neighboring
                    // subdomain
                    local_ghost_indices[neigh_rank].push_back(
                      cell->active_cell_index());

                    // send to neighboring process also the index of the master
                    // cell of the present cell, which identifies the polytope

                    // Notice: since cell is locally owned, I can call
                    // get_master_idx_of_cell(cell) without harm

                    local_indices_ghosted_master_cells[neigh_rank].push_back(
                      master_idx);

                    const CellId &master_cell_id = master_cell->id();

                    local_cell_ids_ghosted_master_cells[neigh_rank].push_back(
                      master_cell_id); // send CellId of master cell

                    // inform the "standard" neighbor about the neighboring id
                    // and its master cell
                    local_cell_ids_neigh_cell[neigh_rank].emplace(
                      cell->id(), master_cell_id);

                    // inform the neighboring rank that this master cell (hence
                    // polytope) has the following DoF indices
                    local_ghost_dofs[neigh_rank].emplace(master_cell_id,
                                                         global_dof_indices);
                  }
              }
          }
      }


  // bounding boxes: inform the neighboring rank that the
  // present agglomerate, identified by the master cell CellId
  // has a certaing bbox. TODO: communicate only when you have a ghosted
  // neighbor.
  for (const auto &master_cell : master_cells_container)
    {
      const auto &bbox =
        bboxes[master2polygon.at(master_cell->active_cell_index())];

      for (const types::subdomain_id neigh_rank : tria->ghost_owners())
        local_ghosted_bbox[neigh_rank].emplace(master_cell->id(), bbox);
    }

  // exchange ghost indices with neighboring ranks
  recv_ghost_indices =
    Utilities::MPI::some_to_some(communicator, local_ghost_indices);

  // exchange indices of master cells with neighboring ranks

  recv_indices_ghosted_master_cells =
    Utilities::MPI::some_to_some(communicator,
                                 local_indices_ghosted_master_cells);

  recv_cell_ids_ghosted_master_cells =
    Utilities::MPI::some_to_some(communicator,
                                 local_cell_ids_ghosted_master_cells);

  recv_cell_ids_neigh_cell =
    Utilities::MPI::some_to_some(communicator, local_cell_ids_neigh_cell);

  // Exchange with neighboring ranks the neighboring bounding boxes
  recv_ghosted_bbox =
    Utilities::MPI::some_to_some(communicator, local_ghosted_bbox);

  // Exchange with neighboring ranks the neighboring ghosted DoFs
  recv_ghost_dofs =
    Utilities::MPI::some_to_some(communicator, local_ghost_dofs);


  std::cout << "ON RANK " << Utilities::MPI::this_mpi_process(communicator)
            << std::endl;
  for (const auto &[sender_rank, ghosted_indices] : recv_ghost_indices)
    {
      std::cout << "From " << sender_rank << " we have "
                << ghosted_indices.size() << " ghosted indices"
                << " and "
                << recv_cell_ids_ghosted_master_cells.at(sender_rank).size()
                << " CellId(s)" << std::endl;
      for (const auto &idx : ghosted_indices)
        std::cout << idx << std::endl;

      std::cout << "With this master cells indices: " << std::endl;
      for (const auto &idx : recv_indices_ghosted_master_cells.at(sender_rank))
        std::cout << idx << std::endl;
    }
  std::cout << std::endl;
}



namespace dealii
{
  namespace internal
  {
    template <int dim, int spacedim>
    class AgglomerationHandlerImplementation
    {
    public:
      static const FEValuesBase<dim, spacedim> &
      reinit_master(
        const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
        const unsigned int face_index,
        std::unique_ptr<NonMatching::FEImmersedSurfaceValues<spacedim>>
          &                                        agglo_isv_ptr,
        const AgglomerationHandler<dim, spacedim> &handler)
      {
        Assert(handler.is_master_cell(cell),
               ExcMessage("This cell must be a master one."));

        const auto &neighbor = agglomerated_neighbor(cell, face_index, handler);

        const CellId polytope_in_id = cell->id();

        CellId polytope_out_id;
        if (neighbor.state() == IteratorState::valid)
          polytope_out_id = neighbor->id();
        else
          polytope_out_id = polytope_in_id; // on the boundary. Same id

        const auto &common_face = handler.polytope_cache.interface.at(
          {polytope_in_id, polytope_out_id});

        std::vector<Point<spacedim>> final_unit_q_points;
        std::vector<double>          final_weights;
        std::vector<Tensor<1, dim>>  final_normals;


        for (const auto &[deal_cell, local_face_idx] : common_face)
          {
            // std::cout << "deal cell = " << deal_cell->active_cell_index()
            //           << std::endl;
            // std::cout << "local face index = " << local_face_idx
            //           << std::endl;
            handler.no_face_values->reinit(deal_cell, local_face_idx);
            auto q_points = handler.no_face_values->get_quadrature_points();

            const auto &JxWs    = handler.no_face_values->get_JxW_values();
            const auto &normals = handler.no_face_values->get_normal_vectors();

            const auto &bbox =
              handler
                .bboxes[handler.master2polygon.at(cell->active_cell_index())];
            const double bbox_measure = bbox.volume();

            std::vector<Point<spacedim>> unit_q_points;
            std::transform(q_points.begin(),
                           q_points.end(),
                           std::back_inserter(unit_q_points),
                           [&](const Point<spacedim> &p) {
                             return bbox.real_to_unit(p);
                           });

            // Weights must be scaled with det(J)*|J^-t n| for each
            // quadrature point. Use the fact that we are using a BBox, so
            // the jacobi entries are the side_length in each direction.
            // Unfortunately, normal vectors will be scaled internally by
            // deal.II by using a covariant transformation. In order not to
            // change the normals, we multiply by the correct factors in
            // order to obtain the original normal after the call to
            // `reinit(cell)`.
            std::vector<double>         scale_factors(q_points.size());
            std::vector<double>         scaled_weights(q_points.size());
            std::vector<Tensor<1, dim>> scaled_normals(q_points.size());

            for (unsigned int q = 0; q < q_points.size(); ++q)
              {
                for (unsigned int direction = 0; direction < spacedim;
                     ++direction)
                  scaled_normals[q][direction] =
                    normals[q][direction] * (bbox.side_length(direction));

                scaled_weights[q] =
                  (JxWs[q] * scaled_normals[q].norm()) / bbox_measure;
                scaled_normals[q] /= scaled_normals[q].norm();
              }

            for (const auto &point : unit_q_points)
              final_unit_q_points.push_back(point);
            for (const auto &weight : scaled_weights)
              final_weights.push_back(weight);
            for (const auto &normal : scaled_normals)
              final_normals.push_back(normal);
          }

        NonMatching::ImmersedSurfaceQuadrature<dim, spacedim> surface_quad(
          final_unit_q_points, final_weights, final_normals);

        agglo_isv_ptr =
          std::make_unique<NonMatching::FEImmersedSurfaceValues<spacedim>>(
            *(handler.euler_mapping),
            *(handler.fe),
            surface_quad,
            handler.agglomeration_face_flags);

        agglo_isv_ptr->reinit(cell);

        return *agglo_isv_ptr;
      }



      /**
       * Given an agglomeration described by the master cell `master_cell`,
       * this function:
       * - enumerates the faces of the agglomeration
       * - stores who is the neighbor, the local face indices from outside and
       * inside*/
      static void
      setup_master_neighbor_connectivity(
        const typename Triangulation<dim, spacedim>::active_cell_iterator
          &                                        master_cell,
        const AgglomerationHandler<dim, spacedim> &handler)
      {
        Assert(
          handler
              .master_slave_relationships[master_cell->active_cell_index()] ==
            -1,
          ExcMessage("The present cell is not a master one."));

        const auto &agglomeration = handler.get_agglomerate(master_cell);
        const types::global_cell_index current_polytope_index =
          handler.master2polygon.at(master_cell->active_cell_index());

        CellId current_polytope_id = master_cell->id();


        std::set<types::global_cell_index> visited_polygonal_neighbors;

        // std::map<std::pair<CellId, unsigned int>, std::pair<bool, CellId>>
        //   bdary_info_current_poly;

        std::map<unsigned int, CellId> face_to_neigh_id;

        std::map<unsigned int, bool> is_face_at_boundary;

        // same as above, but with CellId
        std::set<CellId> visited_polygonal_neighbors_id;
        unsigned int     ghost_counter = 0;

        for (const auto &cell : agglomeration)
          {
            const types::global_cell_index cell_index =
              cell->active_cell_index();

            const CellId cell_id = cell->id();

            for (const auto f : cell->face_indices())
              {
                const auto &neighboring_cell = cell->neighbor(f);
                // Check if:
                const bool valid_neighbor =
                  neighboring_cell.state() == IteratorState::valid;

                if (valid_neighbor)
                  {
                    if (neighboring_cell->is_locally_owned() &&
                        !handler.are_cells_agglomerated(cell, neighboring_cell))
                      {
                        // - cell is not on the boundary,
                        // - it's not agglomerated with the neighbor. If so,
                        // it's a neighbor of the present agglomeration
                        std::cout << "interno (from rank) "
                                  << Utilities::MPI::this_mpi_process(
                                       handler.communicator)
                                  << std::endl;

                        std::cout
                          << "neighbor locally owned? " << std::boolalpha
                          << neighboring_cell->is_locally_owned() << std::endl;
                        // if (neighboring_cell->is_ghost())
                        //   handler.ghosted_indices.push_back(
                        //     neighboring_cell->active_cell_index());

                        // a new face of the agglomeration has been discovered.
                        handler.polygon_boundary[master_cell].push_back(
                          cell->face(f));

                        // global index of neighboring deal.II cell
                        const types::global_cell_index neighboring_cell_index =
                          neighboring_cell->active_cell_index();

                        // master cell for the neighboring polytope
                        const auto &master_of_neighbor =
                          handler.master_slave_relationships_iterators
                            [neighboring_cell_index];

                        const auto nof = cell->neighbor_of_neighbor(f);

                        if (handler.is_slave_cell(neighboring_cell))
                          {
                            // index of the neighboring polytope
                            const types::global_cell_index
                              neighbor_polytope_index =
                                handler.master2polygon.at(
                                  master_of_neighbor->active_cell_index());

                            CellId neighbor_polytope_id =
                              master_of_neighbor->id();

                            if (visited_polygonal_neighbors.find(
                                  neighbor_polytope_index) ==
                                std::end(visited_polygonal_neighbors))
                              {
                                // found a neighbor

                                const unsigned int n_face =
                                  handler.number_of_agglomerated_faces
                                    [current_polytope_index];

                                handler.polytope_cache.cell_face_at_boundary[{
                                  current_polytope_index, n_face}] = {
                                  false, master_of_neighbor};

                                is_face_at_boundary[n_face] = true;

                                ++handler.number_of_agglomerated_faces
                                    [current_polytope_index];

                                visited_polygonal_neighbors.insert(
                                  neighbor_polytope_index);
                              }


                            if (handler.polytope_cache.visited_cell_and_faces
                                  .find({cell_index, f}) ==
                                std::end(handler.polytope_cache
                                           .visited_cell_and_faces))
                              {
                                handler.polytope_cache
                                  .interface[{current_polytope_id,
                                              neighbor_polytope_id}]
                                  .emplace_back(cell, f);

                                handler.polytope_cache.visited_cell_and_faces
                                  .insert({cell_index, f});
                              }


                            if (handler.polytope_cache.visited_cell_and_faces
                                  .find({neighboring_cell_index, nof}) ==
                                std::end(handler.polytope_cache
                                           .visited_cell_and_faces))
                              {
                                handler.polytope_cache
                                  .interface[{neighbor_polytope_id,
                                              current_polytope_id}]
                                  .emplace_back(neighboring_cell, nof);

                                handler.polytope_cache.visited_cell_and_faces
                                  .insert({neighboring_cell_index, nof});
                              }
                          }
                        else
                          {
                            // neighboring cell is a master

                            // save the pair of neighboring cells
                            const types::global_cell_index
                              neighbor_polytope_index =
                                handler.master2polygon.at(
                                  neighboring_cell_index);

                            CellId neighbor_polytope_id =
                              neighboring_cell->id();

                            if (visited_polygonal_neighbors.find(
                                  neighbor_polytope_index) ==
                                std::end(visited_polygonal_neighbors))
                              {
                                // found a neighbor
                                const unsigned int n_face =
                                  handler.number_of_agglomerated_faces
                                    [current_polytope_index];


                                handler.polytope_cache.cell_face_at_boundary[{
                                  current_polytope_index, n_face}] = {
                                  false, neighboring_cell};

                                is_face_at_boundary[n_face] = true;

                                ++handler.number_of_agglomerated_faces
                                    [current_polytope_index];

                                visited_polygonal_neighbors.insert(
                                  neighbor_polytope_index);
                              }



                            if (handler.polytope_cache.visited_cell_and_faces
                                  .find({cell_index, f}) ==
                                std::end(handler.polytope_cache
                                           .visited_cell_and_faces))
                              {
                                handler.polytope_cache
                                  .interface[{current_polytope_id,
                                              neighbor_polytope_id}]
                                  .emplace_back(cell, f);

                                handler.polytope_cache.visited_cell_and_faces
                                  .insert({cell_index, f});
                              }

                            if (handler.polytope_cache.visited_cell_and_faces
                                  .find({neighboring_cell_index, nof}) ==
                                std::end(handler.polytope_cache
                                           .visited_cell_and_faces))
                              {
                                handler.polytope_cache
                                  .interface[{neighbor_polytope_id,
                                              current_polytope_id}]
                                  .emplace_back(neighboring_cell, nof);

                                handler.polytope_cache.visited_cell_and_faces
                                  .insert({neighboring_cell_index, nof});
                              }
                          }
                      }
                    else if (neighboring_cell->is_ghost())
                      {
                        std::cout
                          << "CELLA VISITATA: " << cell->active_cell_index()
                          << std::endl;
                        std::cout
                          << "Sul rank: "
                          << Utilities::MPI::this_mpi_process(
                               handler.communicator)
                          << "\n"
                          << "Dal rank: " << neighboring_cell->subdomain_id()
                          << std::endl;

                        const auto nof = cell->neighbor_of_neighbor(f);

                        // TODO: neighboring cell is a master,slave?

                        // retrieve from neighboring rank the master cell di

                        const auto &master_indices =
                          handler.recv_indices_ghosted_master_cells.at(
                            neighboring_cell->subdomain_id());


                        // retrieve from neighboring rank the master cell id
                        const auto &neighbor_ids =
                          handler.recv_cell_ids_ghosted_master_cells.at(
                            neighboring_cell->subdomain_id());

                        std::cout << "Ho il seguente numero di ids: "
                                  << neighbor_ids.size() << std::endl;

                        std::cout << "Ho i seguenti master indices: "
                                  << master_indices.size() << std::endl;
                        for (const auto &idx : master_indices)
                          std::cout << idx << std::endl;

                        // from neighboring rank,receive the association between
                        // standard cell ids and neighboring polytope.
                        // This tells to the current rank that the
                        // neighboring cell has the following CellId as master
                        // cell.
                        const auto &check_neigh_poly_ids =
                          handler.recv_cell_ids_neigh_cell.at(
                            neighboring_cell->subdomain_id());

                        const CellId neighboring_cell_id =
                          neighboring_cell->id();

                        const CellId &check_neigh_polytope_id =
                          check_neigh_poly_ids.at(neighboring_cell_id);

                        std::cout << "CellId del vicino trovato: "
                                  << check_neigh_polytope_id << std::endl;
                        std::cout << "rapido check" << std::boolalpha
                                  << (visited_polygonal_neighbors_id.find(
                                        check_neigh_polytope_id) ==
                                      std::end(visited_polygonal_neighbors_id))
                                  << std::endl;

                        const auto master_index = master_indices[ghost_counter];

                        if (visited_polygonal_neighbors_id.find(
                              check_neigh_polytope_id) ==
                            std::end(visited_polygonal_neighbors_id))
                          {
                            std::cout << "Sono entrato dunque da "
                                      << cell->active_cell_index()
                                      << "con faccia " << f << std::endl;
                            // found a neighbor


                            handler.polytope_cache.cell_face_at_boundary[{
                              current_polytope_index,
                              handler.number_of_agglomerated_faces
                                [current_polytope_index]}] = {false,
                                                              neighboring_cell};


                            // record the cell id of the neighboring polytope
                            handler.polytope_cache.ghosted_master_id[{
                              current_polytope_id,
                              handler.number_of_agglomerated_faces
                                [current_polytope_index]}] =
                              check_neigh_polytope_id;


                            const unsigned int n_face =
                              handler.number_of_agglomerated_faces
                                [current_polytope_index];

                            face_to_neigh_id[n_face] = check_neigh_polytope_id;

                            is_face_at_boundary[n_face] = false;

                            // std::pair<CellId, unsigned int> p{
                            //   current_polytope_id, n_face};

                            // // not on the dbdary, so give the neighboring
                            // // polytope id
                            // std::pair<bool, CellId> bdary_info{
                            //   false, check_neigh_polytope_id};

                            // bdary_info_current_poly[p] = bdary_info;


                            std::cout
                              << "Poly index: " << current_polytope_index
                              << std::endl;
                            std::cout << "Face index "
                                      << handler.number_of_agglomerated_faces
                                           [current_polytope_index]
                                      << std::endl;



                            // increment number of faces
                            ++handler.number_of_agglomerated_faces
                                [current_polytope_index];

                            visited_polygonal_neighbors_id.insert(
                              check_neigh_polytope_id);

                            std::cout << "Sul rank: "
                                      << Utilities::MPI::this_mpi_process(
                                           handler.communicator)
                                      << "\n"
                                      << "Dal rank: "
                                      << neighboring_cell->subdomain_id()
                                      << " aggiunto questo master index"
                                      << master_index << std::endl;
                            std::cout << "Cell index "
                                      << cell->active_cell_index() << std::endl;

                            // ghosted polytope has been found, increment ghost
                            // counter
                            ++ghost_counter;
                          }



                        if (handler.polytope_cache.visited_cell_and_faces_id
                              .find({cell_id, f}) ==
                            std::end(
                              handler.polytope_cache.visited_cell_and_faces_id))
                          {
                            handler.polytope_cache
                              .interface[{current_polytope_id,
                                          check_neigh_polytope_id}]
                              .emplace_back(cell, f);

                            std::cout << "AGGIUNTO ("
                                      << cell->active_cell_index() << ") TRA "
                                      << current_polytope_id << " e "
                                      << check_neigh_polytope_id << std::endl;

                            handler.polytope_cache.visited_cell_and_faces_id
                              .insert({cell_id, f});
                          }


                        if (handler.polytope_cache.visited_cell_and_faces_id
                              .find({neighboring_cell_id, nof}) ==
                            std::end(
                              handler.polytope_cache.visited_cell_and_faces_id))
                          {
                            handler.polytope_cache
                              .interface[{check_neigh_polytope_id,
                                          current_polytope_id}]
                              .emplace_back(neighboring_cell, nof);

                            handler.polytope_cache.visited_cell_and_faces_id
                              .insert({neighboring_cell_id, nof});
                          }
                      }
                  }
                else if (cell->face(f)->at_boundary())
                  {
                    // std::cout
                    //   << "sul bdary (from rank) "
                    //   <<
                    //   Utilities::MPI::this_mpi_process(handler.communicator)
                    //   << std::endl;

                    // Boundary face of a boundary cell.
                    // Note that the neighboring cell must be invalid.

                    handler.polygon_boundary[master_cell].push_back(
                      cell->face(f));



                    if (visited_polygonal_neighbors.find(
                          std::numeric_limits<unsigned int>::max()) ==
                        std::end(visited_polygonal_neighbors))
                      {
                        // boundary face. Notice that `neighboring_cell` is
                        // invalid here.
                        handler.polytope_cache.cell_face_at_boundary[{
                          current_polytope_index,
                          handler.number_of_agglomerated_faces
                            [current_polytope_index]}] = {true,
                                                          neighboring_cell};



                        const unsigned int n_face =
                          handler.number_of_agglomerated_faces
                            [current_polytope_index];

                        std::pair<CellId, unsigned int> p{current_polytope_id,
                                                          n_face};

                        std::pair<bool, CellId> bdary_info{true, CellId()};

                        // bdary_info_current_poly[p] = bdary_info;

                        is_face_at_boundary[n_face] = true;

                        // handler.local_ghosted_bdary_info[neigh_rank].emplace(
                        //   p, bdary_info);



                        ++handler.number_of_agglomerated_faces
                            [current_polytope_index];

                        visited_polygonal_neighbors.insert(
                          std::numeric_limits<unsigned int>::max());
                      }



                    if (handler.polytope_cache.visited_cell_and_faces.find(
                          {cell_index, f}) ==
                        std::end(handler.polytope_cache.visited_cell_and_faces))
                      {
                        handler.polytope_cache
                          .interface[{current_polytope_id, current_polytope_id}]
                          .emplace_back(cell, f);

                        handler.polytope_cache.visited_cell_and_faces.insert(
                          {cell_index, f});
                      }
                  }
              } // loop over faces
          }     // loop over all cells of agglomerate


        std::cout << "ghost_counter from rank " +
                       std::to_string(Utilities::MPI::this_mpi_process(
                         handler.communicator)) +
                       ": "
                  << ghost_counter << std::endl;


        // TODO
        // handler.debu(cached_tria->get_triangulation());

        // if constexpr (std::is_same_v<typename std::remove_reference<decltype(
        //                                *handler.tria)>::type,
        //                              typename dealii::parallel::distributed::
        //                                Triangulation<dim, spacedim>>)
        //   {
        // if (dynamic_cast<const dealii::parallel::
        //                    DistributedTriangulationBase<dim, spacedim> *>(
        //       &handler.cached_tria->get_triangulation()) != nullptr)
        //   {
        // std::cout << "OK " << std::endl;
        // Assert(false, ExcMessage("OKAY, REMOVE."));
        if (ghost_counter > 0)
          {
            const unsigned int n_faces_current_poly =
              handler.number_of_agglomerated_faces[current_polytope_index];

            // Communicate to neighboring ranks that current_polytope_id has
            // a number of faces equal to n_faces_current_poly faces:
            // current_polytope_id -> n_faces_current_poly
            for (const unsigned int neigh_rank :
                 handler.cached_tria->get_triangulation().ghost_owners())
              {
                handler.local_n_faces[neigh_rank].emplace(current_polytope_id,
                                                          n_faces_current_poly);

                handler.local_bdary_info[neigh_rank].emplace(
                  current_polytope_id, is_face_at_boundary);

                handler.local_ghosted_master_id[neigh_rank].emplace(
                  current_polytope_id, face_to_neigh_id);
              }
          }
      }



      static typename DoFHandler<dim, spacedim>::active_cell_iterator
      agglomerated_neighbor(
        const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
        const unsigned int                                              f,
        const AgglomerationHandler<dim, spacedim> &                     handler)
      {
        Assert(handler.is_master_cell(cell),
               ExcMessage("This cell must be a master one."));
        if (!handler.at_boundary(cell, f))
          {
            const auto &neigh =
              handler.polytope_cache.cell_face_at_boundary
                .at({handler.master2polygon.at(cell->active_cell_index()), f})
                .second;
            typename DoFHandler<dim, spacedim>::active_cell_iterator cell_dh(
              *neigh, &(handler.agglo_dh));
            return cell_dh;
          }
        else
          {
            return {};
          }
      }



      static unsigned int
      neighbor_of_agglomerated_neighbor(
        const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
        const unsigned int                                              f,
        const AgglomerationHandler<dim, spacedim> &                     handler)
      {
        Assert(!is_slave_cell(cell),
               ExcMessage("This cells is not supposed to be a slave."));
        // First, make sure it's not a boundary face.
        if (!at_boundary(cell, f))
          {
            const auto &agglo_neigh =
              agglomerated_neighbor(cell,
                                    f); // returns the neighboring master
            AssertThrow(agglo_neigh.state() == IteratorState::valid,
                        ExcInternalError());

            const unsigned int n_faces_agglomerated_neighbor =
              handler.number_of_agglomerated_faces[handler.master2polygon.at(
                agglo_neigh->active_cell_index())];

            // Loop over all cells of neighboring agglomerate
            for (unsigned int f_out = 0; f_out < n_faces_agglomerated_neighbor;
                 ++f_out)
              {
                // Check if same master cell
                if (agglomerated_neighbor(agglo_neigh, f_out).state() ==
                    IteratorState::valid)
                  if (agglomerated_neighbor(agglo_neigh, f_out)
                        ->active_cell_index() == cell->active_cell_index())
                    return f_out;
              }
            return numbers::invalid_unsigned_int;
          }
        else
          {
            // Face is at boundary
            return numbers::invalid_unsigned_int;
          }
      }
    };



  } // namespace internal
} // namespace dealii



template class AgglomerationHandler<1>;
template void
AgglomerationHandler<1>::create_agglomeration_sparsity_pattern(
  SparsityPattern &               sparsity_pattern,
  const AffineConstraints<double> constraints,
  const bool                      keep_constrained_dofs,
  const types::subdomain_id       subdomain_id);

template class AgglomerationHandler<2>;
template void
AgglomerationHandler<2>::create_agglomeration_sparsity_pattern(
  SparsityPattern &               sparsity_pattern,
  const AffineConstraints<double> constraints,
  const bool                      keep_constrained_dofs,
  const types::subdomain_id       subdomain_id);

template class AgglomerationHandler<3>;
template void
AgglomerationHandler<3>::create_agglomeration_sparsity_pattern(
  SparsityPattern &               sparsity_pattern,
  const AffineConstraints<double> constraints,
  const bool                      keep_constrained_dofs,
  const types::subdomain_id       subdomain_id);
