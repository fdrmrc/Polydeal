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
  Assert(dim == 2 || dim == 3, ExcImpossibleInDim(1));
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

  // Get global index for each cell
  std::vector<types::global_cell_index> global_indices;
  for (const auto &cell : cells)
    global_indices.push_back(cell->active_cell_index());

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
AgglomerationHandler<dim, spacedim>::n_faces(
  const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell) const
{
  Assert(!is_slave_cell(cell), ExcMessage("You cannot pass a slave cell."));
  if (is_standard_cell(cell))
    {
      return cell->n_faces();
    }
  else
    {
      const auto & agglomeration = get_agglomerate(cell);
      unsigned int n_neighbors   = 0;
      for (const auto &cell : agglomeration)
        {
          n_neighbors += n_agglomerated_faces_per_cell(cell);
        }
      return n_neighbors;
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
typename DoFHandler<dim, spacedim>::active_cell_iterator
AgglomerationHandler<dim, spacedim>::agglomerated_neighbor(
  const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
  const unsigned int                                              f) const
{
  Assert(!is_slave_cell(cell),
         ExcMessage("This cells is not supposed to be a slave."));
  if (!at_boundary(cell, f))
    {
      if (is_standard_cell(cell) && is_standard_cell(cell->neighbor(f)))
        {
          return cell->neighbor(f);
        }
      else if (is_master_cell(cell))
        {
          // const auto &neigh = std::get<1>(master_neighbors[{cell, f}]);
          // typename DoFHandler<dim, spacedim>::active_cell_iterator cell_dh(
          //   *neigh, &agglo_dh);
          // return cell_dh;

          // const auto &neigh = std::get<2>(info_cells[{cell, f}][0]);
          const auto &neigh =
            polytope_cache.cell_face_at_boundary
              .at({master2polygon.at(cell->active_cell_index()), f})
              .second;
          typename DoFHandler<dim, spacedim>::active_cell_iterator cell_dh(
            *neigh, &agglo_dh);
          return cell_dh;
        }
      else
        {
          // If I fall here, I want to find the neighbor for a standard cell
          // adjacent to an agglomeration
          const auto &master_cell =
            master_slave_relationships_iterators[cell->neighbor(f)
                                                   ->active_cell_index()];
          typename DoFHandler<dim, spacedim>::active_cell_iterator cell_dh(
            *master_cell, &agglo_dh);
          return cell_dh;
        }
    }
  else
    {
      return {};
    }
}



template <int dim, int spacedim>
unsigned int
AgglomerationHandler<dim, spacedim>::neighbor_of_agglomerated_neighbor(
  const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
  const unsigned int                                              f) const
{
  Assert(!is_slave_cell(cell),
         ExcMessage("This cells is not supposed to be a slave."));
  // First, make sure it's not a boundary face.
  if (!at_boundary(cell, f))
    {
      if (is_standard_cell(cell) && is_standard_cell(cell->neighbor(f)))
        {
          return cell->neighbor_of_neighbor(f);
        }
      else if (is_master_cell(cell) &&
               is_master_cell(agglomerated_neighbor(cell, f)))
        {
          // const auto &current_agglo_info = master_neighbors[{cell, f}];
          // const auto &local_f_idx        = std::get<0>(current_agglo_info);
          // const auto &current_cell       = std::get<3>(current_agglo_info);

          // const auto &agglo_neigh =
          //   agglomerated_neighbor(cell,
          //                         f); // returns the neighboring master
          // const unsigned int n_faces_agglomerated_neighbor =
          //   n_faces(agglo_neigh);
          // // Loop over all cells of neighboring agglomerate
          // for (unsigned int f_out = 0; f_out <
          // n_faces_agglomerated_neighbor;
          //      ++f_out)
          //   {
          //     if (agglomerated_neighbor(agglo_neigh, f_out).state() ==
          //           IteratorState::valid &&
          //         current_cell->neighbor(local_f_idx).state() ==
          //           IteratorState::valid &&
          //         current_cell.state() == IteratorState::valid)
          //       {
          //         const auto &neighboring_agglo_info =
          //           master_neighbors[{agglo_neigh, f_out}];
          //         const auto &local_f_out_idx =
          //           std::get<0>(neighboring_agglo_info);
          //         const auto &current_cell_out =
          //           std::get<3>(neighboring_agglo_info);
          //         const auto &other_standard =
          //           std::get<1>(neighboring_agglo_info);

          //         // Here, an extra condition is needed because there can
          //         be
          //         // more than one face index that returns the same
          //         neighbor
          //         // if you simply check who is f' s.t.
          //         // cell->neigh(f)->neigh(f') == cell. Hence, an extra
          //         // condition must be added.

          //         if (other_standard.state() == IteratorState::valid &&
          //             agglomerated_neighbor(agglo_neigh, f_out)
          //                 ->active_cell_index() ==
          //               cell->active_cell_index() &&
          //             current_cell->active_cell_index() ==
          //               current_cell_out->neighbor(local_f_out_idx)
          //                 ->active_cell_index() &&
          //             current_cell->neighbor(local_f_idx) ==
          //             current_cell_out)
          //           return f_out;
          //       }
          //   }
          // Assert(false, ExcInternalError());
          // return {}; // just to suppress warnings


          // const auto &current_agglo_info = info_cells[{cell, f}][0];
          // const auto &current_cell       = std::get<0>(current_agglo_info);
          // const auto &local_f_idx        = std::get<1>(current_agglo_info);

          const auto &agglo_neigh =
            agglomerated_neighbor(cell,
                                  f); // returns the neighboring master
          AssertThrow(agglo_neigh.state() == IteratorState::valid,
                      ExcInternalError());
          const unsigned int n_faces_agglomerated_neighbor =
            n_agglomerated_faces(agglo_neigh);
          // Loop over all cells of neighboring agglomerate
          for (unsigned int f_out = 0; f_out < n_faces_agglomerated_neighbor;
               ++f_out)
            {
              // if (agglomerated_neighbor(agglo_neigh, f_out).state() ==
              //       IteratorState::valid &&
              //     current_cell->neighbor(local_f_idx).state() ==
              //       IteratorState::valid &&
              //     current_cell.state() == IteratorState::valid)
              //   {
              // const auto &neighbor_info = info_cells[{agglo_neigh,
              // f_out}];

              // Check if same master cell
              if (agglomerated_neighbor(agglo_neigh, f_out).state() ==
                  IteratorState::valid)
                if (agglomerated_neighbor(agglo_neigh, f_out)
                      ->active_cell_index() == cell->active_cell_index())
                  return f_out;

              // for (const auto &[other_deal,
              //                   local_f_out_idx,
              //                   neigh_out,
              //                   dummy] : neighbor_info)
              //   {
              //     if (other_deal->neighbor(local_f_out_idx) ==
              //           current_cell &&
              //         other_deal.state() == IteratorState::valid)
              //       return f_out;
              //   }
              // Here, an extra condition is needed because there can be
              // more than one face index that returns the same neighbor
              // if you simply check who is f' s.t.
              // cell->neigh(f)->neigh(f') == cell. Hence, an extra
              // condition must be added.

              // if (other_deal.state() == IteratorState::valid &&
              //     agglomerated_neighbor(agglo_neigh, f_out)
              //         ->active_cell_index() ==
              //       cell->active_cell_index() &&
              //     current_cell->active_cell_index() ==
              //       current_cell_out->neighbor(local_f_out_idx)
              //         ->active_cell_index() &&
              //     current_cell->neighbor(local_f_idx) ==
              //     current_cell_out)
              //   return f_out;
              // }
            }
          return numbers::invalid_unsigned_int;
        }
      else if (is_master_cell(cell) &&
               is_standard_cell(agglomerated_neighbor(cell, f)))
        {
          return std::get<2>(master_neighbors[{cell, f}]);
        }
      else
        {
          // If I fall here, I want to find the neighbor of neighbor for a
          // standard cell adjacent to an agglomeration.
          const auto &master_cell = agglomerated_neighbor(
            cell, f); // this is the master of the neighboring agglomeration

          return shared_face_agglomeration_idx[{master_cell, cell, f}];
        }
    }
  else
    {
      // Face is at boundary
      return numbers::invalid_unsigned_int;
    }
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

  master_neighbors.clear();
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
    {
      typename DoFHandler<dim, spacedim>::cell_iterator cell_dh(*cell,
                                                                &euler_dh);
      cell_dh->get_dof_indices(dof_indices);
      for (const auto i : cell->vertex_indices())
        pts.push_back(cell->vertex(i));
    }

  bboxes.emplace_back(pts);

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

      Quadrature<dim> agglo_quad = agglomerated_quadrature(agglo_cells, cell);

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
const FEValues<dim, spacedim> &
AgglomerationHandler<dim, spacedim>::reinit(
  const AgglomerationIterator<dim, spacedim> &polytope) const
{
  Assert(euler_mapping,
         ExcMessage("The mapping describing the physical element stemming from "
                    "agglomeration has not been set up."));

  const auto &deal_cell = polytope->as_dof_handler_iterator(agglo_dh);

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
  const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
  const unsigned int face_index) const
{
  Assert(euler_mapping,
         ExcMessage("The mapping describing the physical element stemming from "
                    "agglomeration has not been set up."));

  if ((is_standard_cell(cell) && at_boundary(cell, face_index)) ||
      (is_standard_cell(cell) &&
       is_master_cell(agglomerated_neighbor(cell, face_index))))
    {
      standard_scratch_face_any =
        std::make_unique<ScratchData>(*mapping,
                                      fe_collection[2],
                                      agglomeration_quad,
                                      agglomeration_flags,
                                      agglomeration_face_quad,
                                      agglomeration_face_flags);
      return standard_scratch_face_any->reinit(cell, face_index);
    }
  else
    {
      Assert(is_master_cell(cell), ExcMessage("This should be true."));
      return internal::AgglomerationHandlerImplementation<dim, spacedim>::
        reinit_master(cell, face_index, agglomerated_isv_bdary, *this);
    }
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
  // both are masters. That means you want to compute the jumps or
  // averages between a face shared by two neighboring agglomerations.

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

      const auto &fe_in =
        internal::AgglomerationHandlerImplementation<dim, spacedim>::
          reinit_master(cell_in, local_in, agglomerated_isv, *this);
      const auto &fe_out =
        internal::AgglomerationHandlerImplementation<dim, spacedim>::
          reinit_master(neigh_cell, local_neigh, agglomerated_isv_neigh, *this);
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
        // const auto &info_neighbors =
        //   handler.master_neighbors[{cell, face_index}];
        // const auto &local_face_idx = std::get<0>(info_neighbors);
        // const auto &deal_cell      = std::get<3>(info_neighbors);

        // const auto  cells_and_faces = handler.info_cells.at({cell,
        // face_index});
        const auto &neighbor = handler.agglomerated_neighbor(cell, face_index);

        // std::cout << "reinit master " << cell->active_cell_index() <<
        // std::endl; std::cout << "agglomerate face " << face_index <<
        // std::endl;

        types::global_cell_index polytope_in =
          handler.master2polygon.at(cell->active_cell_index());
        types::global_cell_index polytope_out;

        if (neighbor.state() == IteratorState::valid)
          polytope_out =
            handler.master2polygon.at(neighbor->active_cell_index());
        else
          polytope_out = std::numeric_limits<unsigned int>::max();

        const auto common_face =
          handler.polytope_cache.interface.at({polytope_in, polytope_out});

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

        unsigned int n_agglo_faces = 0;

        std::set<types::global_cell_index> visited_polygonal_neighbors;


        for (const auto &cell : agglomeration)
          {
            const types::global_cell_index cell_index =
              cell->active_cell_index();
            for (const auto f : cell->face_indices())
              {
                const auto &neighboring_cell = cell->neighbor(f);

                // Check if:
                // - cell is not on the boundary,
                // - it's not agglomerated with the neighbor. If so, it's a
                // neighbor of the present agglomeration
                if (neighboring_cell.state() == IteratorState::valid &&
                    !handler.are_cells_agglomerated(cell, neighboring_cell))
                  {
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



                    const auto &cell_and_face =
                      typename AgglomerationHandler<dim, spacedim>::CellAndFace(
                        master_cell, n_agglo_faces);                //(agglo,f)
                    const auto nof = cell->neighbor_of_neighbor(f); // loc(f')

                    if (handler.is_slave_cell(neighboring_cell))
                      {
                        // index of the neighboring polytope
                        const types::global_cell_index neighbor_polytope_index =
                          handler.master2polygon.at(
                            master_of_neighbor->active_cell_index());

                        handler.master_neighbors.emplace(
                          cell_and_face,
                          std::make_tuple(f, master_of_neighbor, nof, cell));


                        if (visited_polygonal_neighbors.find(
                              neighbor_polytope_index) ==
                            std::end(visited_polygonal_neighbors))
                          {
                            // found a neighbor
                            handler.polytope_cache.cell_face_at_boundary[{
                              current_polytope_index,
                              handler.number_of_agglomerated_faces
                                [current_polytope_index]}] = {
                              false, master_of_neighbor};

                            ++handler.number_of_agglomerated_faces
                                [current_polytope_index];

                            visited_polygonal_neighbors.insert(
                              neighbor_polytope_index);
                          }


                        if (handler.polytope_cache.visited_cell_and_faces.find(
                              {cell_index, f}) ==
                            std::end(
                              handler.polytope_cache.visited_cell_and_faces))
                          {
                            handler.polytope_cache
                              .interface[{current_polytope_index,
                                          neighbor_polytope_index}]
                              .emplace_back(cell, f);

                            handler.polytope_cache.visited_cell_and_faces
                              .insert({cell_index, f});
                          }


                        if (handler.polytope_cache.visited_cell_and_faces.find(
                              {neighboring_cell_index, nof}) ==
                            std::end(
                              handler.polytope_cache.visited_cell_and_faces))
                          {
                            handler.polytope_cache
                              .interface[{neighbor_polytope_index,
                                          current_polytope_index}]
                              .emplace_back(neighboring_cell, nof);

                            handler.polytope_cache.visited_cell_and_faces
                              .insert({neighboring_cell_index, nof});
                          }
                      }
                    else
                      {
                        handler.master_neighbors.emplace(
                          cell_and_face,
                          std::make_tuple(
                            f,
                            neighboring_cell,
                            nof,
                            cell)); //(agglo,f)
                                    //->(loc(f),other_deal_cell,loc(f'),dealcell)

                        // save the pair of neighboring cells
                        if (handler.is_master_cell(neighboring_cell))
                          {
                            const types::global_cell_index
                              neighbor_polytope_index =
                                handler.master2polygon.at(
                                  neighboring_cell_index);

                            if (visited_polygonal_neighbors.find(
                                  neighbor_polytope_index) ==
                                std::end(visited_polygonal_neighbors))
                              {
                                // found a neighbor

                                handler.polytope_cache.cell_face_at_boundary[{
                                  current_polytope_index,
                                  handler.number_of_agglomerated_faces
                                    [current_polytope_index]}] = {
                                  false, neighboring_cell};

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
                                  .interface[{current_polytope_index,
                                              neighbor_polytope_index}]
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
                                  .interface[{neighbor_polytope_index,
                                              current_polytope_index}]
                                  .emplace_back(neighboring_cell, nof);

                                handler.polytope_cache.visited_cell_and_faces
                                  .insert({neighboring_cell_index, nof});
                              }
                          }
                      }



                    // Now, link the index of the agglomerated face with the
                    // master and the neighboring cell.
                    handler.shared_face_agglomeration_idx.emplace(
                      typename AgglomerationHandler<dim, spacedim>::
                        MasterAndNeighborAndFace(master_cell,
                                                 neighboring_cell,
                                                 nof),
                      n_agglo_faces);
                    ++n_agglo_faces;
                  }
                else if (cell->face(f)->at_boundary())
                  {
                    // Boundary face of a boundary cell.
                    // Note that the neighboring cell must be invalid.
                    const auto &cell_and_face =
                      typename AgglomerationHandler<dim, spacedim>::CellAndFace(
                        master_cell, n_agglo_faces);

                    handler.polygon_boundary[master_cell].push_back(
                      cell->face(f));

                    handler.master_neighbors.emplace(
                      cell_and_face,
                      std::make_tuple(f,
                                      neighboring_cell,
                                      std::numeric_limits<unsigned int>::max(),
                                      cell)); // TODO: check what the last
                                              // element should be...
                    ++n_agglo_faces;



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
                          .interface[{current_polytope_index,
                                      std::numeric_limits<unsigned int>::max()}]
                          .emplace_back(cell, f);

                        handler.polytope_cache.visited_cell_and_faces.insert(
                          {cell_index, f});
                      }
                  }
              }
          } // loop over all cells of agglomerate
      }
    };



  } // namespace internal
} // namespace dealii



template class AgglomerationHandler<1>;
template class AgglomerationHandler<2>;
template class AgglomerationHandler<3>;