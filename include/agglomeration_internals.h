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
        const auto &neighbor = agglomerated_neighbor(cell, face_index, handler);

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



      static typename DoFHandler<dim, spacedim>::active_cell_iterator
      agglomerated_neighbor(
        const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
        const unsigned int                                              f,
        const AgglomerationHandler<dim, spacedim> &                     handler)
      {
        // assert?
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
              n_agglomerated_faces(agglo_neigh);
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
