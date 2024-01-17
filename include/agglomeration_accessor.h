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


#ifndef agglomeration_accessor_h
#define agglomeration_accessor_h

#include <deal.II/base/config.h>

#include <deal.II/base/iterator_range.h>

#include <deal.II/grid/filtered_iterator.h>

#include <vector>

using namespace dealii;


// Forward declarations
#ifndef DOXYGEN
template <int, int>
class AgglomerationHandler;
template <int, int>
class AgglomerationIterator;
#endif


/**
 * Accessor class used by AgglomerationIterator to access agglomeration data.
 */
template <int dim, int spacedim = dim>
class AgglomerationAccessor
{
public:
  /**
   * Type for storing the polygons in an agglomerate.
   */
  using agglomeration_container =
    std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>;


  /**
   * Get the DoFs indices associated to the agglomerate.
   */
  void
  get_dof_indices(std::vector<types::global_dof_index> &) const;

  /**
   * Return, for a cell, the number of faces. In case the cell is a standard
   * cell, then the number of faces is the classical one. If it's a master cell,
   * then it returns the number of faces of the agglomeration identified by the
   * master cell itself.
   */
  unsigned int
  n_faces() const;

  /**
   * Return the agglomerate which shares face f.
   */
  const AgglomerationIterator<dim, spacedim>
  neighbor(const unsigned int f) const;

  /**
   * Return the present index (seen from the neighboring agglomerate) of the
   * present face f.
   */
  unsigned int
  neighbor_of_agglomerated_neighbor(const unsigned int f) const;

  /**
   *
   * This function generalizes the behaviour of cell->face(f)->at_boundary()
   * in the case where f is an index out of the range [0,..., n_faces).
   * In practice, if you call this function with a standard deal.II cell, you
   * have precisely the same result as calling cell->face(f)->at_boundary().
   * Otherwise, if the cell is a master one, you have a boolean returning true
   * is that face for the agglomeration is on the boundary or not.
   */
  bool
  at_boundary(const unsigned int f) const;

  /**
   * Return a vector of face iterators describing the boundary of agglomerate.
   */
  const std::vector<typename Triangulation<dim>::active_face_iterator> &
  polytope_boundary() const;

  /**
   *
   * Return the volume of a polytope.
   */
  double
  volume() const;

  /**
   * Return the diameter of the present polytopal element.
   */
  double
  diameter() const;

  /**
   * Returns the deal.II cells that build the agglomerate.
   */
  agglomeration_container
  get_agglomerate() const;

  /**
   * Return the index of the present polygon.
   */
  types::global_cell_index
  index() const;

  /**
   * Returns an active cell iterator for the dof_handler, matching the polytope
   * referenced by the input iterator. The type of the returned object is a
   * DoFHandler::active_cell_iterator which can be used to initialize
   * FiniteElement data.
   */
  typename DoFHandler<dim, spacedim>::active_cell_iterator
  as_dof_handler_iterator(const DoFHandler<dim, spacedim> &dof_handler) const;


private:
  /**
   * Private default constructor. This is not supposed to be used and hence will
   * throw.
   */
  AgglomerationAccessor();

  /**
   * Private constructor for an agglomerate. This is meant to be invoked by
   * the AgglomerationIterator class. It takes as input the master cell of the
   * agglomerate and a pointer to the handler.
   */
  AgglomerationAccessor(
    const typename Triangulation<dim, spacedim>::active_cell_iterator
      &                                        master_cell,
    const AgglomerationHandler<dim, spacedim> *ah);

  /**
   * Default destructor.
   */
  ~AgglomerationAccessor() = default;


  /**
   * The unique deal.II cell associated to the present polytope.
   */
  typename Triangulation<dim, spacedim>::active_cell_iterator master_cell;

  /**
   * The index of the present polytope.
   */
  types::global_cell_index present_index;

  /**
   * A pointer to the Handler.
   */
  AgglomerationHandler<dim, spacedim> *handler;

  /**
   * Return the number of agglomerated faces.
   */
  unsigned int
  n_agglomerated_faces_per_cell(
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
    const;

  /**
   * Comparison operator for Accessor. Two accessors are equal if they refer to
   * the same polygonal element.
   */
  bool
  operator==(const AgglomerationAccessor<dim, spacedim> &other) const;

  /**
   * Compare for inequality.
   */
  bool
  operator!=(const AgglomerationAccessor<dim, spacedim> &other) const;

  /**
   * Move to the next cell in the polygonal mesh.
   */
  void
  next();

  /**
   * Move to the previous cell in the polygonal mesh.
   */
  void
  prev();


  template <int, int>
  friend class AgglomerationIterator;
};



template <int dim, int spacedim>
unsigned int
AgglomerationAccessor<dim, spacedim>::n_agglomerated_faces_per_cell(
  const typename Triangulation<dim, spacedim>::active_cell_iterator &cell) const
{
  unsigned int n_neighbors = 0;
  for (const auto &f : cell->face_indices())
    {
      const auto &neighboring_cell = cell->neighbor(f);
      if ((cell->face(f)->at_boundary()) ||
          (neighboring_cell->is_active() &&
           !handler->are_cells_agglomerated(cell, neighboring_cell)))
        {
          ++n_neighbors;
        }
    }
  return n_neighbors;
}


template <int dim, int spacedim>
unsigned int
AgglomerationAccessor<dim, spacedim>::n_faces() const
{
  Assert(!handler->is_slave_cell(master_cell),
         ExcMessage("You cannot pass a slave cell."));
  if (handler->is_standard_cell(master_cell))
    {
      return master_cell->n_faces();
    }
  else
    {
      const auto & agglomeration = get_agglomerate();
      unsigned int n_neighbors   = 0;
      for (const auto &cell : agglomeration)
        {
          n_neighbors += n_agglomerated_faces_per_cell(cell);
        }
      return n_neighbors;
    }
}



template <int dim, int spacedim>
const AgglomerationIterator<dim, spacedim>
AgglomerationAccessor<dim, spacedim>::neighbor(const unsigned int f) const
{
  // Assert(!handler->is_slave_cell(cell),
  //        ExcMessage("This cells is not supposed to be a slave."));
  if (!at_boundary(f))
    {
      if (handler->is_standard_cell(master_cell) &&
          handler->is_standard_cell(master_cell->neighbor(f)))
        {
          // return master_cell->neighbor(f);
          return {master_cell->neighbor(f), handler};
        }
      else if (handler->is_master_cell(master_cell))
        {
          const auto &neigh =
            std::get<1>(handler->master_neighbors[{master_cell, f}]);
          // typename DoFHandler<dim, spacedim>::active_cell_iterator cell_dh(
          //   *neigh, &(handler->agglo_dh));
          // return cell_dh;
          return {neigh, handler};
        }
      else
        {
          // If I fall here, I want to find the neighbor for a standard cell
          // adjacent to an agglomeration
          const auto &cell = handler->master_slave_relationships_iterators
                               [master_cell->neighbor(f)->active_cell_index()];
          // typename DoFHandler<dim, spacedim>::active_cell_iterator cell_dh(
          //   *master_cell, &(handler->agglo_dh));
          // return cell_dh;
          return {cell, handler};
        }
    }
  else
    {
      return {};
    }
}



template <int dim, int spacedim>
unsigned int
AgglomerationAccessor<dim, spacedim>::neighbor_of_agglomerated_neighbor(
  const unsigned int f) const
{
  // Assert(!is_slave_cell(cell),
  //        ExcMessage("This cells is not supposed to be a slave."));
  // First, make sure it's not a boundary face.
  if (!at_boundary(f))
    {
      if (handler->is_standard_cell(master_cell) &&
          handler->is_standard_cell(master_cell->neighbor(f)))
        {
          return master_cell->neighbor_of_neighbor(f);
        }
      else if (handler->is_master_cell(master_cell) &&
               handler->is_master_cell(neighbor(f).master_cell()))
        {
          const auto &current_agglo_info =
            handler->master_neighbors[{master_cell, f}];
          const auto &local_f_idx  = std::get<0>(current_agglo_info);
          const auto &current_cell = std::get<3>(current_agglo_info);

          const auto &agglo_neigh =
            neighbor(f); // returns the neighboring master
          const unsigned int n_faces_agglomerated_neighbor =
            agglo_neigh->n_faces();

          // Loop over all cells of neighboring agglomerate
          for (unsigned int f_out = 0; f_out < n_faces_agglomerated_neighbor;
               ++f_out)
            {
              if (agglo_neigh->neighbor(f_out).state() ==
                    IteratorState::valid &&
                  current_cell->neighbor(local_f_idx).state() ==
                    IteratorState::valid &&
                  current_cell.state() == IteratorState::valid)
                {
                  const auto &neighboring_agglo_info =
                    handler
                      ->master_neighbors[{agglo_neigh.master_cell(), f_out}];
                  const auto &local_f_out_idx =
                    std::get<0>(neighboring_agglo_info);
                  const auto &current_cell_out =
                    std::get<3>(neighboring_agglo_info);
                  const auto &other_standard =
                    std::get<1>(neighboring_agglo_info);

                  // Here, an extra condition is needed because there can be
                  // more than one face index that returns the same neighbor
                  // if you simply check who is f' s.t.
                  // cell->neigh(f)->neigh(f') == cell. Hence, an extra
                  // condition must be added.

                  if (other_standard.state() == IteratorState::valid &&
                      agglo_neigh->neighbor(f_out)->index() == index() &&
                      current_cell->active_cell_index() ==
                        current_cell_out->neighbor(local_f_out_idx)
                          ->active_cell_index() &&
                      current_cell->neighbor(local_f_idx) == current_cell_out)
                    return f_out;
                }
            }
          Assert(false, ExcInternalError());
          return {}; // just to suppress warnings
        }
      else if (handler->is_master_cell(master_cell) &&
               handler->is_standard_cell(neighbor(f).master_cell()))
        {
          return std::get<2>(handler->master_neighbors[{master_cell, f}]);
        }
      else
        {
          // TODO : check
          // If I fall here, I want to find the neighbor of neighbor for a
          // standard cell adjacent to an agglomeration.
          const auto &master_neigh =
            neighbor(f); // this is the master of the neighboring agglomeration
          return handler->shared_face_agglomeration_idx[{
            master_neigh.master_cell(), master_cell, f}];
        }
    }
  else
    {
      // Face is at boundary
      return numbers::invalid_unsigned_int;
    }
}

// ------------------------------ inline functions -------------------------

template <int dim, int spacedim>
inline AgglomerationAccessor<dim, spacedim>::AgglomerationAccessor()
{}



template <int dim, int spacedim>
inline AgglomerationAccessor<dim, spacedim>::AgglomerationAccessor(
  const typename Triangulation<dim, spacedim>::active_cell_iterator &cell,
  const AgglomerationHandler<dim, spacedim> *                        ah)
{
  handler = const_cast<AgglomerationHandler<dim, spacedim> *>(ah);
  if (&(*handler->master_cells_container.end()) == std::addressof(cell))
    {
      present_index = handler->master_cells_container.size();
      master_cell   = handler->master_cells_container[present_index];
    }
  else
    {
      present_index = handler->master2polygon.at(cell->active_cell_index());
      master_cell   = cell;
    }
}



template <int dim, int spacedim>
inline void
AgglomerationAccessor<dim, spacedim>::get_dof_indices(
  std::vector<types::global_dof_index> &dof_indices) const
{
  Assert(dof_indices.size() > 0,
         ExcMessage(
           "The vector of DoFs indices must be already properly resized."));
  // Forward the call to the master cell
  typename DoFHandler<dim, spacedim>::cell_iterator master_cell_dh(
    *master_cell, &(handler->agglo_dh));
  master_cell_dh->get_dof_indices(dof_indices);
}



template <int dim, int spacedim>
inline typename AgglomerationAccessor<dim, spacedim>::agglomeration_container
AgglomerationAccessor<dim, spacedim>::get_agglomerate() const
{
  auto agglomeration =
    handler->get_slaves_of_idx(master_cell->active_cell_index());
  agglomeration.push_back(master_cell);
  return agglomeration;
}



template <int dim, int spacedim>
bool
AgglomerationAccessor<dim, spacedim>::at_boundary(const unsigned int f) const
{
  Assert(!handler->is_slave_cell(master_cell),
         ExcMessage("This function should not be called for a slave cell."));
  if (handler->is_standard_cell(master_cell))
    return master_cell->face(f)->at_boundary();
  else
    return std::get<2>(handler->master_neighbors[{master_cell, f}]) ==
           std::numeric_limits<unsigned int>::max();
}



template <int dim, int spacedim>
inline const std::vector<typename Triangulation<dim>::active_face_iterator> &
AgglomerationAccessor<dim, spacedim>::polytope_boundary() const
{
  return handler->polygon_boundary[master_cell];
}



template <int dim, int spacedim>
inline double
AgglomerationAccessor<dim, spacedim>::diameter() const
{
  Assert(!handler->is_slave_cell(master_cell),
         ExcMessage("The present function cannot be called for slave cells."));

  if (handler->is_master_cell(master_cell))
    {
      // Get the bounding box associated with the master cell
      const auto &bdary_pts =
        handler->bboxes[present_index].get_boundary_points();
      return (bdary_pts.second - bdary_pts.first).norm();
    }
  else
    {
      // Standard deal.II way to get the measure of a cell.
      return master_cell->diameter();
    }
}



template <int dim, int spacedim>
inline double
AgglomerationAccessor<dim, spacedim>::volume() const
{
  Assert(!handler->is_slave_cell(master_cell),
         ExcMessage("The present function cannot be called for slave cells."));

  if (handler->is_master_cell(master_cell))
    {
      return handler->bboxes[present_index].volume();
    }
  else
    {
      return master_cell->measure();
    }
}



template <int dim, int spacedim>
inline void
AgglomerationAccessor<dim, spacedim>::next()
{
  // Increment the present index and update the polygon
  ++present_index;
  master_cell = handler->master_cells_container[present_index];
}



template <int dim, int spacedim>
inline void
AgglomerationAccessor<dim, spacedim>::prev()
{
  // Decrement the present index and update the polygon
  --present_index;
  master_cell = handler->master_cells_container[present_index];
}


template <int dim, int spacedim>
inline bool
AgglomerationAccessor<dim, spacedim>::operator==(
  const AgglomerationAccessor<dim, spacedim> &other) const
{
  return present_index == other.present_index;
}

template <int dim, int spacedim>
inline bool
AgglomerationAccessor<dim, spacedim>::operator!=(
  const AgglomerationAccessor<dim, spacedim> &other) const
{
  return !(*this == other);
}



template <int dim, int spacedim>
inline types::global_cell_index
AgglomerationAccessor<dim, spacedim>::index() const
{
  return present_index;
}



template <int dim, int spacedim>
typename DoFHandler<dim, spacedim>::active_cell_iterator
AgglomerationAccessor<dim, spacedim>::as_dof_handler_iterator(
  const DoFHandler<dim, spacedim> &dof_handler) const
{
  // Forward the call to the master cell using the right DoFHandler.
  return master_cell->as_dof_handler_iterator(dof_handler);
}



#endif
