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

#include <deal.II/base/bounding_box.h>
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
  using AgglomerationContainer =
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
  AgglomerationContainer
  get_agglomerate() const;

  /**
   * Return the BoundingBox which bounds the present polytope.
   */
  const BoundingBox<dim> &
  get_bounding_box() const;


  /**
   * Return the index of the present polytope.
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

  /**
   * Returns the number of classical deal.II cells that are building the present
   * polygon.
   */
  unsigned int
  n_background_cells() const;


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

  /**
   * Returns the slaves of the present agglomeration.
   */
  const AgglomerationContainer &
  get_slaves() const;

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
  // if (handler->is_standard_cell(master_cell))
  //   {
  //     return master_cell->n_faces();
  //   }
  return handler->number_of_agglomerated_faces[present_index];
}



template <int dim, int spacedim>
const AgglomerationIterator<dim, spacedim>
AgglomerationAccessor<dim, spacedim>::neighbor(const unsigned int f) const
{
  if (!at_boundary(f))
    {
      const auto &neigh =
        handler->polytope_cache.cell_face_at_boundary
          .at({handler->master2polygon.at(master_cell->active_cell_index()), f})
          .second;
      typename DoFHandler<dim, spacedim>::active_cell_iterator cell_dh(
        *neigh, &(handler->agglo_dh));
      return {cell_dh, handler};
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
  // First, make sure it's not a boundary face.
  if (!at_boundary(f))
    {
      const auto &neigh_polytope =
        neighbor(f); // returns the neighboring master

      AssertThrow(neigh_polytope.state() == IteratorState::valid,
                  ExcInternalError());
      const unsigned int n_faces_agglomerated_neighbor =
        neigh_polytope->n_faces();

      // Loop over all cells of neighboring agglomerate
      for (unsigned int f_out = 0; f_out < n_faces_agglomerated_neighbor;
           ++f_out)
        {
          // Check if same master cell
          if (neigh_polytope->neighbor(f_out).state() == IteratorState::valid)
            if (neigh_polytope->neighbor(f_out)->index() == index())
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
inline typename AgglomerationAccessor<dim, spacedim>::AgglomerationContainer
AgglomerationAccessor<dim, spacedim>::get_agglomerate() const
{
  auto agglomeration = get_slaves();
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
    {
      typename DoFHandler<dim, spacedim>::active_cell_iterator cell_dh(
        *master_cell, &(handler->agglo_dh));
      return handler->at_boundary(cell_dh, f);
    }
  // return std::get<2>(handler->master_neighbors[{master_cell, f}]) ==
  //        std::numeric_limits<unsigned int>::max();
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
inline const BoundingBox<dim> &
AgglomerationAccessor<dim, spacedim>::get_bounding_box() const
{
  return handler->bboxes[present_index];
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
  // Increment the present index and update the polytope
  ++present_index;
  master_cell = handler->master_cells_container[present_index];
}



template <int dim, int spacedim>
inline void
AgglomerationAccessor<dim, spacedim>::prev()
{
  // Decrement the present index and update the polytope
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



template <int dim, int spacedim>
inline const typename AgglomerationAccessor<dim,
                                            spacedim>::AgglomerationContainer &
AgglomerationAccessor<dim, spacedim>::get_slaves() const
{
  return handler->master2slaves.at(master_cell->active_cell_index());
}



template <int dim, int spacedim>
inline unsigned int
AgglomerationAccessor<dim, spacedim>::n_background_cells() const
{
  AssertThrow(get_agglomerate().size() > 0, ExcMessage("Empty agglomeration."));
  return get_agglomerate().size();
}



#endif
