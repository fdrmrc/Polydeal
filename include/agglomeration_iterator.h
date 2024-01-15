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


#ifndef agglomeration_iterator_h
#define agglomeration_iterator_h


#include <agglomeration_accessor.h>


/**
 * A class that is used to iterate over polygons. Together with the
 * AgglomerationAccessor class this is used to hide the internal implementation
 * of the particle class and the particle container.
 */
template <int dim, int spacedim = dim>
class AgglomerationIterator
{
public:
  using agglomeration_container =
    typename AgglomerationAccessor<dim, spacedim>::agglomeration_container;

  /**
   * Empty constructor. Such an object is not usable!
   */
  AgglomerationIterator() = default;

  /**
   * Constructor of the iterator. Takes a reference to the cells forming the
   * actual polygon.
   */
  AgglomerationIterator(
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell,
    const AgglomerationHandler<dim, spacedim> *                        handler);

  /**
   * Dereferencing operator, returns a reference to an accessor. Usage is thus
   * like <tt>(*i).get_dof_indices ();</tt>
   */
  const AgglomerationAccessor<dim, spacedim> &
  operator*() const;

  /**
   * Dereferencing operator, non-@p const version.
   */
  AgglomerationAccessor<dim, spacedim> &
  operator*();

  /**
   * Dereferencing operator, returns a pointer of the particle pointed to.
   * Usage is thus like <tt>i->get_dof_indices ();</tt>
   *
   * There is a @p const and a non-@p const version.
   */
  const AgglomerationAccessor<dim, spacedim> *
  operator->() const;

  /**
   * Dereferencing operator, non-@p const version.
   */
  AgglomerationAccessor<dim, spacedim> *
  operator->();

  /**
   * Compare for equality.
   */
  bool
  operator==(const AgglomerationIterator<dim, spacedim> &) const;

  /**
   * Compare for inequality.
   */
  bool
  operator!=(const AgglomerationIterator<dim, spacedim> &) const;

  /**
   * Prefix <tt>++</tt> operator: <tt>++iterator</tt>. This operator advances
   * the iterator to the next element and returns a reference to
   * <tt>*this</tt>.
   */
  AgglomerationIterator &
  operator++();

  /**
   * Postfix <tt>++</tt> operator: <tt>iterator++</tt>. This operator advances
   * the iterator to the next element, but returns an iterator to the element
   * previously pointed to.
   */
  AgglomerationIterator
  operator++(int);

  // /**
  //  * Prefix <tt>\--</tt> operator: <tt>\--iterator</tt>. This operator moves
  //  * the iterator to the previous element and returns a reference to
  //  * <tt>*this</tt>.
  //  */
  // AgglomerationIterator &
  // operator--();

  // /**
  //  * Postfix <tt>\--</tt> operator: <tt>iterator\--</tt>. This operator moves
  //  * the iterator to the previous element, but returns an iterator to the
  //  * element previously pointed to.
  //  */
  // AgglomerationIterator
  // operator--(int);

  /**
   * Mark the class as bidirectional iterator and declare some alias which
   * are standard for iterators and are used by algorithms to enquire about
   * the specifics of the iterators they work on.
   */
  using iterator_category = std::bidirectional_iterator_tag;
  using value_type        = AgglomerationAccessor<dim, spacedim>;
  using difference_type   = std::ptrdiff_t;
  using pointer           = AgglomerationAccessor<dim, spacedim> *;
  using reference         = AgglomerationAccessor<dim, spacedim> &;

private:
  /**
   * The accessor to the actual polygon.
   */
  AgglomerationAccessor<dim, spacedim> accessor;
};



// ------------------------------ inline functions -------------------------

template <int dim, int spacedim>
inline AgglomerationIterator<dim, spacedim>::AgglomerationIterator(
  const typename Triangulation<dim, spacedim>::active_cell_iterator
    &                                        master_cell,
  const AgglomerationHandler<dim, spacedim> *handler)
  : accessor(master_cell, handler)
{}



template <int dim, int spacedim>
inline AgglomerationAccessor<dim, spacedim> &
AgglomerationIterator<dim, spacedim>::operator*()
{
  return accessor;
}



template <int dim, int spacedim>
inline AgglomerationAccessor<dim, spacedim> *
AgglomerationIterator<dim, spacedim>::operator->()
{
  return &(this->operator*());
}



template <int dim, int spacedim>
inline const AgglomerationAccessor<dim, spacedim> &
AgglomerationIterator<dim, spacedim>::operator*() const
{
  return accessor;
}



template <int dim, int spacedim>
inline const AgglomerationAccessor<dim, spacedim> *
AgglomerationIterator<dim, spacedim>::operator->() const
{
  return &(this->operator*());
}



template <int dim, int spacedim>
inline bool
AgglomerationIterator<dim, spacedim>::operator!=(
  const AgglomerationIterator<dim, spacedim> &other) const
{
  return accessor != other.accessor;
}



template <int dim, int spacedim>
inline bool
AgglomerationIterator<dim, spacedim>::operator==(
  const AgglomerationIterator<dim, spacedim> &other) const
{
  return accessor == other.accessor;
}



template <int dim, int spacedim>
inline AgglomerationIterator<dim, spacedim> &
AgglomerationIterator<dim, spacedim>::operator++()
{
  accessor.next();
  return *this;
}



template <int dim, int spacedim>
inline AgglomerationIterator<dim, spacedim>
AgglomerationIterator<dim, spacedim>::operator++(int)
{
  AgglomerationIterator tmp(*this);
                        operator++();

  return tmp;
}



// template <int dim, int spacedim>
// inline AgglomerationIterator<dim, spacedim> &
// AgglomerationIterator<dim, spacedim>::operator--()
// {
//   accessor.prev();
//   return *this;
// }



// template <int dim, int spacedim>
// inline AgglomerationIterator<dim, spacedim>
// AgglomerationIterator<dim, spacedim>::operator--(int)
// {
//   AgglomerationIterator tmp(*this);
//                         operator--();

//   return tmp;
// }



#endif