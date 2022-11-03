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


#ifndef tests_h
#define tests_h


#include <deal.II/fe/fe_dgq.h>

#include "include/agglomeration_handler.h"

namespace Tests
{
  template <int dim, int spacedim = dim>
  void
  collect_cells_for_agglomeration(
    const Triangulation<dim, spacedim> &tria,
    std::vector<unsigned int>          &cell_idxs,
    std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>
      &cells_to_be_agglomerated)
  {
    Assert(cells_to_be_agglomerated.size() == 0,
           ExcMessage(
             "The vector of cells is supposed to be filled by this function."));
    for (const auto &cell : tria.active_cell_iterators())
      if (std::find(cell_idxs.begin(),
                    cell_idxs.end(),
                    cell->active_cell_index()) != cell_idxs.end())
        {
          cells_to_be_agglomerated.push_back(cell);
        }
  }

} // namespace Tests

#endif // tests_h