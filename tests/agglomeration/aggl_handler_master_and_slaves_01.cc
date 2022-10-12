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


// Select some cells of a tria, agglomerated them together and check that the
// vector describing the agglomeration has the right information, i.e.
// v[idx] = -1 if cell is master, otherwise index of the master of idx-th cell.

#include <deal.II/grid/grid_generator.h>

#include "agglomeration_handler.h"

int
main()
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria, -1, 1);
  tria.refine_global(2);
  AgglomerationHandler<2> ah(tria);

  std::vector<unsigned int> idxs_to_be_agglomerated = {3, 6, 9, 12, 13};
  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated;
  for (const auto &cell : tria.active_cell_iterators())
    {
      if (std::find(idxs_to_be_agglomerated.begin(),
                    idxs_to_be_agglomerated.end(),
                    cell->active_cell_index()) != idxs_to_be_agglomerated.end())
        {
          cells_to_be_agglomerated.push_back(cell);
        }
    }


  // Agglomerate the cells just stored
  ah.agglomerate_cells(cells_to_be_agglomerated);
  ah.print_agglomeration(std::cout);
}