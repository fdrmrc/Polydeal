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
#include <deal.II/grid/grid_out.h>

#include "../tests.h"

int
main()
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria, -1, 1);
  tria.refine_global(3);
  std::unique_ptr<GridTools::Cache<2>> cached_tria( new GridTools::Cache<2>(tria,MappingQ<2>(1)));
  AgglomerationHandler<2> ah(cached_tria);

  std::vector<unsigned int> idxs_to_be_agglomerated = {
    3, 6, 9, 12, 13}; //{8, 9, 10, 11};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated;
  Tests::collect_cells_for_agglomeration(tria,
                                         idxs_to_be_agglomerated,
                                         cells_to_be_agglomerated);

  std::vector<unsigned int> idxs_to_be_agglomerated2 = {15, 36, 37};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated2;
  Tests::collect_cells_for_agglomeration(tria,
                                         idxs_to_be_agglomerated2,
                                         cells_to_be_agglomerated2);


  std::vector<unsigned int> idxs_to_be_agglomerated3 = {57, 60, 54};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated3;
  Tests::collect_cells_for_agglomeration(tria,
                                         idxs_to_be_agglomerated3,
                                         cells_to_be_agglomerated3);

  std::vector<unsigned int> idxs_to_be_agglomerated4 = {25, 19, 22};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated4;
  Tests::collect_cells_for_agglomeration(tria,
                                         idxs_to_be_agglomerated4,
                                         cells_to_be_agglomerated4);

  // Agglomerate the cells just stored
  ah.agglomerate_cells(cells_to_be_agglomerated);
  ah.agglomerate_cells(cells_to_be_agglomerated2);
  ah.agglomerate_cells(cells_to_be_agglomerated3);
  ah.agglomerate_cells(cells_to_be_agglomerated4);
  ah.setup_neighbors_of_agglomeration(cells_to_be_agglomerated);
  ah.setup_neighbors_of_agglomeration(cells_to_be_agglomerated2);
  ah.setup_neighbors_of_agglomeration(cells_to_be_agglomerated3);
  ah.setup_neighbors_of_agglomeration(cells_to_be_agglomerated4);


  for (const auto &value : get_agglomerated_connectivity(ah))
    { // value is a set now
      for (const auto &x : value.second)
        {
          std::cout << "Cell with idx: " << (value.first)->active_cell_index()
                    << " has neighbor with idx: "
                    << (x.first)->active_cell_index()
                    << " and local face numbering: " << x.second << std::endl;
        }
    }
}