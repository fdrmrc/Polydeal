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

// Start from a 4x4 grid, agglomerate cells so that you have 4 cells:
// x  -  -   x  -   -  x
// |         |         |
//     12         15
// |         |         |
// x  -  -   x  -  -   x
// |         |         |
//      3          7
// |         |         |
// x   -  -  x  -  -   x
//
// Here, 3, 7, 12, 15 are the indices of the master cells of the agglomeration.
// In this situation, for each cell T, there are *two* face indices from the
// neighboring cell such that the neighbor is T.
// This test checks that `neighbor_of_agglomerated_neighbor` can be called for
// each face of the agglomeration. As per documentation, when a face is a
// boundary face of the triangulation, and invalid unsigned int is returned.

#include <deal.II/grid/grid_generator.h>

#include "../tests.h"

int
main()
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria, -1, 1);
  MappingQ<2> mapping(1);
  tria.refine_global(2);
  GridTools::Cache<2>     cached_tria(tria, mapping);
  AgglomerationHandler<2> ah(cached_tria);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated = {0, 1, 2, 3};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated;
  Tests::collect_cells_for_agglomeration(tria,
                                         idxs_to_be_agglomerated,
                                         cells_to_be_agglomerated);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated2 = {4, 5, 6, 7};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated2;
  Tests::collect_cells_for_agglomeration(tria,
                                         idxs_to_be_agglomerated2,
                                         cells_to_be_agglomerated2);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated3 = {8,
                                                                    9,
                                                                    10,
                                                                    11};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated3;
  Tests::collect_cells_for_agglomeration(tria,
                                         idxs_to_be_agglomerated3,
                                         cells_to_be_agglomerated3);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated4 = {12,
                                                                    13,
                                                                    14,
                                                                    15};

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

  std::vector<std::vector<typename Triangulation<2>::active_cell_iterator>>
    agglomerations{cells_to_be_agglomerated,
                   cells_to_be_agglomerated2,
                   cells_to_be_agglomerated3,
                   cells_to_be_agglomerated4};

  FE_DGQ<2> fe_dg(1);
  ah.distribute_agglomerated_dofs(fe_dg);
  for (const auto &cell :
       ah.agglomeration_cell_iterators() |
         IteratorFilters::ActiveFEIndexEqualTo(ah.CellAgglomerationType::master))
    {
      std::cout << "Cell with idx: " << cell->active_cell_index() << std::endl;
      const unsigned int n_faces = ah.n_faces(cell);
      std::cout << "Number of faces for this cell: " << n_faces << std::endl;
      for (unsigned int f = 0; f < n_faces; ++f)
        {
          std::cout << "Neighbor of neighbor for (" << cell->active_cell_index()
                    << "," << f
                    << ") = " << ah.neighbor_of_agglomerated_neighbor(cell, f)
                    << std::endl;
        }


      std::cout << std::endl;
    }
}
