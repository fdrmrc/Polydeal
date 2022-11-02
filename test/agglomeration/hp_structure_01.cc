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

// Create a 4x4 rectangular grid with 4 master cells like in the following
// (marked with M):
// x - x - x - x - x
// |   |   |   |   |
// x - x - x - x - x
// | M |   | M |   |
// x - x - x - x - x
// |   |   |   |   |
// x - x - x - x - x
// | M |   | M |   |
// x - x - x - x - x
// Check they're the only active ones and print DoFs indices and vertices.

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

  std::vector<unsigned int> idxs_to_be_agglomerated = {0, 1, 2, 3};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated;
  Tests::collect_cells_for_agglomeration(tria,
                                         idxs_to_be_agglomerated,
                                         cells_to_be_agglomerated);

  std::vector<unsigned int> idxs_to_be_agglomerated2 = {4, 5, 6, 7};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated2;
  Tests::collect_cells_for_agglomeration(tria,
                                         idxs_to_be_agglomerated2,
                                         cells_to_be_agglomerated2);

  std::vector<unsigned int> idxs_to_be_agglomerated3 = {8, 9, 10, 11};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated3;
  Tests::collect_cells_for_agglomeration(tria,
                                         idxs_to_be_agglomerated3,
                                         cells_to_be_agglomerated3);

  std::vector<unsigned int> idxs_to_be_agglomerated4 = {12, 13, 14, 15};

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

  ah.initialize_hp_structure();
  ah.setup_connectivity_of_agglomeration();
  std::vector<types::global_dof_index> dof_indices(4);

  for (const auto &cell :
       ah.agglo_dh.active_cell_iterators() |
         IteratorFilters::ActiveFEIndexEqualTo(ah.AggloIndex::master))
    {
      cell->get_dof_indices(dof_indices);
      std::cout << "Cell with global index: " << cell->active_cell_index()
                << " has global DoF indices: " << std::endl;
      for (const auto &idx : dof_indices)
        {
          std::cout << idx << std::endl;
        }
      std::cout << " and vertices: " << std::endl;
      typename Triangulation<2>::cell_iterator cell_it(*cell, &ah.agglo_dh);
      for (const auto i : cell_it->vertex_indices())
        {
          std::cout << cell_it->vertex(i) << std::endl;
        }
    }

  return 0;
}