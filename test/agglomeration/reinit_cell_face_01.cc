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

// Check that FEImmsersedSurfaceValues work correctly by computing the perimeter
// of standard and agglomerated cells.

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include "../tests.h"

int
main()
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria, -1, 1);
  MappingQ<2> mapping(1);
  tria.refine_global(3);
  GridTools::Cache<2>     cached_tria(tria, mapping);
  AgglomerationHandler<2> ah(cached_tria);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated = {
    3, 6, 9, 12, 13}; //{8, 9, 10, 11};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated;
  Tests::collect_cells_for_agglomeration(tria,
                                         idxs_to_be_agglomerated,
                                         cells_to_be_agglomerated);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated2 = {15, 36, 37};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated2;
  Tests::collect_cells_for_agglomeration(tria,
                                         idxs_to_be_agglomerated2,
                                         cells_to_be_agglomerated2);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated3 = {57, 60, 54};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated3;
  Tests::collect_cells_for_agglomeration(tria,
                                         idxs_to_be_agglomerated3,
                                         cells_to_be_agglomerated3);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated4 = {25, 19, 22};

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
  double perimeter = 0.;
  for (const auto &cell :
       ah.agglomeration_cell_iterators() |
         IteratorFilters::ActiveFEIndexEqualTo(ah.AggloIndex::master))
    {
      unsigned int n_agglomerated_faces_per_cell = ah.n_faces(cell);
      for (size_t f = 0; f < n_agglomerated_faces_per_cell; ++f)
        {
          const auto &test_feisv = ah.reinit(cell, f);
          perimeter += std::accumulate(test_feisv.get_JxW_values().begin(),
                                       test_feisv.get_JxW_values().end(),
                                       0.);
        }
      std::cout << "Perimeter of agglomeration with master idx: "
                << cell->active_cell_index() << " is " << perimeter
                << std::endl;
      perimeter = 0.;
    }
}