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

#include <agglomeration_handler.h>
#include <poly_utils.h>

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
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated,
                                             cells_to_be_agglomerated);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated2 = {15, 36, 37};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated2;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated2,
                                             cells_to_be_agglomerated2);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated3 = {57, 60, 54};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated3;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated3,
                                             cells_to_be_agglomerated3);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated4 = {25, 19, 22};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated4;
  PolyUtils::collect_cells_for_agglomeration(tria,
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
  for (const auto &cell : ah.agglomeration_cell_iterators() |
                            IteratorFilters::ActiveFEIndexEqualTo(
                              ah.CellAgglomerationType::master))
    {
      std::cout << "Cell with idx: " << cell->active_cell_index() << std::endl;
      unsigned int n_agglomerated_faces_per_cell = ah.n_faces(cell);
      std::cout << "Number of faces for the agglomeration: "
                << n_agglomerated_faces_per_cell << std::endl;
      for (unsigned int f = 0; f < n_agglomerated_faces_per_cell; ++f)
        {
          std::cout << "Agglomerated face with idx: " << f << std::endl;
          auto agglomerate_connectivity = ah.get_agglomerated_connectivity();
          const auto &[local_face_idx, neigh, local_face_idx_out, dummy] =
            agglomerate_connectivity[{cell, f}];
          {
            std::cout << "Face idx: " << local_face_idx << std::endl;
            if (neigh.state() == IteratorState::valid)
              {
                std::cout << "Neighbor idx: " << neigh->active_cell_index()
                          << std::endl;
              }

            std::cout << "Face idx from outside: " << local_face_idx_out
                      << std::endl;
          }
          std::cout << std::endl;
        }
    }
}
