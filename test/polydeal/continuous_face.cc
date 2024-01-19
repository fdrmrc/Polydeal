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


// On a 2x2 mesh, agglomerate together cells 0,1,2 (call it K1) and create a
// dummy agglomerate (K2) with only cell 3. Later, check the number of faces for
// each agglomerate.
// - - - - - - -
// |     |  K2  |
// |     | - - -
// |  K1        |
// - - - - - - -
//
// From the picture, its clear that:
// K1 has 2 faces (the two lines neighbouring K2) and all the boundary lines
// K2 has 2 faces (the two lines neighbouring K1) and all the boundary lines



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
  tria.refine_global(1);
  GridTools::Cache<2>     cached_tria(tria, mapping);
  AgglomerationHandler<2> ah(cached_tria);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated = {0, 1, 2};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated,
                                             cells_to_be_agglomerated);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated2 = {3};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated2;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated2,
                                             cells_to_be_agglomerated2);
  // Agglomerate the cells just stored
  ah.insert_agglomerate(cells_to_be_agglomerated);
  ah.insert_agglomerate(cells_to_be_agglomerated2);

  std::vector<std::vector<typename Triangulation<2>::active_cell_iterator>>
    agglomerations{cells_to_be_agglomerated, cells_to_be_agglomerated2};

  FE_DGQ<2> fe_dg(1);
  ah.distribute_agglomerated_dofs(fe_dg);
  const auto &info = ah.get_info();
  for (const auto &cell : ah.agglomeration_cell_iterators() |
                            IteratorFilters::ActiveFEIndexEqualTo(
                              ah.CellAgglomerationType::master))
    {
      std::cout << "Master cell index = " << cell->active_cell_index()
                << std::endl;
      unsigned int n_faces = ah.n_agglomerated_faces(cell);
      std::cout << "Number of agglomerated faces = " << n_faces << std::endl;
      for (unsigned int f = 0; f < n_faces; ++f)
        {
          std::cout << "Agglomerate face index = " << f << std::endl;
          const auto &vec_pair = info.at({cell, f});
          for (const auto &p : vec_pair)
            {
              std::cout << "deal.II cell index = "
                        << p.first->active_cell_index() << std::endl;
              std::cout << "Local face idx = " << p.second << std::endl;
            }
          std::cout << std::endl;
        }
    }
}