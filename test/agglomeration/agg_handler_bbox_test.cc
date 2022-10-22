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

// Select some cells of a tria, agglomerated them together and check the
// bounding box of the resulting agglomeration.

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/data_out.h>

#include "../tests.h"

int main() {
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria, -1, 1);
  tria.refine_global(2);
  MappingQ<2> mapping(1);
  GridTools::Cache<2> cached_tria(tria, mapping);
  AgglomerationHandler<2> ah(cached_tria);

  std::vector<unsigned int> idxs_to_be_agglomerated = {3, 6, 9, 12, 13};
  std::vector<typename Triangulation<2>::active_cell_iterator>
      cells_to_be_agglomerated;
  Tests::collect_cells_for_agglomeration(tria, idxs_to_be_agglomerated,
                                         cells_to_be_agglomerated);

  ah.agglomerate_cells(cells_to_be_agglomerated);
  const auto bbox_agglomeration_pts = get_bboxes(ah)[13].get_boundary_points();
  std::cout << "p0: =" << bbox_agglomeration_pts.first << std::endl;
  std::cout << "p1: =" << bbox_agglomeration_pts.second << std::endl;
}
