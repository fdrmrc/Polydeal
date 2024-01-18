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

#include <deal.II/base/bounding_box_data_out.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/numerics/data_out.h>

#include <agglomeration_handler.h>
#include <poly_utils.h>

template <int dim>
void
test()
{
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, -1, 1);
  tria.refine_global(2);
  MappingQ<dim>             mapping(1);
  GridTools::Cache<dim>     cached_tria(tria, mapping);
  AgglomerationHandler<dim> ah(cached_tria);

  if constexpr (dim == 2)
    {
      std::vector<types::global_cell_index> idxs_to_be_agglomerated = {
        3, 6, 9, 12, 13};
      std::vector<typename Triangulation<dim>::active_cell_iterator>
        cells_to_be_agglomerated;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated,
                                                 cells_to_be_agglomerated);


      auto polytope_iterator = ah.insert_agglomerate(cells_to_be_agglomerated);
      const auto &bbox_agglomeration_pts =
        polytope_iterator->get_bounding_box().get_boundary_points();
      std::cout << "p0: =" << bbox_agglomeration_pts.first << std::endl;
      std::cout << "p1: =" << bbox_agglomeration_pts.second << std::endl;
    }
  else if constexpr (dim == 3)
    {
      std::vector<types::global_cell_index> idxs_to_be_agglomerated = {30, 58};
      std::vector<typename Triangulation<dim>::active_cell_iterator>
        cells_to_be_agglomerated;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated,
                                                 cells_to_be_agglomerated);

      auto polytope_iterator = ah.insert_agglomerate(cells_to_be_agglomerated);
      const auto &bbox_agglomeration_pts =
        polytope_iterator->get_bounding_box().get_boundary_points();
      std::cout << "p0: =" << bbox_agglomeration_pts.first << std::endl;
      std::cout << "p1: =" << bbox_agglomeration_pts.second << std::endl;

      // std::ofstream           ofile("bboxes.vtu");
      // BoundingBoxDataOut<dim> data_out;
      // data_out.build_patches(ah.get_bboxes());
      // data_out.write_vtu(ofile);
    }
}
int
main()
{
  test<2>();
  test<3>();
}
