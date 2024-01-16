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

// Agglomerate some cells in a grid, and create a finite element space on the
// bounding box of an agglomeration. To check the correctness, compute the area
// of the agglomerated cells using the weights of a custom quadrature rule over
// the agglomerated element.

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <agglomeration_handler.h>
#include <poly_utils.h>

template <int dim>
void
test()
{
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, -1, 1);
  MappingQ<dim> mapping(1);
  tria.refine_global(3);
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

      std::vector<types::global_cell_index> idxs_to_be_agglomerated2 = {15,
                                                                        36,
                                                                        37};

      std::vector<typename Triangulation<dim>::active_cell_iterator>
        cells_to_be_agglomerated2;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated2,
                                                 cells_to_be_agglomerated2);

      std::vector<types::global_cell_index> idxs_to_be_agglomerated3 = {57,
                                                                        60,
                                                                        54};

      std::vector<typename Triangulation<dim>::active_cell_iterator>
        cells_to_be_agglomerated3;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated3,
                                                 cells_to_be_agglomerated3);

      std::vector<types::global_cell_index> idxs_to_be_agglomerated4 = {25,
                                                                        19,
                                                                        22};

      std::vector<typename Triangulation<dim>::active_cell_iterator>
        cells_to_be_agglomerated4;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated4,
                                                 cells_to_be_agglomerated4);

      // Agglomerate the cells just stored
      ah.insert_agglomerate(cells_to_be_agglomerated);
      ah.insert_agglomerate(cells_to_be_agglomerated2);
      ah.insert_agglomerate(cells_to_be_agglomerated3);
      ah.insert_agglomerate(cells_to_be_agglomerated4);
    }
  else if constexpr (dim == 3)
    {
      std::vector<types::global_cell_index> idxs_to_be_agglomerated = {463,
                                                                       459};
      std::vector<typename Triangulation<dim>::active_cell_iterator>
        cells_to_be_agglomerated;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated,
                                                 cells_to_be_agglomerated);

      ah.insert_agglomerate(cells_to_be_agglomerated);
    }

  FE_DGQ<dim> fe_dg(1);
  ah.distribute_agglomerated_dofs(fe_dg);
  ah.initialize_fe_values(QGauss<dim>(1), update_JxW_values);

  // Variant using iterators
  for (const auto &polytope : ah.polytope_iterators())
    {
      const auto &fev = ah.reinit(polytope);
      double      sum = 0.;
      for (const auto weight : fev.get_JxW_values())
        sum += weight;
      std::cout << "Sum is: " << sum << std::endl;
    }
}


int
main()
{
  test<2>();
  test<3>();
  return 0;
}
