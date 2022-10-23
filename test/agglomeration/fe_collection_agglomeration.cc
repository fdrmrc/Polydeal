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

// Loop over all cells of a triangulation, where some cells have been
// agglomerated together, and compute the measure of the mesh by adding the
// weights coming from every cell. What has to happen is that contributions are
// coming from agglomerations and standard cells, while slave cells have no
// quadrature rule over them.

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include "../tests.h"

void test_hyper_cube(Triangulation<2> &tria) {
  GridGenerator::hyper_cube(tria, -1, 1);
  MappingQ<2> mapping(1);
  tria.refine_global(3);
  GridTools::Cache<2> cached_tria(tria, mapping);
  AgglomerationHandler<2> ah(cached_tria);

  std::vector<unsigned int> idxs_to_be_agglomerated = {3, 6, 9, 12, 13};

  std::vector<typename Triangulation<2>::active_cell_iterator>
      cells_to_be_agglomerated;
  Tests::collect_cells_for_agglomeration(tria, idxs_to_be_agglomerated,
                                         cells_to_be_agglomerated);

  std::vector<unsigned int> idxs_to_be_agglomerated2 = {15, 36, 37};

  std::vector<typename Triangulation<2>::active_cell_iterator>
      cells_to_be_agglomerated2;
  Tests::collect_cells_for_agglomeration(tria, idxs_to_be_agglomerated2,
                                         cells_to_be_agglomerated2);

  std::vector<unsigned int> idxs_to_be_agglomerated3 = {57, 60, 54};

  std::vector<typename Triangulation<2>::active_cell_iterator>
      cells_to_be_agglomerated3;
  Tests::collect_cells_for_agglomeration(tria, idxs_to_be_agglomerated3,
                                         cells_to_be_agglomerated3);

  std::vector<unsigned int> idxs_to_be_agglomerated4 = {25, 19, 22};

  std::vector<typename Triangulation<2>::active_cell_iterator>
      cells_to_be_agglomerated4;
  Tests::collect_cells_for_agglomeration(tria, idxs_to_be_agglomerated4,
                                         cells_to_be_agglomerated4);

  // Agglomerate the cells just stored
  ah.agglomerate_cells(cells_to_be_agglomerated);
  ah.agglomerate_cells(cells_to_be_agglomerated2);
  ah.agglomerate_cells(cells_to_be_agglomerated3);
  ah.agglomerate_cells(cells_to_be_agglomerated4);

  std::vector<std::vector<typename Triangulation<2>::active_cell_iterator>>
      agglomerations{cells_to_be_agglomerated, cells_to_be_agglomerated2,
                     cells_to_be_agglomerated3, cells_to_be_agglomerated4};

  ah.initialize_hp_structure();
  ah.set_agglomeration_flags(update_JxW_values);
  ah.set_quadrature_degree(1);
  double total_sum = 0.;
  for (const auto &cell : ah.agglo_dh.active_cell_iterators()) {
    const auto &fev_general = ah.reinit(cell);
    for (const auto weight : fev_general.get_JxW_values()) total_sum += weight;
  }

  Assert(total_sum == GridTools::volume(tria, mapping),
         ExcMessage("Integration did not succeed"));
  std::cout << "Ok" << std::endl;
}

void test_hyper_ball(Triangulation<2> &tria) {
  GridGenerator::hyper_ball(tria, {}, 2.);
  MappingQ<2> mapping(1);
  tria.refine_global(4);
  GridTools::Cache<2> cached_tria(tria, mapping);
  AgglomerationHandler<2> ah(cached_tria);

  std::vector<unsigned int> idxs_to_be_agglomerated = {3, 6, 9, 12, 13};

  std::vector<typename Triangulation<2>::active_cell_iterator>
      cells_to_be_agglomerated;
  Tests::collect_cells_for_agglomeration(tria, idxs_to_be_agglomerated,
                                         cells_to_be_agglomerated);

  std::vector<unsigned int> idxs_to_be_agglomerated2 = {15, 36, 37};

  std::vector<typename Triangulation<2>::active_cell_iterator>
      cells_to_be_agglomerated2;
  Tests::collect_cells_for_agglomeration(tria, idxs_to_be_agglomerated2,
                                         cells_to_be_agglomerated2);

  std::vector<unsigned int> idxs_to_be_agglomerated3 = {57, 60, 54};

  std::vector<typename Triangulation<2>::active_cell_iterator>
      cells_to_be_agglomerated3;
  Tests::collect_cells_for_agglomeration(tria, idxs_to_be_agglomerated3,
                                         cells_to_be_agglomerated3);

  std::vector<unsigned int> idxs_to_be_agglomerated4 = {25, 19, 22};

  std::vector<typename Triangulation<2>::active_cell_iterator>
      cells_to_be_agglomerated4;
  Tests::collect_cells_for_agglomeration(tria, idxs_to_be_agglomerated4,
                                         cells_to_be_agglomerated4);

  // Agglomerate the cells just stored
  ah.agglomerate_cells(cells_to_be_agglomerated);
  ah.agglomerate_cells(cells_to_be_agglomerated2);
  ah.agglomerate_cells(cells_to_be_agglomerated3);
  ah.agglomerate_cells(cells_to_be_agglomerated4);

  std::vector<std::vector<typename Triangulation<2>::active_cell_iterator>>
      agglomerations{cells_to_be_agglomerated, cells_to_be_agglomerated2,
                     cells_to_be_agglomerated3, cells_to_be_agglomerated4};

  ah.initialize_hp_structure();
  ah.set_agglomeration_flags(update_JxW_values);
  ah.set_quadrature_degree(2);
  double total_sum = 0.;
  for (const auto &cell : ah.agglo_dh.active_cell_iterators()) {
    const auto &fev_general = ah.reinit(cell);
    for (const auto weight : fev_general.get_JxW_values()) total_sum += weight;
  }

  Assert(total_sum == GridTools::volume(tria, mapping),
         ExcMessage("Integration did not succeed"));
  std::cout << "Ok" << std::endl;
}

int main() {
  Triangulation<2> tria;
  test_hyper_cube(tria);
  tria.clear();
  test_hyper_ball(tria);

  return 0;
}
