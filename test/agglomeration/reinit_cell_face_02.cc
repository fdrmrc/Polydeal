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

// Select some cells of a tria, agglomerated them together and check that you
// can call reinit(cell,f) on both standard and agglomerated (internal) cells,
// as it happens when you assemble a jump with DG.
// More specifically, given a face shared by two neighbors, you want to be able
// to call reinit(cell,f)
// reinit(cell->neighbor(f),cell->neighbor_of_neighbor(f))
// when the two cells may be agglomerated or not. See the following sketch:
//
// x - - - - x - - - - x
// |         |         |
// | T_0  f0 | f1  T_1 |
// |         |         |
// x - - - - x - - - - x
//
// This test checks that a FEFaceValues with the same number of DoFs is reinited
// on each side of the shared face.

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include "../tests.h"

static MappingQ<2> mapping(1);


template <int dim>
void
reinit_on_neighbor(Triangulation<dim> &tria)
{
  GridGenerator::hyper_cube(tria, -1, 1);
  tria.refine_global(3);
  GridTools::Cache<2>     cached_tria(tria, mapping);
  AgglomerationHandler<2> ah(cached_tria);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated = {
    3, 6, 9}; //{8, 9, 10, 11};

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


  FE_DGQ<2> fe_dg(1);
  ah.distribute_agglomerated_dofs(fe_dg);
  for (const auto &cell : ah.agglomeration_cell_iterators())
    {
      if (!ah.is_slave_cell(cell))
        {
          unsigned int n_faces = ah.n_faces(cell);
          std::cout << "Cell with index " << cell->active_cell_index()
                    << " has " << n_faces << " faces" << std::endl;
          for (unsigned int f = 0; f < n_faces; ++f)
            {
              if (!ah.at_boundary(cell, f))
                {
                  const auto &neigh_cell = ah.agglomerated_neighbor(cell, f);
                  std::cout
                    << "Neighbor index= " << neigh_cell->active_cell_index()
                    << std::endl;
                  unsigned int nofn =
                    ah.neighbor_of_agglomerated_neighbor(cell, f);
                  std::cout << "Neighbor of neighbor(" << f
                            << ") = " << neigh_cell->active_cell_index()
                            << std::endl;

                  const auto &interface_fe_face_values =
                    ah.reinit_interface(cell, neigh_cell, f, nofn);
                  Assert(interface_fe_face_values.first.dofs_per_cell == 4 &&
                           interface_fe_face_values.second.dofs_per_cell == 4,
                         ExcMessage("Das kann nicht wahr sein..."));
                }
              else
                {
                  std::cout << "Face with idx: " << f << " is a boundary face."
                            << std::endl;
                }
            }
          std::cout << std::endl;
        }
    }
}

int
main()
{
  Triangulation<2> tria;
  reinit_on_neighbor(tria);
}