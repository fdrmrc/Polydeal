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

#include <agglomeration_handler.h>
#include <poly_utils.h>


static MappingQ<2> mapping(1);


template <int dim>
void
reinit_on_neighbor(Triangulation<dim> &tria)
{
  GridGenerator::hyper_cube(tria, -1, 1);
  tria.refine_global(3);
  GridTools::Cache<2>     cached_tria(tria, mapping);
  AgglomerationHandler<2> ah(cached_tria);
  FE_DGQ<2>               fe_dg(1);


  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells; // each cell = an agglomerate
  for (const auto &cell : tria.active_cell_iterators())
    cells.push_back(cell);

  std::vector<types::global_cell_index> flagged_cells;
  const auto                            store_flagged_cells =
    [&flagged_cells](
      const std::vector<types::global_cell_index> &idxs_to_be_agglomerated) {
      for (const int idx : idxs_to_be_agglomerated)
        flagged_cells.push_back(idx);
    };

  std::vector<types::global_cell_index> idxs_to_be_agglomerated = {
    3, 6, 9}; //{8, 9, 10, 11};
  store_flagged_cells(idxs_to_be_agglomerated);
  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated,
                                             cells_to_be_agglomerated);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated2 = {15, 36, 37};
  store_flagged_cells(idxs_to_be_agglomerated2);

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated2;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated2,
                                             cells_to_be_agglomerated2);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated3 = {57, 60, 54};
  store_flagged_cells(idxs_to_be_agglomerated3);

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated3;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated3,
                                             cells_to_be_agglomerated3);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated4 = {25, 19, 22};
  store_flagged_cells(idxs_to_be_agglomerated4);

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated4;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated4,
                                             cells_to_be_agglomerated4);

  // Agglomerate the cells just stored
  ah.define_agglomerate(cells_to_be_agglomerated);
  ah.define_agglomerate(cells_to_be_agglomerated2);
  ah.define_agglomerate(cells_to_be_agglomerated3);
  ah.define_agglomerate(cells_to_be_agglomerated4);

  for (std::size_t i = 0; i < tria.n_active_cells(); ++i)
    {
      // If not present, agglomerate all the singletons
      if (std::find(flagged_cells.begin(),
                    flagged_cells.end(),
                    cells[i]->active_cell_index()) == std::end(flagged_cells))
        ah.define_agglomerate({cells[i]});
    }

  ah.distribute_agglomerated_dofs(fe_dg);
  ah.initialize_fe_values(QGauss<2>(1), update_default);
  for (const auto &polytope : ah.polytope_iterators())
    {
      unsigned int n_faces = polytope->n_faces();
      std::cout << "Cell with index "
                << polytope.master_cell()->active_cell_index() << " has "
                << n_faces << " faces" << std::endl;
      for (unsigned int f = 0; f < n_faces; ++f)
        {
          if (!polytope->at_boundary(f))
            {
              const auto &neigh_polytope = polytope->neighbor(f);
              std::cout << "Neighbor index= "
                        << neigh_polytope.master_cell()->active_cell_index()
                        << std::endl;
              unsigned int nofn =
                polytope->neighbor_of_agglomerated_neighbor(f);
              std::cout << "Neighbor of neighbor(" << f << ") = "
                        << neigh_polytope.master_cell()->active_cell_index()
                        << std::endl;

              AssertThrow(neigh_polytope->neighbor(nofn)->index() ==
                            polytope->index(),
                          ExcMessage("Mismatch!"));

              const auto &interface_fe_face_values =
                ah.reinit_interface(polytope, neigh_polytope, f, nofn);

              AssertThrow(interface_fe_face_values.first.dofs_per_cell == 4 &&
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

int
main()
{
  Triangulation<2> tria;
  reinit_on_neighbor(tria);
}