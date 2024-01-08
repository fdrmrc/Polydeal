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

// Check that quadrature points from both sides of a face are the same
// x - - - - x - - - - x
// |         |         |
// | T_0  f0 | f1  T_1 |
// |         |         |
// x - - - - x - - - - x
//
// If two cells are standard deal.II cells, this is exactly equivalent to
// the creation of two FEFaceValues, one reinited from T_0 and the other one
// reinited from T_1. If one of the two is a face of an agglomeration, then you
// want again the same behaviour, i.e. you want to have the same quadrature
// points, in the same order, as in the standard case.

#include <deal.II/grid/grid_generator.h>

#include <agglomeration_handler.h>
#include <poly_utils.h>


static MappingQ<2> mapping(1);

template <int dim>
void
test_q_points_agglomerated_face(Triangulation<dim> &tria)
{
  GridGenerator::hyper_cube(tria, -1, 1);
  tria.refine_global(3);
  GridTools::Cache<2>     cached_tria(tria, mapping);
  AgglomerationHandler<2> ah(cached_tria);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated = {3, 6, 9};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated,
                                             cells_to_be_agglomerated);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated2 = {36, 37};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated2;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated2,
                                             cells_to_be_agglomerated2);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated3 = {25, 19};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated3;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated3,
                                             cells_to_be_agglomerated3);

  // Agglomerate the cells just stored
  ah.agglomerate_cells(cells_to_be_agglomerated);
  ah.agglomerate_cells(cells_to_be_agglomerated2);
  ah.agglomerate_cells(cells_to_be_agglomerated3);


  FE_DGQ<2> fe_dg(1);
  ah.distribute_agglomerated_dofs(fe_dg);
  ah.initialize_fe_values(QGauss<2>(1), update_default);
  for (const auto &cell : ah.agglo_dh.active_cell_iterators())
    {
      if (!ah.is_slave_cell(cell))
        {
          std::cout << "Cell with index " << cell->active_cell_index()
                    << " has " << ah.n_faces(cell) << " faces" << std::endl;


          for (unsigned int f = 0; f < ah.n_faces(cell); ++f)
            {
              if (!ah.at_boundary(cell, f))
                {
                  const auto & neigh_cell = ah.agglomerated_neighbor(cell, f);
                  unsigned int nofn =
                    ah.neighbor_of_agglomerated_neighbor(cell, f);
                  std::cout
                    << "Neighbor is: " << neigh_cell->active_cell_index()
                    << std::endl;
                  const auto &fe_faces =
                    ah.reinit_interface(cell, neigh_cell, f, nofn);

                  const auto &q_points_neigh =
                    fe_faces.first.get_quadrature_points();
                  const auto &q_points_inside =
                    fe_faces.second.get_quadrature_points();
                  for (unsigned int q_index :
                       fe_faces.first.quadrature_point_indices())
                    {
                      Assert(
                        (q_points_neigh[q_index] - q_points_inside[q_index])
                            .norm() < 1e-15,
                        ExcMessage(
                          "Quadrature points should be the same when seen from the neighboring face."));
                    }
                }
            }
        }
    }
}


int
main()
{
  Triangulation<2> tria;
  test_q_points_agglomerated_face(tria);
  std::cout << "Ok" << std::endl;
}