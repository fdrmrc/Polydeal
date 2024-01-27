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

  std::vector<types::global_cell_index> idxs_to_be_agglomerated = {3, 6, 9};
  store_flagged_cells(idxs_to_be_agglomerated);

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated,
                                             cells_to_be_agglomerated);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated2 = {36, 37};
  store_flagged_cells(idxs_to_be_agglomerated2);

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated2;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated2,
                                             cells_to_be_agglomerated2);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated3 = {25, 19};
  store_flagged_cells(idxs_to_be_agglomerated3);

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated3;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated3,
                                             cells_to_be_agglomerated3);

  // Agglomerate the cells just stored
  ah.define_agglomerate(cells_to_be_agglomerated);
  ah.define_agglomerate(cells_to_be_agglomerated2);
  ah.define_agglomerate(cells_to_be_agglomerated3);

  for (std::size_t i = 0; i < tria.n_active_cells(); ++i)
    {
      // If not present, agglomerate all the singletons
      if (std::find(flagged_cells.begin(),
                    flagged_cells.end(),
                    cells[i]->active_cell_index()) == std::end(flagged_cells))
        ah.insert_agglomerate({cells[i]});
    }



  FE_DGQ<2> fe_dg(1);
  ah.distribute_agglomerated_dofs(fe_dg);
  ah.initialize_fe_values(QGauss<2>(1), update_default);
  for (const auto &polytope : ah.polytope_iterators())
    {
      std::cout << "Cell with index "
                << polytope.master_cell()->active_cell_index() << " has "
                << polytope->n_faces() << " faces" << std::endl;

      for (unsigned int f = 0; f < polytope->n_faces(); ++f)
        {
          if (!polytope->at_boundary(f))
            {
              const auto & neigh_polytope = polytope->neighbor(f);
              unsigned int nofn =
                polytope->neighbor_of_agglomerated_neighbor(f);
              std::cout << "Neighbor is: "
                        << neigh_polytope.master_cell()->active_cell_index()
                        << std::endl;
              const auto &fe_faces =
                ah.reinit_interface(polytope, neigh_polytope, f, nofn);

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


int
main()
{
  Triangulation<2> tria;
  test_q_points_agglomerated_face(tria);
  std::cout << "Ok" << std::endl;
}