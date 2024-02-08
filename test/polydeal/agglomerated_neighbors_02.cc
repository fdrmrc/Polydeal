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
// connectivity information for master cells is correct.

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
  tria.refine_global(2);
  GridTools::Cache<2>     cached_tria(tria, mapping);
  AgglomerationHandler<2> ah(cached_tria);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated = {0, 1, 2, 3};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated,
                                             cells_to_be_agglomerated);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated2 = {4, 5, 6, 7};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated2;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated2,
                                             cells_to_be_agglomerated2);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated3 = {8,
                                                                    9,
                                                                    10,
                                                                    11};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated3;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated3,
                                             cells_to_be_agglomerated3);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated4 = {12,
                                                                    13,
                                                                    14,
                                                                    15};

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


  FE_DGQ<2> fe_dg(1);
  ah.distribute_agglomerated_dofs(fe_dg);
  const auto &interface = ah.get_interface();

  for (const auto &polytope : ah.polytope_iterators())
    {
      std::cout << "Polytope with idx: " << polytope->index() << std::endl;
      unsigned int n_agglomerated_faces_per_cell = polytope->n_faces();
      std::cout << "Number of faces for the agglomeration: "
                << n_agglomerated_faces_per_cell << std::endl;
      for (unsigned int f = 0; f < n_agglomerated_faces_per_cell; ++f)
        {
          if (!polytope->at_boundary(f))
            {
              std::cout << "Agglomerated face with idx: " << f << std::endl;

              const auto &neigh_polytope = polytope->neighbor(f);
              const auto  vec_cells_and_faces =
                interface.at({polytope->id(), neigh_polytope->id()});
              for (const auto &cell_and_face : vec_cells_and_faces)
                {
                  std::cout << "deal.II cell idx: "
                            << cell_and_face.first->active_cell_index()
                            << std::endl;
                  std::cout << "deal.II face idx: " << cell_and_face.second
                            << std::endl;
                }
            }
          std::cout << std::endl;
        }
    }
}
