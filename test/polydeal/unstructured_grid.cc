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

// Compute the perimeter of a polytope arising from the agglomeration of some
// elements in unstructured grids.

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <agglomeration_handler.h>
#include <poly_utils.h>

template <int dim>
void
test_internal_grid(Triangulation<dim> &tria)
{
  GridGenerator::hyper_ball(tria);
  MappingQ<2> mapping(1);
  tria.refine_global(1);
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

  std::vector<types::global_cell_index> idxs_to_be_agglomerated = {8, 5};
  store_flagged_cells(idxs_to_be_agglomerated);

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated,
                                             cells_to_be_agglomerated);
  // For debugging purposes only
  // double sum = 0.;
  // for (const auto &cell : tria.active_cell_iterators())
  //   {
  //     if (cell->active_cell_index() == 8 || cell->active_cell_index() == 5)
  //       {
  //         for (unsigned int f = 0; f < 4; ++f)
  //           {
  //             sum += cell->face(f)->measure();
  //           }
  //       }
  //   }

  // Agglomerate the cells just stored
  ah.define_agglomerate(cells_to_be_agglomerated);
  for (std::size_t i = 0; i < tria.n_active_cells(); ++i)
    {
      // If not present, agglomerate all the singletons
      if (std::find(flagged_cells.begin(),
                    flagged_cells.end(),
                    cells[i]->active_cell_index()) == std::end(flagged_cells))
        ah.define_agglomerate({cells[i]});
    }


  FE_DGQ<2> fe_dg(1);
  ah.distribute_agglomerated_dofs(fe_dg);
  ah.initialize_fe_values(QGauss<2>(1), update_JxW_values);
  double perimeter = 0.;
  for (const auto &cell : ah.polytope_iterators())
    {
      unsigned int n_agglomerated_faces_per_cell = cell->n_faces();
      if (n_agglomerated_faces_per_cell == 5)
        {
          std::cout << "Number of faces of this cell: "
                    << n_agglomerated_faces_per_cell << std::endl;
          for (size_t f = 0; f < n_agglomerated_faces_per_cell; ++f)
            {
              const auto &test_feisv = ah.reinit(cell, f);
              const auto &normals    = test_feisv.get_normal_vectors();
              perimeter += std::accumulate(test_feisv.get_JxW_values().begin(),
                                           test_feisv.get_JxW_values().end(),
                                           0.);
              std::cout << "For face with index f =" << f << " the normal is "
                        << normals[0] << std::endl;
            }
          std::cout << "Perimeter of agglomeration with index " << cell->index()
                    << " is " << perimeter << std::endl;
          perimeter = 0.;
        }
    }
}



template <int dim>
void
test_external_grid(Triangulation<2> &tria)
{
  GridIn<dim> grid_in;
  grid_in.attach_triangulation(tria);
  std::ifstream input_file(SOURCE_DIR "/circle-grid.inp");
  grid_in.read_ucd(input_file);
  tria.refine_global(1);
  MappingQ<2>             mapping(1);
  GridTools::Cache<2>     cached_tria(tria, mapping);
  AgglomerationHandler<2> ah(cached_tria);

  // {
  //   std::ofstream out("test_circle.vtk");
  //   GridOut       grid_out;
  //   grid_out.write_vtk(tria, out);
  //   std::cout << "Grid written " << std::endl;
  // }

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

  std::vector<types::global_cell_index> idxs_to_be_agglomerated = {25, 44};
  store_flagged_cells(idxs_to_be_agglomerated);

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated,
                                             cells_to_be_agglomerated);


  // Agglomerate the cells just stored
  ah.define_agglomerate(cells_to_be_agglomerated);

  for (std::size_t i = 0; i < tria.n_active_cells(); ++i)
    {
      // If not present, agglomerate all the singletons
      if (std::find(flagged_cells.begin(),
                    flagged_cells.end(),
                    cells[i]->active_cell_index()) == std::end(flagged_cells))
        ah.define_agglomerate({cells[i]});
    }

  FE_DGQ<2> fe_dg(1);
  ah.distribute_agglomerated_dofs(fe_dg);
  ah.initialize_fe_values(QGauss<2>(1), update_JxW_values);
  double perimeter = 0.;
  for (const auto &cell : ah.polytope_iterators())
    {
      unsigned int n_agglomerated_faces_per_cell = cell->n_faces();
      if (n_agglomerated_faces_per_cell == 6)
        {
          std::cout << "Number of faces of this cell: "
                    << n_agglomerated_faces_per_cell << std::endl;
          for (size_t f = 0; f < n_agglomerated_faces_per_cell; ++f)
            {
              const auto &test_feisv = ah.reinit(cell, f);
              const auto &normals    = test_feisv.get_normal_vectors();
              perimeter += std::accumulate(test_feisv.get_JxW_values().begin(),
                                           test_feisv.get_JxW_values().end(),
                                           0.);
              std::cout << "For face with index f =" << f << " the normal is "
                        << normals[0] << std::endl;
            }
          std::cout << "Perimeter of agglomeration index: " << cell->index()
                    << " is " << perimeter << std::endl;
          perimeter = 0.;
        }
    }
}

int
main()
{
  Triangulation<2> tria;
  test_external_grid<2>(tria);
  tria.clear();
  test_internal_grid<2>(tria);
}
