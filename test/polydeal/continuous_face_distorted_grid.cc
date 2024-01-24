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


// Same test of continuous_face.cc, but with a distorted grid.


#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include <agglomeration_handler.h>
#include <poly_utils.h>


void
perimeter_test(AgglomerationHandler<2> &ah)
{
  const auto &info      = ah.get_info();
  double      perimeter = 0.;
  for (const auto &cell : ah.agglomeration_cell_iterators() |
                            IteratorFilters::ActiveFEIndexEqualTo(
                              ah.CellAgglomerationType::master))
    {
      std::cout << "Master cell index = " << cell->active_cell_index()
                << std::endl;
      unsigned int n_faces = ah.n_agglomerated_faces(cell);
      std::cout << "Number of agglomerated faces = " << n_faces << std::endl;
      for (unsigned int f = 0; f < n_faces; ++f)
        {
          std::cout << "Agglomerate face index = " << f << std::endl;
          const auto &neighbor = ah.agglomerated_neighbor(cell, f);
          if (!ah.at_boundary(cell, f))
            {
              std::cout << "Neighbor = " << neighbor->active_cell_index()
                        << std::endl;
              std::cout << "Neighbor of neighbor = "
                        << ah.neighbor_of_agglomerated_neighbor(cell, f)
                        << std::endl;
            }
          else
            {
              const auto &test_feisv = ah.reinit(cell, f);
              perimeter += std::accumulate(test_feisv.get_JxW_values().begin(),
                                           test_feisv.get_JxW_values().end(),
                                           0.);
            }

          const auto &vec_tuple = info.at({cell, f});
          for (const auto &[deal_cell,
                            local_face_idx,
                            neighboring_master,
                            dummy] : vec_tuple)
            {
              std::cout << "deal.II cell index = "
                        << deal_cell->active_cell_index() << std::endl;
              std::cout << "Local face idx = " << local_face_idx << std::endl;
              if (neighboring_master.state() == IteratorState::valid)
                std::cout << "Neighboring master cell index = "
                          << neighboring_master->active_cell_index()
                          << std::endl;
            }
        }
      std::cout << std::endl;
    }
  std::cout << "Perimeter = " << perimeter << std::endl;
}



void
test_neighbors(AgglomerationHandler<2> &ah)
{
  std::cout << "Check on neighbors and neighbors of neighbors:" << std::endl;
  for (const auto &cell : ah.agglomeration_cell_iterators() |
                            IteratorFilters::ActiveFEIndexEqualTo(
                              ah.CellAgglomerationType::master))
    {
      unsigned int n_faces = ah.n_agglomerated_faces(cell);
      for (unsigned int f = 0; f < n_faces; ++f)
        {
          const auto &neighbor = ah.agglomerated_neighbor(cell, f);
          if (!ah.at_boundary(cell, f))
            {
              Assert((ah.agglomerated_neighbor(
                          neighbor,
                          ah.neighbor_of_agglomerated_neighbor(cell, f))
                        ->active_cell_index() == cell->active_cell_index()),
                     ExcMessage("Mismatch!"));
            }
        }
    }
  std::cout << "Ok" << std::endl;
}



void
test_face_qpoints(AgglomerationHandler<2> &ah)
{
  std::cout << "Check on quadrature points:" << std::endl;
  for (const auto &cell : ah.agglomeration_cell_iterators() |
                            IteratorFilters::ActiveFEIndexEqualTo(
                              ah.CellAgglomerationType::master))
    {
      unsigned int n_faces = ah.n_agglomerated_faces(cell);
      for (unsigned int f = 0; f < n_faces; ++f)
        {
          const auto &neigh_cell = ah.agglomerated_neighbor(cell, f);
          if (!ah.at_boundary(cell, f))
            {
              const unsigned int nofn =
                ah.neighbor_of_agglomerated_neighbor(cell, f);
              const auto &fe_faces =
                ah.reinit_interface(cell, neigh_cell, f, nofn);

              const auto &fe_faces0 = fe_faces.first;
              const auto &fe_faces1 = fe_faces.second;

              const auto &points0 = fe_faces0.get_quadrature_points();
              const auto &points1 = fe_faces1.get_quadrature_points();
              for (size_t i = 0; i < fe_faces1.get_quadrature_points().size();
                   ++i)
                {
                  double d = (points0[i] - points1[i]).norm();
                  Assert(d < 1e-15,
                         ExcMessage(
                           "Face qpoints at the interface do not match!"));
                }
            }
        }
    }
  std::cout << "Ok" << std::endl;
}



void
test0(const Triangulation<2> &tria, AgglomerationHandler<2> &ah)
{
  std::vector<types::global_cell_index> idxs_to_be_agglomerated = {
    0, 1, 2, 3, 4, 5, 6, 7};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated,
                                             cells_to_be_agglomerated);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated2 = {
    8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated2;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated2,
                                             cells_to_be_agglomerated2);

  // Agglomerate the cells just stored
  ah.insert_agglomerate(cells_to_be_agglomerated);
  ah.insert_agglomerate(cells_to_be_agglomerated2);
}



void
test1(const Triangulation<2> &tria, AgglomerationHandler<2> &ah)
{
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
  ah.insert_agglomerate(cells_to_be_agglomerated);
  ah.insert_agglomerate(cells_to_be_agglomerated2);
  ah.insert_agglomerate(cells_to_be_agglomerated3);
  ah.insert_agglomerate(cells_to_be_agglomerated4);
}

int
main()
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria, -1, 1);
  MappingQ<2> mapping(1);
  tria.refine_global(2);
  GridTools::distort_random(0.25, tria);
  GridTools::Cache<2>     cached_tria(tria, mapping);
  AgglomerationHandler<2> ah(cached_tria);

  test0(tria, ah);
  test1(tria, ah);

  FE_DGQ<2> fe_dg(1);
  ah.distribute_agglomerated_dofs(fe_dg);
  ah.initialize_fe_values(QGauss<2>(1),
                          update_JxW_values | update_quadrature_points);

  perimeter_test(ah);
  std::cout << "- - - - - - - - - - - -" << std::endl;
  test_neighbors(ah);
  std::cout << "- - - - - - - - - - - -" << std::endl;
  test_face_qpoints(ah);
}