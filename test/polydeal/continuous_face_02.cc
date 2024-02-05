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


// Similar to continuous_face.cc, but with more complicated configurations.
// - - - - - - - - -|
// |      | K5 | K6 |
// |  K2  | - - - - |
// |      | K3 | K4 |
// |- - - - - - - - |
// |     K1         |
// - - - - - - - - -|



#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/data_out.h>

#include <agglomeration_handler.h>
#include <poly_utils.h>


void
perimeter_test(AgglomerationHandler<2> &ah)
{
  double perimeter = 0.;
  for (const auto &polytope : ah.polytope_iterators())
    {
      std::cout << "Master cell index = "
                << polytope.master_cell()->active_cell_index() << std::endl;
      const auto &info = ah.get_interface();

      unsigned int n_faces = polytope->n_faces();
      std::cout << "Number of agglomerated faces = " << n_faces << std::endl;
      for (unsigned int f = 0; f < n_faces; ++f)
        {
          std::cout << "Agglomerate face index = " << f << std::endl;
          if (!polytope->at_boundary(f))
            {
              const auto &neighbor = polytope->neighbor(f);
              std::cout << "Neighbor polytope index = " << neighbor->index()
                        << std::endl;
              std::cout << "Neighbor of neighbor = "
                        << polytope->neighbor_of_agglomerated_neighbor(f)
                        << std::endl;


              const auto &common_face =
                info.at({polytope->index(), neighbor->index()});

              for (const auto &[deal_cell, local_face_idx] : common_face)
                {
                  std::cout
                    << "deal.II cell index = " << deal_cell->active_cell_index()
                    << std::endl;
                  std::cout << "Local face idx = " << local_face_idx
                            << std::endl;
                  std::cout << "Neighboring master cell index = "
                            << neighbor.master_cell()->active_cell_index()
                            << std::endl;
                }
            }
          else
            {
              const auto &test_feisv = ah.reinit(polytope, f);
              perimeter += std::accumulate(test_feisv.get_JxW_values().begin(),
                                           test_feisv.get_JxW_values().end(),
                                           0.);
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
  for (const auto &polytope : ah.polytope_iterators())
    {
      unsigned int n_faces = polytope->n_faces();
      for (unsigned int f = 0; f < n_faces; ++f)
        {
          if (!polytope->at_boundary(f))
            {
              const auto &neighbor_polytope = polytope->neighbor(f);
              AssertThrow(neighbor_polytope
                              ->neighbor(
                                polytope->neighbor_of_agglomerated_neighbor(f))
                              ->index() == polytope->index(),
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
  for (const auto &polytope : ah.polytope_iterators())
    {
      unsigned int n_faces = polytope->n_faces();
      for (unsigned int f = 0; f < n_faces; ++f)
        {
          if (!polytope->at_boundary(f))
            {
              const auto &       neigh_polytope = polytope->neighbor(f);
              const unsigned int nofn =
                polytope->neighbor_of_agglomerated_neighbor(f);
              const auto &fe_faces =
                ah.reinit_interface(polytope, neigh_polytope, f, nofn);

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

  std::vector<types::global_cell_index> idxs_to_be_agglomerated2 = {8,
                                                                    9,
                                                                    10,
                                                                    11};
  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated2;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated2,
                                             cells_to_be_agglomerated2);


  std::vector<types::global_cell_index> idxs_to_be_agglomerated3 = {12};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated3;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated3,
                                             cells_to_be_agglomerated3);


  std::vector<types::global_cell_index> idxs_to_be_agglomerated4 = {13};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated4;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated4,
                                             cells_to_be_agglomerated4);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated5 = {14};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated5;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated5,
                                             cells_to_be_agglomerated5);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated6 = {15};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated6;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated6,
                                             cells_to_be_agglomerated6);

  // Agglomerate the cells just stored
  ah.define_agglomerate(cells_to_be_agglomerated);
  ah.define_agglomerate(cells_to_be_agglomerated2);
  ah.define_agglomerate(cells_to_be_agglomerated3);
  ah.define_agglomerate(cells_to_be_agglomerated4);
  ah.define_agglomerate(cells_to_be_agglomerated5);
  ah.define_agglomerate(cells_to_be_agglomerated6);
}



void
test1(const Triangulation<2> &tria, AgglomerationHandler<2> &ah)
{
  std::vector<types::global_cell_index> idxs_to_be_agglomerated = {
    0, 1, 2, 3, 4, 5, 6, 7};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated,
                                             cells_to_be_agglomerated);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated2 = {8,
                                                                    9,
                                                                    10,
                                                                    11};
  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated2;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated2,
                                             cells_to_be_agglomerated2);


  std::vector<types::global_cell_index> idxs_to_be_agglomerated3 = {12,
                                                                    13,
                                                                    14,
                                                                    15};
  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated3;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated3,
                                             cells_to_be_agglomerated3);

  // Agglomerate the cells just stored
  ah.define_agglomerate(cells_to_be_agglomerated);
  ah.define_agglomerate(cells_to_be_agglomerated2);
  ah.define_agglomerate(cells_to_be_agglomerated3);
}



void
test2(const Triangulation<2> &tria, AgglomerationHandler<2> &ah)
{
  std::vector<types::global_cell_index> idxs_to_be_agglomerated = {3, 6, 9, 12};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated,
                                             cells_to_be_agglomerated);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated2 = {0, 1, 4, 5};
  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated2;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated2,
                                             cells_to_be_agglomerated2);


  std::vector<types::global_cell_index> idxs_to_be_agglomerated3 = {2, 8, 10};
  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated3;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated3,
                                             cells_to_be_agglomerated3);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated4 = {11, 14, 15};
  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated4;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated4,
                                             cells_to_be_agglomerated4);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated5 = {7, 13};
  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated5;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated5,
                                             cells_to_be_agglomerated5);

  // Agglomerate the cells just stored
  ah.define_agglomerate(cells_to_be_agglomerated);
  ah.define_agglomerate(cells_to_be_agglomerated2);
  ah.define_agglomerate(cells_to_be_agglomerated3);
  ah.define_agglomerate(cells_to_be_agglomerated4);
  ah.define_agglomerate(cells_to_be_agglomerated5);
}



void
test3(const Triangulation<2> & tria,
      AgglomerationHandler<2> &ah,
      const unsigned int       n_subdomains)
{
  std::vector<std::vector<typename Triangulation<2>::active_cell_iterator>>
    cells_per_subdomain(n_subdomains);
  for (const auto &cell : tria.active_cell_iterators())
    cells_per_subdomain[cell->subdomain_id()].push_back(cell);

  for (std::size_t i = 0; i < cells_per_subdomain.size(); ++i)
    ah.define_agglomerate(cells_per_subdomain[i]);
}

int
main()
{
  {
    Triangulation<2> tria;
    GridGenerator::hyper_cube(tria, -1, 1);
    MappingQ<2> mapping(1);
    tria.refine_global(2);
    GridTools::Cache<2>     cached_tria(tria, mapping);
    AgglomerationHandler<2> ah(cached_tria);
    FE_DGQ<2>               fe_dg(1);

    test0(tria, ah);

    ah.distribute_agglomerated_dofs(fe_dg);
    ah.initialize_fe_values(QGauss<2>(1),
                            update_JxW_values | update_quadrature_points);

    perimeter_test(ah);
    std::cout << "- - - - - - - - - - - -" << std::endl;
    test_neighbors(ah);
    std::cout << "- - - - - - - - - - - -" << std::endl;
    test_face_qpoints(ah);
    std::cout << "End Test0" << std::endl;
  }

  {
    Triangulation<2> tria;
    GridGenerator::hyper_cube(tria, -1, 1);
    MappingQ<2> mapping(1);
    tria.refine_global(2);
    GridTools::Cache<2>     cached_tria(tria, mapping);
    AgglomerationHandler<2> ah(cached_tria);
    FE_DGQ<2>               fe_dg(1);

    test1(tria, ah);

    ah.distribute_agglomerated_dofs(fe_dg);
    ah.initialize_fe_values(QGauss<2>(1),
                            update_JxW_values | update_quadrature_points);

    perimeter_test(ah);
    std::cout << "- - - - - - - - - - - -" << std::endl;
    test_neighbors(ah);
    std::cout << "- - - - - - - - - - - -" << std::endl;
    test_face_qpoints(ah);
    std::cout << "End Test1" << std::endl;
  }

  {
    Triangulation<2> tria;
    GridGenerator::hyper_cube(tria, -1, 1);
    MappingQ<2> mapping(1);
    tria.refine_global(2);
    GridTools::Cache<2>     cached_tria(tria, mapping);
    AgglomerationHandler<2> ah(cached_tria);
    FE_DGQ<2>               fe_dg(1);

    test2(tria, ah);

    ah.distribute_agglomerated_dofs(fe_dg);
    ah.initialize_fe_values(QGauss<2>(1),
                            update_JxW_values | update_quadrature_points);

    perimeter_test(ah);
    std::cout << "- - - - - - - - - - - -" << std::endl;
    test_neighbors(ah);
    std::cout << "- - - - - - - - - - - -" << std::endl;
    test_face_qpoints(ah);
    std::cout << "End Test2" << std::endl;
  }

  {
    Triangulation<2> tria;
    GridGenerator::hyper_cube(tria, -1, 1);
    MappingQ<2> mapping(1);
    tria.refine_global(3);

    const unsigned int n_subdomains = 10;

    GridTools::partition_triangulation(n_subdomains,
                                       tria,
                                       SparsityTools::Partitioner::metis);

    GridTools::Cache<2>     cached_tria(tria, mapping);
    AgglomerationHandler<2> ah(cached_tria);
    FE_DGQ<2>               fe_dg(1);

    test3(tria, ah, n_subdomains);


    ah.distribute_agglomerated_dofs(fe_dg);
    ah.initialize_fe_values(QGauss<2>(1),
                            update_JxW_values | update_quadrature_points);

    perimeter_test(ah);
    std::cout << "- - - - - - - - - - - -" << std::endl;
    test_neighbors(ah);
    std::cout << "- - - - - - - - - - - -" << std::endl;
    test_face_qpoints(ah);
    std::cout << "End Test3" << std::endl;
  }
}
