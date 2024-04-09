
/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2022 by the polyDEAL authors
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


// Check that you can query quadrature points and JxWs in the following case
// ------------------------------------
// |           |                    |
// |           |---|                |
// |       K2      |      K3        |
// |               |                |
// ------------------------------------
// |               |                |
// |       K0      |- -      K1     |
// |                   |            |
// |                   |            |
// ------------------------------------

// where:
// K0 is owned by process 0
// K1, K2 are owned by process 1
// K3 is owned by process 2

#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <agglomeration_handler.h>
#include <poly_utils.h>



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  const MPI_Comm &                 comm = MPI_COMM_WORLD;
  const unsigned n_ranks                = Utilities::MPI::n_mpi_processes(comm);
  Assert(n_ranks == 3,
         ExcMessage("This test is meant to be run with 3 ranks only."));
  if (Utilities::MPI::this_mpi_process(comm) == 0)
    std::cout << "Running with " << n_ranks << " MPI ranks." << std::endl;

  parallel::distributed::Triangulation<2> tria(comm);

  GridGenerator::hyper_cube(tria);
  tria.refine_global(3);

  GridTools::Cache<2>     cached_tria(tria);
  AgglomerationHandler<2> ah(cached_tria);

  unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);

  // Helper lambda to store the indices of all the cells that are locally owned
  const auto &get_all_locally_owned_indices = [&tria]() {
    std::vector<types::global_cell_index> local_indices;
    for (const auto &cell : tria.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          local_indices.push_back(cell->active_cell_index());
      }
    return local_indices;
  };

  if (my_rank == 0)
    {
      std::vector<types::global_cell_index> idxs_to_be_agglomerated0 = {5,
                                                                        6,
                                                                        7,
                                                                        8};
      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated0;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated0,
                                                 cells_to_be_agglomerated0);
      ah.define_agglomerate(cells_to_be_agglomerated0);

      std::vector<types::global_cell_index> idxs_to_be_agglomerated1 = {9,
                                                                        10,
                                                                        11,
                                                                        12};
      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated1;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated1,
                                                 cells_to_be_agglomerated1);
      ah.define_agglomerate(cells_to_be_agglomerated1);

      std::vector<types::global_cell_index> idxs_to_be_agglomerated2 = {21,
                                                                        22,
                                                                        23,
                                                                        24};
      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated2;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated2,
                                                 cells_to_be_agglomerated2);
      ah.define_agglomerate(cells_to_be_agglomerated2);

      std::vector<types::global_cell_index> idxs_to_be_agglomerated3 = {13,
                                                                        14,
                                                                        15,
                                                                        16};
      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated3;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated3,
                                                 cells_to_be_agglomerated3);
      ah.define_agglomerate(cells_to_be_agglomerated3);

      std::vector<types::global_cell_index> idxs_to_be_agglomerated4 = {17,
                                                                        18,
                                                                        19,
                                                                        20};
      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated4;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated4,
                                                 cells_to_be_agglomerated4);
      ah.define_agglomerate(cells_to_be_agglomerated4);
    }
  else if (my_rank == 1)
    {
      std::vector<types::global_cell_index> idxs_to_be_agglomerated0 = {18,
                                                                        19,
                                                                        20,
                                                                        21};
      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated0;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated0,
                                                 cells_to_be_agglomerated0);
      ah.define_agglomerate(cells_to_be_agglomerated0);

      std::vector<types::global_cell_index> idxs_to_be_agglomerated1 = {22,
                                                                        23,
                                                                        24,
                                                                        25};
      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated1;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated1,
                                                 cells_to_be_agglomerated1);
      ah.define_agglomerate(cells_to_be_agglomerated1);

      std::vector<types::global_cell_index> idxs_to_be_agglomerated2 = {26,
                                                                        27,
                                                                        28,
                                                                        29};
      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated2;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated2,
                                                 cells_to_be_agglomerated2);
      ah.define_agglomerate(cells_to_be_agglomerated2);

      std::vector<types::global_cell_index> idxs_to_be_agglomerated3 = {30,
                                                                        31,
                                                                        32,
                                                                        33};
      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated3;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated3,
                                                 cells_to_be_agglomerated3);
      ah.define_agglomerate(cells_to_be_agglomerated3);

      std::vector<types::global_cell_index> idxs_to_be_agglomerated4 = {34,
                                                                        35,
                                                                        36,
                                                                        37};
      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated4;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated4,
                                                 cells_to_be_agglomerated4);
      ah.define_agglomerate(cells_to_be_agglomerated4);

      std::vector<types::global_cell_index> idxs_to_be_agglomerated5 = {38,
                                                                        39,
                                                                        40,
                                                                        41};
      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated5;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated5,
                                                 cells_to_be_agglomerated5);
      ah.define_agglomerate(cells_to_be_agglomerated5);
    }
  else
    {
      const auto &idxs_to_be_agglomerated = get_all_locally_owned_indices();

      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated,
                                                 cells_to_be_agglomerated);
      ah.define_agglomerate(cells_to_be_agglomerated);
    }

  FE_DGQ<2>          fe_dg(0);
  const unsigned int quadrature_degree      = 2 * fe_dg.get_degree() + 1;
  const unsigned int face_quadrature_degree = 2 * fe_dg.get_degree() + 1;
  ah.initialize_fe_values(QGauss<2>(quadrature_degree),
                          update_gradients | update_JxW_values |
                            update_quadrature_points | update_JxW_values |
                            update_values,
                          QGauss<1>(face_quadrature_degree));
  ah.distribute_agglomerated_dofs(fe_dg);
  const auto &interface = ah.get_interface();


  auto polytope = ah.begin();
  for (; polytope != ah.end(); ++polytope)
    {
      if (polytope->is_locally_owned())
        {
          unsigned int n_faces = polytope->n_faces();

          for (unsigned int f = 0; f < n_faces; ++f)
            {
              if (!polytope->at_boundary(f))
                {
                  const auto &neighbor_polytope = polytope->neighbor(f);

                  const unsigned int nofn =
                    polytope->neighbor_of_agglomerated_neighbor(f);

                  Assert((neighbor_polytope->neighbor(nofn)->id() ==
                          polytope->id()),
                         ExcMessage("Mismatch!"));


                  if (polytope->id() < neighbor_polytope->id())
                    {
                      // check only with ghosted neighbors.
                      if (!neighbor_polytope->is_locally_owned())
                        {
                          unsigned int nofn =
                            polytope->neighbor_of_agglomerated_neighbor(f);

                          const auto &fe_faces = ah.reinit_interface(
                            polytope, neighbor_polytope, f, nofn);
                          const auto &fe_faces0 = fe_faces.first;

                          types::subdomain_id neigh_rank =
                            neighbor_polytope->subdomain_id();
                          const auto &test_vec =
                            ah.recv_qpoints.at(neigh_rank)
                              .at({neighbor_polytope->id(), nofn});

                          const auto &test_jxws =
                            ah.recv_jxws.at(neigh_rank)
                              .at({neighbor_polytope->id(), nofn});


                          // Check on qpoints
                          const auto &points0 =
                            fe_faces0.get_quadrature_points();
                          const auto &points1 = test_vec;
                          for (size_t i = 0; i < points0.size(); ++i)
                            {
                              double d = (points0[i] - points1[i]).norm();
                              AssertThrow(
                                d < 1e-15,
                                ExcMessage(
                                  "Face qpoints at the interface do not match!"));
                            }


                          // Check on JxWs
                          const auto &jxws0 = fe_faces0.get_JxW_values();
                          const auto &jxws1 = test_jxws;
                          for (size_t i = 0; i < jxws0.size(); ++i)
                            {
                              double d = (jxws0[i] - jxws1[i]);
                              AssertThrow(
                                std::fabs(d) < 1e-15,
                                ExcMessage(
                                  "JxWs at the interface do not match!"));
                            }


                        } // check only with ghosted neighbors.
                    }     // only once
                }         // not on boundary
            }             // every face
        }                 // locally owned
    }

  MPI_Barrier(comm);
  if (Utilities::MPI::this_mpi_process(comm) == 0)
    std::cout << "Ok" << std::endl;
}