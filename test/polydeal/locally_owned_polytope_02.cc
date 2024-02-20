
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


// Check that neighbor() and neighbor_of_neighbor() works in the distributed
// case on the following example.
// ------------------------------------
// |                 |               |
// |                 |               |
// |       K2        |      K3       |
// |                 |               |
// ------------------------------------
// |                 |               |
// |       K0        |      K1       |
// |                 |               |
// |                 |               |
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
  tria.refine_global(2);

  GridTools::Cache<2>     cached_tria(tria);
  AgglomerationHandler<2> ah(cached_tria);

  unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);
  if (my_rank == 0)
    {
      std::vector<types::global_cell_index> idxs_to_be_agglomerated = {0,
                                                                       1,
                                                                       2,
                                                                       3};
      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated,
                                                 cells_to_be_agglomerated);
      ah.define_agglomerate(cells_to_be_agglomerated);
    }
  else if (my_rank == 1)
    {
      std::vector<types::global_cell_index> idxs_to_be_agglomerated = {4,
                                                                       5,
                                                                       6,
                                                                       7};
      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated,
                                                 cells_to_be_agglomerated);
      ah.define_agglomerate(cells_to_be_agglomerated);

      std::vector<types::global_cell_index> idxs_to_be_agglomerated2 = {8,
                                                                        9,
                                                                        10,
                                                                        11};
      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated2;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated2,
                                                 cells_to_be_agglomerated2);
      ah.define_agglomerate(cells_to_be_agglomerated2);
    }
  else
    {
      std::vector<types::global_cell_index> idxs_to_be_agglomerated = {12,
                                                                       13,
                                                                       14,
                                                                       15};
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
          // std::cout << "Master cell index = "
          //           << polytope.master_cell()->active_cell_index() <<
          //           std::endl;
          // std::cout << "Polytope index" << polytope->index() << std::endl;
          // std::cout << "Polytope id" << polytope->id() << std::endl;

          unsigned int n_faces = polytope->n_faces();
          // std::cout << "Number of agglomerated faces = " << n_faces
          //           << std::endl;
          for (unsigned int f = 0; f < n_faces; ++f)
            {
              // std::cout << "Agglomerate face index = " << f << std::endl;
              if (!polytope->at_boundary(f))
                {
                  const auto &neighbor_polytope = polytope->neighbor(f);
                  // std::cout
                  //   << "Neighbor polytope id = " << neighbor_polytope->id()
                  //   << std::endl;
                  const unsigned int nofn =
                    polytope->neighbor_of_agglomerated_neighbor(f);
                  // std::cout << "Neighbor of neighbor id " << nofn <<
                  // std::endl;


                  Assert((neighbor_polytope->neighbor(nofn)->id() ==
                          polytope->id()),
                         ExcMessage("Mismatch!"));

                  const auto &common_face =
                    interface.at({polytope->id(), neighbor_polytope->id()});

                  for (const auto &[deal_cell, local_face_idx] : common_face)
                    {
                      if (Utilities::MPI::this_mpi_process(comm) == 0)
                        {
                          std::cout << "deal.II cell index = "
                                    << deal_cell->active_cell_index()
                                    << std::endl;
                          std::cout << "Local face idx = " << local_face_idx
                                    << std::endl;
                        }
                    }
                }
            }
        }
    }

  MPI_Barrier(comm);
  if (Utilities::MPI::this_mpi_process(comm) == 0)
    std::cout << "Ok" << std::endl;
}