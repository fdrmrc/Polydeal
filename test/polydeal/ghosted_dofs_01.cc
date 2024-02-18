
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


// Check that we can get the DoFs of a ghosted polytope. The mesh is the
// following:
// ------------------------------------
// |                 |               |
// |       K5        |      K7       |
// |                 |               |
// ------------------------------------
// |                 |               |
// |       K4        |      K6       |
// |                 |               |
// ------------------------------------
// |                 |               |
// |       K1        |      K3       |
// |                 |               |
// ------------------------------------
// |                 |               |
// |       K0        |      K2       |
// |                 |               |
// ------------------------------------

// where:
// K0, K1 are owned by process 0
// K2, K3, K4, K5 are owned by process 1
// K6, K7 are owned by process 2

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
      std::vector<types::global_cell_index> idxs_to_be_agglomerated0 = {0, 1};
      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated0;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated0,
                                                 cells_to_be_agglomerated0);
      ah.define_agglomerate(cells_to_be_agglomerated0);

      std::vector<types::global_cell_index> idxs_to_be_agglomerated1 = {2, 3};
      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated1;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated1,
                                                 cells_to_be_agglomerated1);
      ah.define_agglomerate(cells_to_be_agglomerated1);
    }
  else if (my_rank == 1)
    {
      std::vector<types::global_cell_index> idxs_to_be_agglomerated0 = {4, 5};
      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated0;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated0,
                                                 cells_to_be_agglomerated0);
      ah.define_agglomerate(cells_to_be_agglomerated0);

      std::vector<types::global_cell_index> idxs_to_be_agglomerated1 = {6, 7};
      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated1;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated1,
                                                 cells_to_be_agglomerated1);
      ah.define_agglomerate(cells_to_be_agglomerated1);


      std::vector<types::global_cell_index> idxs_to_be_agglomerated2 = {8, 9};
      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated2;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated2,
                                                 cells_to_be_agglomerated2);
      ah.define_agglomerate(cells_to_be_agglomerated2);

      std::vector<types::global_cell_index> idxs_to_be_agglomerated3 = {10, 11};
      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated3;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated3,
                                                 cells_to_be_agglomerated3);
      ah.define_agglomerate(cells_to_be_agglomerated3);
    }
  else
    {
      std::vector<types::global_cell_index> idxs_to_be_agglomerated0 = {12, 13};
      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated0;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated0,
                                                 cells_to_be_agglomerated0);
      ah.define_agglomerate(cells_to_be_agglomerated0);

      std::vector<types::global_cell_index> idxs_to_be_agglomerated1 = {14, 15};
      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated1;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated1,
                                                 cells_to_be_agglomerated1);
      ah.define_agglomerate(cells_to_be_agglomerated1);
    }


  FE_DGQ<2> fe_dg(1);
  ah.distribute_agglomerated_dofs(fe_dg);


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

                  // If the neighborign polytope is not locally owned by the
                  // current process, get its BoundingBox and display the
                  // coordinates.
                  if (!neighbor_polytope->is_locally_owned())
                    {
                      std::vector<types::global_dof_index> ghosted_dofs(
                        fe_dg.dofs_per_cell);
                      neighbor_polytope->get_dof_indices(ghosted_dofs);

                      std::cout
                        << "DoFs indices from neighboring ghosted polytope:"
                        << std::endl;
                      for (const types::global_dof_index idx : ghosted_dofs)
                        std::cout << idx << std::endl;
                    }
                }
            }
        }
    }

  MPI_Barrier(comm);
  if (Utilities::MPI::this_mpi_process(comm) == 0)
    std::cout << "Ok" << std::endl;
}
