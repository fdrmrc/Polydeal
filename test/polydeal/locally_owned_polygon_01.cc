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


// Check that polygons can be locally owned.


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
  if (Utilities::MPI::this_mpi_process(comm) == 0)
    std::cout << "Running with " << Utilities::MPI::n_mpi_processes(comm)
              << " MPI ranks." << std::endl;

  parallel::distributed::Triangulation<2> tria(comm);

  GridGenerator::hyper_cube(tria, -1, 1);
  tria.refine_global(2);

  GridTools::Cache<2>     cached_tria(tria);
  AgglomerationHandler<2> ah(cached_tria);

  for (const auto &cell : tria.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        ah.insert_agglomerate({cell});
    }

  unsigned int polygons_per_rank = 0;

  for (const auto &polygon : ah.polytope_iterators())
    {
      if (polygon->is_locally_owned())
        ++polygons_per_rank;
    }
  MPI_Send(&polygons_per_rank, 1, MPI_UNSIGNED, 0, 0, comm);

  if (Utilities::MPI::this_mpi_process(comm) == 0)
    {
      for (unsigned int p = 0; p < n_ranks; ++p)
        {
          unsigned int polygons_from_rank_p = 0;
          MPI_Recv(&polygons_from_rank_p,
                   1,
                   MPI_UNSIGNED,
                   p,
                   0,
                   comm,
                   MPI_STATUS_IGNORE);
          std::cout << "Rank " << p << " owns " << polygons_from_rank_p
                    << " polygons." << std::endl;
        }
    }
}
