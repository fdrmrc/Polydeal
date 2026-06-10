// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) XXXX - YYYY by the polyDEAL authors
//
// This file is part of the polyDEAL library.
//
// Detailed license information governing the source code
// can be found in LICENSE.md at the top level directory.
//
// -----------------------------------------------------------------------------

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "continuous_agglo_utils.h"

using namespace dealii;

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const MPI_Comm     comm    = MPI_COMM_WORLD;
  const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(comm);
  const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);

  Assert(n_ranks == 3,
         ExcMessage("This test is meant to be run with 3 MPI ranks only."));

  if (my_rank == 0)
    std::cout << "Running distributed bounding box triangulation test with "
              << n_ranks << " MPI ranks." << std::endl;

  std::vector<BoundingBox<2>> local_boxes;
  const Point<2>              p_min(static_cast<double>(my_rank), 0.0);
  const Point<2>              p_max(static_cast<double>(my_rank) + 1.0, 1.0);
  local_boxes.emplace_back(std::make_pair(p_min, p_max));

  parallel::fullydistributed::Triangulation<2, 2> distributed_tria(comm);

  dealii::ContinuousAggloUtils::create_distributed_tria_from_local_boxes(
    distributed_tria, local_boxes, comm);

  assert(distributed_tria.n_global_active_cells() == n_ranks);
  assert(distributed_tria.n_locally_owned_active_cells() == local_boxes.size());

  for (const auto &cell : distributed_tria.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        const Point<2> center = cell->center();
        assert(std::abs(center[0] - (static_cast<double>(my_rank) + 0.5)) <
               1e-10);
        assert(std::abs(center[1] - 0.5) < 1e-10);
      }

  if (my_rank == 0)
    std::cout << "Distributed bounding box triangulation test passed!"
              << std::endl;

  return 0;
}