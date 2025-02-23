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


// Check that volume and perimeter work properly also in case of a p::d::T. The
// configuration is:

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

static constexpr double TOL = 1e-15;


int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  const MPI_Comm                  &comm = MPI_COMM_WORLD;
  const unsigned n_ranks                = Utilities::MPI::n_mpi_processes(comm);
  AssertThrow(n_ranks == 3,
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

  double local_perimeter = 0.;
  double local_volume    = 0.;
  for (const auto &polytope : ah.polytope_iterators())
    {
      if (polytope->is_locally_owned())
        {
          const auto &agglo_values = ah.reinit(polytope);
          local_volume +=
            std::accumulate(agglo_values.get_JxW_values().cbegin(),
                            agglo_values.get_JxW_values().cend(),
                            0.);
          const auto        &q_points  = agglo_values.get_quadrature_points();
          const unsigned int n_qpoints = q_points.size();

          // Face terms
          const unsigned int n_faces = polytope->n_faces();
          AssertThrow(n_faces > 0, ExcMessage("Invalid element!"));

          for (unsigned int f = 0; f < n_faces; ++f)
            {
              if (polytope->at_boundary(f))
                {
                  const auto &test_feisv = ah.reinit(polytope, f);
                  local_perimeter +=
                    std::accumulate(test_feisv.get_JxW_values().begin(),
                                    test_feisv.get_JxW_values().end(),
                                    0.);
                } // at boundary

            } // Loop over faces of current cell
        }     // Loop over locally owned polytopes
    }


  double global_volume = 0;
  MPI_Reduce(&local_volume, &global_volume, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
  double global_perimeter = 0;
  MPI_Reduce(
    &local_perimeter, &global_perimeter, 1, MPI_DOUBLE, MPI_SUM, 0, comm);



  if (Utilities::MPI::this_mpi_process(comm) == 0)
    {
      Assert(std::fabs(global_volume - 1.) < TOL,
             ExcMessage("Wrong volume, it should be " + std::to_string(1) +
                        " but it is " + std::to_string(global_volume)));
      Assert(std::fabs(global_perimeter - 4.) < TOL,
             ExcMessage("Wrong perimeter, it should be " + std::to_string(4.) +
                        " but it is " + std::to_string(global_perimeter)));
      std::cout << "Ok" << std::endl;
    }
}