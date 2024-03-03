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


// Check that we can agglomerate locally owned regions of distribute grids.



#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/data_out.h>

#include <agglomeration_handler.h>
#include <poly_utils.h>


using namespace dealii;

template <int dim, int spacedim>
LinearAlgebra::distributed::Vector<double>
partition_distributed_triangulation(const Triangulation<dim, spacedim> &tria_in,
                                    const unsigned int n_partitions)
{
  const auto tria =
    dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(&tria_in);

  Assert(tria, ExcNotImplemented());

  LinearAlgebra::distributed::Vector<double> partition(
    tria->global_active_cell_index_partitioner().lock());

  for (const auto &cell :
       tria_in.active_cell_iterators() | IteratorFilters::LocallyOwnedCell())
    partition[cell->global_active_cell_index()] =
      std::floor(cell->center()[0] * n_partitions);

  partition.update_ghost_values();

  return partition;
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  const MPI_Comm &                 comm = MPI_COMM_WORLD;
  static constexpr unsigned int    dim  = 2;
  const unsigned n_ranks                = Utilities::MPI::n_mpi_processes(comm);
  AssertThrow(n_ranks == 3,
              ExcMessage("This test is meant to be run with 3 ranks only."));


  parallel::distributed::Triangulation<dim> tria(
    comm,
    Triangulation<dim>::none,
    parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy);
  GridGenerator::hyper_cube(tria);
  tria.refine_global(4);

  const auto partition_new = partition_distributed_triangulation(tria, n_ranks);

  // repartition triangulation so that it has strided partitioning
  const auto construction_data =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      tria, partition_new, TriangulationDescription::Settings::default_setting);

  parallel::fullydistributed::Triangulation<dim> tria_pft(comm);
  tria_pft.create_triangulation(construction_data);

  if (Utilities::MPI::this_mpi_process(comm) == 0)
    std::cout << "Running with " << n_ranks << " MPI ranks and "
              << tria_pft.n_global_active_cells() << " cells" << std::endl;
  const unsigned int n_agglomerates =
    10; // number of agglomerates within each local subdomain

  PolyUtils::partition_locally_owned_regions(n_agglomerates,
                                             tria_pft,
                                             SparsityTools::Partitioner::metis);

  std::cout << "Ok." << std::endl;
  // Uncomment the following to see the agglomerates
  /*if (Utilities::MPI::this_mpi_process(comm) == 1)
    for (const auto &cell : tria_pft.active_cell_iterators())
      if (cell->is_locally_owned())
        std::cout << "Cell index: " << cell->active_cell_index()
                  << " -> material_id: " << cell->material_id() << std::endl;*/



  /*DoFHandler<dim> dh(tria_pft);
  FE_DGQ<dim>     fe(0);
  dh.distribute_dofs(fe);
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dh);


  Vector<float> material_id(tria_pft.n_active_cells());
  unsigned int  i = 0;
  for (const auto &cell : tria_pft.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        material_id(cell->active_cell_index()) = cell->material_id();
    }


  data_out.add_data_vector(material_id, "material_id");

  data_out.build_patches();
  const std::string filename =
    ("test_pft_id." +
     Utilities::int_to_string(tria_pft.locally_owned_subdomain(), 4));
  std::ofstream output((filename + ".vtu").c_str());
  data_out.write_vtu(output);*/
}
