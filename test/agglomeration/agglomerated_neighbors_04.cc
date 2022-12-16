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

// Start from an external grid and agglomerate cells using graph partition from
// METIS library.

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include "../tests.h"

int
main()
{
  Triangulation<2> tria;
  unsigned int     n_partitions = 8;
  GridIn<2>        grid_in;
  grid_in.attach_triangulation(tria);
  std::ifstream input_file(SOURCE_DIR "/circle-grid.inp");
  grid_in.read_ucd(input_file);
  MappingQ<2> mapping(1);

  GridTools::partition_triangulation(n_partitions,
                                     tria,
                                     SparsityTools::Partitioner::metis);
  {
    std::ofstream out("test_circle");
    GridOut       grid_out;
    grid_out.write_mesh_per_processor_as_vtu(tria, "test_circle");
    std::cout << "Grid written " << std::endl;
  }
  GridTools::Cache<2>     cached_tria(tria, mapping);
  AgglomerationHandler<2> ah(cached_tria);

  std::multimap<types::global_cell_index,
                typename Triangulation<2>::active_cell_iterator>
    subdomain_2_vec;

  for (unsigned int subd_idx = 0; subd_idx < n_partitions; ++subd_idx)
    {
      for (const auto &cell : tria.active_cell_iterators())
        {
          if (cell->subdomain_id() == subd_idx)
            {
              subdomain_2_vec.insert({subd_idx, cell});
            }
        }
    }

  std::vector<typename Triangulation<2>::active_cell_iterator>
                                        cells_to_be_agglomerated;
  std::vector<types::global_cell_index> idxs_to_be_agglomerated;

  for (unsigned int subd_idx = 0; subd_idx < n_partitions; ++subd_idx)
    {
      auto range = subdomain_2_vec.equal_range(subd_idx);

      for (auto it = range.first; it != range.second; ++it)
        {
          idxs_to_be_agglomerated.push_back(it->second->active_cell_index());
        }
      Tests::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated,
                                             cells_to_be_agglomerated);

      ah.agglomerate_cells(cells_to_be_agglomerated);
      cells_to_be_agglomerated.clear();
      idxs_to_be_agglomerated.clear();
    }

  ah.print_agglomeration(std::cout);

  FE_DGQ<2> fe_dg(1);
  ah.distribute_agglomerated_dofs(fe_dg);
  for (const auto &cell : ah.agglomeration_cell_iterators() |
                            IteratorFilters::ActiveFEIndexEqualTo(
                              ah.CellAgglomerationType::master))
    {
      std::cout << "Cell with idx: " << cell->active_cell_index() << std::endl;
      const unsigned int n_faces = ah.n_faces(cell);
      std::cout << "Number of faces for this cell: " << n_faces << std::endl;
      for (unsigned int f = 0; f < n_faces; ++f)
        {
          std::cout << "Neighbor of (" << cell->active_cell_index() << "," << f
                    << ") = " << ah.agglomerated_neighbor(cell, f) << std::endl;
          std::cout << "Neighbor of neighbor of (" << cell->active_cell_index()
                    << "," << f
                    << ") = " << ah.neighbor_of_agglomerated_neighbor(cell, f)
                    << std::endl;

          if (!ah.at_boundary(cell, f))
            Assert(ah.agglomerated_neighbor(
                       ah.agglomerated_neighbor(cell, f),
                       ah.neighbor_of_agglomerated_neighbor(cell, f))
                       ->active_cell_index() == cell->active_cell_index(),
                   ExcMessage("Connectivity is not okay."));
        }


      std::cout << std::endl;
    }
}
