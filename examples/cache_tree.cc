// ---------------------------------------------------------------------
//
// Copyright (C) 2018 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

// Extract cell bounding boxes rtree from the cache, and try to use it


#include <deal.II/base/bounding_box.h>
#include <deal.II/base/bounding_box_data_out.h>

#include <deal.II/boost_adaptors/point.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>

#include <agglomeration_handler.h>

#include <algorithm>

#include "../tests.h"

using namespace dealii;
namespace bg  = boost::geometry;
namespace bgi = boost::geometry::index;



template <int dim, int spacedim, unsigned int max_elem_per_node>
void
test(const unsigned int ref = 6, const unsigned int extraction_level = 0)
{
  Triangulation<dim, spacedim> tria;
  MappingQ<dim>                mapping(1);
  GridGenerator::hyper_ball(tria);
  tria.refine_global(ref);
  GridTools::Cache<dim, spacedim> cache(tria);
  AgglomerationHandler<dim>       ah(cache);

  std::vector<
    std::pair<BoundingBox<spacedim>,
              typename Triangulation<dim, spacedim>::active_cell_iterator>>
               boxes(tria.n_active_cells());
  unsigned int i = 0;
  for (const auto &cell : tria.active_cell_iterators())
    boxes[i++] = std::make_pair(mapping.get_bounding_box(cell), cell);

  const auto tree = pack_rtree<bgi::rstar<max_elem_per_node>>(boxes);
  // const auto tree = pack_rtree<bgi::linear<max_elem_per_node>>(boxes);

  const auto vec_boxes = extract_rtree_level(tree, extraction_level);

  std::vector<BoundingBox<spacedim>> bboxes;
  for (const auto &box : vec_boxes)
    bboxes.push_back(box);

  i = 0;
  std::vector<
    std::pair<BoundingBox<spacedim>,
              typename Triangulation<dim, spacedim>::active_cell_iterator>>
    cells;
  std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>
                                        cells_to_agglomerate;
  std::vector<types::global_cell_index> idxs_to_agglomerate;
  for (const auto &box : vec_boxes)
    {
      tree.query(bgi::within(box), std::back_inserter(cells));
      std::cout << "Number of cells inside " << std::to_string(i)
                << "-th bounding box: " << cells.size() << std::endl;
      for (const auto &my_pair : cells)
        {
          std::cout << my_pair.second->active_cell_index() << std::endl;
          idxs_to_agglomerate.push_back(my_pair.second->active_cell_index());
        }

      Tests::collect_cells_for_agglomeration(tria,
                                             idxs_to_agglomerate,
                                             cells_to_agglomerate);
      ah.agglomerate_cells(cells_to_agglomerate);
      cells.clear();
      cells_to_agglomerate.clear();
      idxs_to_agglomerate.clear();
      ++i;
    }

  {
    std::ofstream           ofile("bboxes.vtu");
    BoundingBoxDataOut<dim> data_out;
    data_out.build_patches(bboxes);
    data_out.write_vtu(ofile);
    std::cout << "BBoxes written " << std::endl;

    std::ofstream out("grid_ball.vtk");
    GridOut       grid_out;
    grid_out.write_vtk(tria, out);
    std::cout << "Grid written " << std::endl;

    ah.print_agglomeration(std::cout);
  }

  unsigned int n_master_cells = 0;
  for (const auto &cell : tria.active_cell_iterators())
    {
      if (ah.is_master_cell(cell))
        n_master_cells++;
    }
  std::cout << "REFINEMENTS: " << ref << std::endl;
  std::cout << "LEVEL " + std::to_string(extraction_level)
            << " has N boxes = " << vec_boxes.size() << std::endl;
  std::cout << "NUMBER OF TOTAL CELLS: " << tria.n_active_cells() << std::endl;
  std::cout << "Number of master cells: " << n_master_cells << std::endl;

  FE_DGQ<dim, spacedim> dg_fe(1);
  ah.distribute_agglomerated_dofs(dg_fe);


  for (const auto &cell : ah.agglomeration_cell_iterators())
    {
      const unsigned int n_faces = ah.n_faces(cell);
      std::cout << cell->active_cell_index() << " has " << n_faces << std::endl;
      for (unsigned int f = 0; f < n_faces; ++f)
        {
          const auto &neigh_cell = ah.agglomerated_neighbor(cell, f);
        }
    }
}

int
main()
{
  // const unsigned int            extraction_levels  = 14;
  const unsigned int            global_refinements = 4;
  static constexpr unsigned int max_per_node       = 16;
  // for (unsigned int i = 0; i < extraction_levels; ++i)
  //   test<2, 2>(global_refinements, i);
  test<2, 2, max_per_node>(global_refinements, 1);
}
