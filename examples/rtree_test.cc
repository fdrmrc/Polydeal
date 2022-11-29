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

#include <deal.II/base/bounding_box_data_out.h>
#include <deal.II/base/patterns.h>

#include <deal.II/boost_adaptors/point.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>

#include <algorithm>

using namespace dealii;

namespace bg  = boost::geometry;
namespace bgi = boost::geometry::index;

template <int dim, int spacedim>
void
test(const unsigned int ref, const unsigned int level)
{
  Triangulation<dim, spacedim> tria;
  GridGenerator::hyper_cube(tria, -1, 1);
  tria.refine_global(ref);
  GridTools::Cache<dim, spacedim> cache(tria);

  const auto &b_tree     = cache.get_cell_bounding_boxes_rtree();
  const auto &vec_bboxes = extract_rtree_level(b_tree, level);
  std::cout << "Number of BBOxes: " << vec_bboxes.size() << std::endl;
  {
    std::ofstream ofile("tria.vtu");
    GridOut().write_vtu(tria, ofile);
  }
  {
    std::ofstream ofile("boxes_level_" + std::to_string(level) + ".vtu");
    BoundingBoxDataOut<dim> data_out;
    data_out.build_patches(vec_bboxes);
    data_out.write_vtu(ofile);
  }

  std::vector<std::vector<types::global_cell_index>> vec_of_agglo_indices;
  std::vector<types::global_cell_index>              bbox_indices;
  for (const auto &bbox : vec_bboxes)
    {
      for (const auto &cell : tria.active_cell_iterators())
        {
          if (bbox.point_inside(cell->vertex(0)) &&
              bbox.point_inside(cell->vertex(3)))
            {
              bbox_indices.push_back(cell->active_cell_index());
              std::cout << "Idx " << cell->active_cell_index() << " is inside"
                        << std::endl;
            }
        }
      std::cout << std::endl;
      bbox_indices.clear();
    }
}

int
main()
{
  test<2, 2>(3, 1); // level=1
  test<2, 2>(3, 2);
}
