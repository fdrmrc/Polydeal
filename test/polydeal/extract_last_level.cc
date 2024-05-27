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


#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/reference_cell.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <agglomeration_handler.h>
#include <agglomerator.h>
#include <utils.h>


// Test we can extract also the first and last level of the tree.

enum class GridType
{
  grid_generator, // hyper_cube or hyper_ball
  unstructured    // square generated with gmsh, unstructured
};



template <int dim>
class TestExtractor
{
public:
  Triangulation<dim>                         tria;
  MappingQ<dim>                              mapping;
  std::unique_ptr<AgglomerationHandler<dim>> ah;
  std::unique_ptr<GridTools::Cache<dim>>     cached_tria;

  TestExtractor(const GridType &grid_type = GridType::grid_generator,
                const unsigned int        = 0);
  void
  run();

  GridType     grid_type;
  unsigned int extraction_level;
};



template <int dim>
TestExtractor<dim>::TestExtractor(const GridType    &grid_type,
                                  const unsigned int extraction_level)
  : mapping(1)
  , grid_type(grid_type)
  , extraction_level(extraction_level)
{}

template <int dim>
void
TestExtractor<dim>::run()
{
  GridIn<dim> grid_in;
  if (grid_type == GridType::unstructured)
    {
      if constexpr (dim == 2)
        {
          grid_in.attach_triangulation(tria);
          std::ifstream gmsh_file(
            "../../meshes/t3.msh"); // unstructured square [0,1]^2
          grid_in.read_msh(gmsh_file);
          tria.refine_global(5); // 4
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }
    }
  else
    {
      GridGenerator::hyper_cube(tria, 0., 1.);
      tria.refine_global(5);
    }
  std::cout << "Size of tria: " << tria.n_active_cells() << std::endl;
  cached_tria = std::make_unique<GridTools::Cache<dim>>(tria, mapping);

  // Partition with Rtree

  namespace bgi = boost::geometry::index;
  static constexpr unsigned int max_elem_per_node =
    Utils::constexpr_pow(2, dim); // 2^dim
  std::vector<std::pair<BoundingBox<dim>,
                        typename Triangulation<dim>::active_cell_iterator>>
               boxes(tria.n_active_cells());
  unsigned int i = 0;
  for (const auto &cell : tria.active_cell_iterators())
    boxes[i++] = std::make_pair(mapping.get_bounding_box(cell), cell);

  auto tree = pack_rtree<bgi::rstar<max_elem_per_node>>(boxes);
  std::cout << "Total number of available levels: " << n_levels(tree)
            << std::endl;

#ifdef AGGLO_DEBUG
  // boost::geometry::index::detail::rtree::utilities::print(std::cout,
  // tree);
  Assert(n_levels(tree) >= 2, ExcMessage("At least two levels are needed."));
#endif

  CellsAgglomerator<dim, decltype(tree)> agglomerator{tree, extraction_level};
  const auto agglomerates = agglomerator.extract_agglomerates();

  std::size_t agglo_index = 0;
  for (std::size_t i = 0; i < agglomerates.size(); ++i)
    {
      // std::cout << "AGGLO " + std::to_string(i) << std::endl;
      const auto &agglo = agglomerates[i];
      for (const auto &el : agglo)
        el->set_subdomain_id(agglo_index);
      ++agglo_index; // one agglomerate has been processed, increment
                     // counter
    }

  const unsigned int n_subdomains = agglo_index;

  std::cout << "N subdomains = " << n_subdomains << std::endl;

  for (unsigned int j = 0; j < n_subdomains; ++j)
    std::cout << GridTools::count_cells_with_subdomain_association(tria, j)
              << " cells are composing agglomerate " << j << std::endl;
}



int
main()
{
  for (unsigned int extraction_level : {0, 1, 2, 3, 4})
    {
      std::cout << "Extract level: " << extraction_level << std::endl;
      TestExtractor<2> problem{GridType::grid_generator, extraction_level};
      problem.run();
      std::cout << std::endl;
    }

  return 0;
}
