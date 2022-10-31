#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <algorithm>

#include "../tests.h"

#include "../include/agglomeration_handler.h"

int
main()
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria, -1, 1);

  tria.refine_global(3);
  MappingQ<2>             mapping(1);
  GridTools::Cache<2>     cached_tria(tria, mapping);
  AgglomerationHandler<2> ah(cached_tria);

  std::vector<unsigned int> idxs_to_be_agglomerated = {3, 6, 9, 12, 13};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated;

  Tests::collect_cells_for_agglomeration(tria,
                                         idxs_to_be_agglomerated,
                                         cells_to_be_agglomerated);

  std::vector<unsigned int> idxs_to_be_agglomerated2 = {15, 36, 37};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated2;
  Tests::collect_cells_for_agglomeration(tria,
                                         idxs_to_be_agglomerated2,
                                         cells_to_be_agglomerated2);

  std::vector<unsigned int> idxs_to_be_agglomerated3 = {57, 60, 54};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated3;
  Tests::collect_cells_for_agglomeration(tria,
                                         idxs_to_be_agglomerated3,
                                         cells_to_be_agglomerated3);

  std::vector<unsigned int> idxs_to_be_agglomerated4 = {25, 19, 22};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated4;
  Tests::collect_cells_for_agglomeration(tria,
                                         idxs_to_be_agglomerated4,
                                         cells_to_be_agglomerated4);

  // Agglomerate the cells just stored
  ah.agglomerate_cells(cells_to_be_agglomerated);
  ah.agglomerate_cells(cells_to_be_agglomerated2);
  ah.agglomerate_cells(cells_to_be_agglomerated3);
  ah.agglomerate_cells(cells_to_be_agglomerated4);
  ah.initialize_hp_structure();
  ah.setup_connectivity_of_agglomeration();

  for (const auto &cell : ah.agglo_dh.active_cell_iterators())
    {
      const unsigned int agglo_faces_per_cell =
        ah.n_agglomerated_faces_per_cell(cell);
      std::cout << "Number of agglomerated faces for cell "
                << cell->active_cell_index() << " is " << agglo_faces_per_cell
                << std::endl;
    }

  std::ofstream out("snippet_grid.vtk");
  GridOut       grid_out;
  grid_out.write_vtk(tria, out);


  const auto test_quad_over_agglomerated =
    ah.agglomerated_quadrature(cells_to_be_agglomerated, QGauss<2>(1));
  for (const auto &cell :
       ah.agglo_dh.active_cell_iterators() |
         IteratorFilters::ActiveFEIndexEqualTo(ah.AggloIndex::master))
    {
      std::cout << "Cell with idx: " << cell->active_cell_index() << std::endl;
      unsigned int n_agglomerated_faces_per_cell =
        ah.n_agglomerated_faces_per_agglomeration(cell);
      std::cout << "Agglo faces: " << n_agglomerated_faces_per_cell
                << std::endl;
      for (unsigned int f = 0; f < n_agglomerated_faces_per_cell; ++f)
        {
          std::cout << "Agglomerated face with idx: " << f << std::endl;
          const auto &info_about_neighbors = ah.master_neighbors[{cell, f}];
          const auto &test_feisv           = ah.reinit(cell, f);
          double      sum                  = 0.;
          for (const auto &w : test_feisv.get_JxW_values())
            sum += w;
          std::cout << sum << std::endl;
          sum = 0.;
          std::cout << "Face idx: " << std::get<0>(info_about_neighbors)
                    << std::endl;
          std::cout << "Neighbor idx: "
                    << std::get<1>(info_about_neighbors)->active_cell_index()
                    << std::endl;
          std::cout << "Face idx from outside "
                    << std::get<2>(info_about_neighbors) << std::endl;
        }
      return 0;
    }
}
