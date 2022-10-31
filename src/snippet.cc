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


  for (const auto &cell :
       ah.agglo_dh.active_cell_iterators() |
         IteratorFilters::ActiveFEIndexEqualTo(ah.AggloIndex::master))
    {
      std::cout << "Cell with idx: " << cell->active_cell_index() << std::endl;
      ah.setup_master_neighbor_connectivity(cell);
      unsigned int n_agglomerated_faces_per_cell =
        ah.n_agglomerated_faces_per_agglomeration(cell);
      std::cout << "Agglo faces: " << n_agglomerated_faces_per_cell
                << std::endl;
      for (unsigned int f = 0; f < n_agglomerated_faces_per_cell; ++f)
        {
          std::cout << "Agglomerated face with idx: " << f << std::endl;
          auto my_value = ah.master_neighbors[{cell, f}];
          for (const auto &t : my_value)
            {
              std::cout << "Face idx: " << std::get<0>(t) << std::endl;
              std::cout << "Neighbor idx: "
                        << std::get<1>(t)->active_cell_index() << std::endl;
              std::cout << "Face idx from outside " << std::get<2>(t)
                        << std::endl;
            }
        }
    }

  return 0;
}
