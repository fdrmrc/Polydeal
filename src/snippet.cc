#include <deal.II/grid/grid_generator.h>

#include <algorithm>

#include "../include/agglomeration_handler.h"

int
main()
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria, -1, 1);
  tria.refine_global(2);
  MappingQ<2>             mapping(1);
  GridTools::Cache<2>     cached_tria(tria, mapping);
  AgglomerationHandler<2> ah(cached_tria);
  // agglomerate cells 3,6,9,12 . First, store iterators to them into an array
  std::vector<unsigned int> idxs_to_be_agglomerated = {3, 6, 9, 12, 13};
  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated;
  for (const auto &cell : tria.active_cell_iterators())
    {
      if (std::find(idxs_to_be_agglomerated.begin(),
                    idxs_to_be_agglomerated.end(),
                    cell->active_cell_index()) != idxs_to_be_agglomerated.end())
        {
          cells_to_be_agglomerated.push_back(cell);
        }
    }
  // Let's agglomerate the cells just stored
  ah.agglomerate_cells(cells_to_be_agglomerated);

  // std::vector<typename Triangulation<2>::active_cell_iterator>
  //                           cells_to_be_agglomerated2;
  // std::vector<unsigned int> idxs_to_be_agglomerated2 = {4, 5, 6, 7};
  // for (const auto &cell : tria.active_cell_iterators())
  //   {
  //     if (std::find(idxs_to_be_agglomerated2.begin(),
  //                   idxs_to_be_agglomerated2.end(),
  //                   cell->active_cell_index()) !=
  //         idxs_to_be_agglomerated2.end())
  //       {
  //         cells_to_be_agglomerated2.push_back(cell);
  //       }
  //   }
  // // Let's agglomerate the cells just stored
  // ah.agglomerate_cells(cells_to_be_agglomerated2);

  print_agglomeration(std::cout, ah);

  // Print the cells agglomerated with the third one
  // for (const auto &cell :
  //      ah.get_agglomerated_cells(++(++(++(tria.begin_active())))))
  //   std::cout << cell->active_cell_index() << std::endl;

  const auto test_quad_over_agglomerated =
    ah.agglomerated_quadrature(cells_to_be_agglomerated, QGauss<2>(1));

  double sum = 0.;
  for (const auto &weights : test_quad_over_agglomerated.get_weights())
    sum += weights;
  std::cout << sum << std::endl;

  auto it_start = tria.begin_active();
  std::advance(it_start, 3);
  std::cout << "Number of agglomerated faces (so far): "
            << ah.n_agglomerated_faces(it_start) << std::endl;
  return 0;
}
