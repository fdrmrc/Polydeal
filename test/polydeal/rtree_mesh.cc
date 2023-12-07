#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/reference_cell.h>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <agglomeration_handler.h>
#include <poly_utils.h>

#include <algorithm>

#include "../tests.h"
enum class GridTypes
{
  grid_generator,
  unstructured
};


template <int dim>
class Poisson
{
private:
  void
  make_grid();

  Triangulation<dim>                         tria;
  MappingQ<dim>                              mapping;
  std::unique_ptr<AgglomerationHandler<dim>> ah;

public:
  Poisson(const GridTypes &grid_type = GridTypes::grid_generator,
          const unsigned int         = 0);
  void
  run();

  GridTypes    grid_type;
  unsigned int extraction_level;
};



template <int dim>
Poisson<dim>::Poisson(const GridTypes &  grid_type,
                      const unsigned int extraction_level)
  : mapping(1)
  , grid_type(grid_type)
  , extraction_level(extraction_level)

{}

template <int dim>
void
Poisson<dim>::make_grid()
{
  GridIn<dim> grid_in;
  if (grid_type == GridTypes::unstructured)
    {
      grid_in.attach_triangulation(tria);
      std::ifstream gmsh_file("../../meshes/t2.msh");
      grid_in.read_msh(gmsh_file);
      tria.refine_global(4);
    }
  else
    {
      GridGenerator::hyper_cube(tria, 0., 1.);
      tria.refine_global(5); // 32x32 = 1024 elements
    }
  std::cout << "Size of fine triangulation: " << tria.n_active_cells()
            << std::endl;

  // Partition with Rtree
  namespace bgi                                   = boost::geometry::index;
  static constexpr unsigned int max_elem_per_node = 4;
  std::vector<std::pair<BoundingBox<dim>,
                        typename Triangulation<dim>::active_cell_iterator>>
    boxes(tria.n_active_cells());

  unsigned int i = 0;
  for (const auto &cell : tria.active_cell_iterators())
    boxes[i++] = std::make_pair(mapping.get_bounding_box(cell), cell);


  const auto tree = pack_rtree<bgi::rstar<max_elem_per_node>>(boxes);

  Assert(n_levels(tree) >= 2, ExcMessage("At least two levels are needed."));
  std::cout << "Total number of available levels: " << n_levels(tree)
            << std::endl;
  // Rough description of the tria with a tree of BBoxes
  const auto vec_boxes = extract_rtree_level(tree, extraction_level);
  std::vector<BoundingBox<dim>> bboxes;
  for (const auto &box : vec_boxes)
    bboxes.push_back(box);

  std::vector<std::pair<BoundingBox<dim>,
                        typename Triangulation<dim, dim>::active_cell_iterator>>
    cells;
  std::vector<typename Triangulation<dim, dim>::active_cell_iterator>
             cells_to_agglomerate;
  const auto csr_and_agglomerates =
    PolyUtils::extract_children_of_level(tree, extraction_level);

  const auto &agglomerates   = csr_and_agglomerates.second; // vec<vec<>>
  [[maybe_unused]] auto csrs = csr_and_agglomerates.first;

  std::size_t agglo_index = 0;
  for (std::size_t i = 0; i < agglomerates.size(); ++i)
    {
      const auto &agglo = agglomerates[i]; // i-th agglomerate
      for (const auto &el : agglo)
        el->set_subdomain_id(agglo_index);

      ++agglo_index; // one agglomerate has been processed, increment counter
    }

  std::cout << "N subdomains = " << agglo_index << std::endl;

  for (unsigned int j = 0; j < agglo_index; ++j)
    std::cout << GridTools::count_cells_with_subdomain_association(tria, j)
              << " cells have subdomain id = " + std::to_string(j) << std::endl;

  // // Check number of agglomerates
  // {
  //   GridOut           grid_out_svg;
  //   GridOutFlags::Svg svg_flags;
  //   svg_flags.label_subdomain_id = true;
  //   svg_flags.coloring           = GridOutFlags::Svg::subdomain_id;
  //   grid_out_svg.set_flags(svg_flags);
  //   std::string   grid_type = "agglomerated_grid";
  //   std::ofstream out(grid_type + ".svg");
  //   grid_out_svg.write_svg(tria, out);
  // }
}

template <int dim>
void
Poisson<dim>::run()
{
  make_grid();
}

int
main()
{
  for (const unsigned int extraction_level : {1, 2, 3})
    {
      std::cout << "Extraction level = " << extraction_level << std::endl;
      Poisson<2> poisson_problem{GridTypes::grid_generator, extraction_level};
      poisson_problem.run();
      std::cout << std::endl;
    }



  return 0;
}
