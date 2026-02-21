#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/reference_cell.h>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/data_out.h>

#include <agglomeration_handler.h>
#include <poly_utils.h>

#include <algorithm>
#include <chrono>


template <typename T>
constexpr T
constexpr_pow(T num, unsigned int pow)
{
  return (pow >= sizeof(unsigned int) * 8) ? 0 :
         pow == 0                          ? 1 :
                                             num * constexpr_pow(num, pow - 1);
}

enum class GridType
{
  grid_generator,
  unstructured
};

enum class PartitionerType
{
  metis,
  rtree
};



template <int dim>
class AgglomerationBenchmark
{
private:
  void
  make_grid();

  Triangulation<dim>                         tria;
  MappingFE<dim>                             mapping;
  std::unique_ptr<AgglomerationHandler<dim>> ah;

public:
  AgglomerationBenchmark(
    const GridType        &grid_type        = GridType::grid_generator,
    const PartitionerType &partitioner_type = PartitionerType::rtree,
    const unsigned int     extraction_level = 0,
    const unsigned int     n_subdomains     = 0);
  void
  run();

  double          penalty_constant = 10.;
  GridType        grid_type;
  PartitionerType partitioner_type;
  unsigned int    extraction_level;
  unsigned int    n_subdomains;
};



template <int dim>
AgglomerationBenchmark<dim>::AgglomerationBenchmark(
  const GridType        &grid_type,
  const PartitionerType &partitioner_type,
  const unsigned int     extraction_level,
  const unsigned int     n_subdomains)
  : mapping(FE_SimplexDGP<dim>(1))
  , grid_type(grid_type)
  , partitioner_type(partitioner_type)
  , extraction_level(extraction_level)
  , n_subdomains(n_subdomains)
{}

template <int dim>
void
AgglomerationBenchmark<dim>::make_grid()
{
  GridIn<dim> grid_in;
  if (grid_type == GridType::unstructured)
    {
      if constexpr (dim == 2)
        {
          grid_in.attach_triangulation(tria);
          std::ifstream gmsh_file(
            std::string(MESH_DIR) +
            "/meshes/t3.msh"); // unstructured square [0,1]^2
          grid_in.read_msh(gmsh_file);
          tria.refine_global(5); // 4
        }
      else if constexpr (dim == 3)
        {
          // {
          //   grid_in.attach_triangulation(tria);
          //   std::ifstream gmsh_file("../../meshes/piston_3.inp"); // piston
          //   mesh grid_in.read_abaqus(gmsh_file); tria.refine_global(1);
          // }
          grid_in.attach_triangulation(tria);
          std::ifstream mesh_file(
            std::string(MESH_DIR) +
            "/meshes/gray_level_image1.vtk"); // liver mesh
          grid_in.read_vtk(mesh_file);
          // grid_in.read_ucd(mesh_file);
          // std::string(MESH_DIR) +
          // "/meshes/csf_brain_filled_centered_UCD.inp"); // brain mesh
          // tria.refine_global(1);
        }
    }
  else
    {
      GridGenerator::hyper_ball(tria);
      tria.refine_global(6); // 6
      // GridGenerator::hyper_cube(tria, 0., 1.);
      // tria.refine_global(5); //
      // {
      //   std::vector<unsigned int> holes(dim);
      //   holes[0] = 3;
      //   holes[1] = 2;
      //   GridGenerator::cheese(tria, holes); // 3 holes
      //   tria.refine_global(3);
      // }
      // GridGenerator::hyper_rectangle(tria, {0., 0.}, {1.25, 1.});
      // tria.refine_global(7); //
    }
  std::cout << "Size of fine tria: " << tria.n_active_cells() << std::endl;

  if (partitioner_type == PartitionerType::metis)
    {
      // Partition the triangulation with graph partitioner.
      auto start = std::chrono::system_clock::now();
      GridTools::partition_triangulation(n_subdomains,
                                         tria,
                                         SparsityTools::Partitioner::metis);
      std::chrono::duration<double> wctduration =
        (std::chrono::system_clock::now() - start);
      std::cout << "METIS built in " << wctduration.count()
                << " seconds [Wall Clock]" << std::endl;
      std::cout << "N subdomains: " << n_subdomains << std::endl;
    }
  else if (partitioner_type == PartitionerType::rtree)
    {
      // Partition with Rtree

      namespace bgi = boost::geometry::index;
      // const unsigned int            extraction_level  = 4; // 3 okay too
      static constexpr unsigned int max_elem_per_node =
        constexpr_pow(2, dim); // 2^dim
      std::vector<std::pair<BoundingBox<dim>,
                            typename Triangulation<dim>::active_cell_iterator>>
                   boxes(tria.n_active_cells());
      unsigned int i = 0;
      for (const auto &cell : tria.active_cell_iterators())
        boxes[i++] = std::make_pair(mapping.get_bounding_box(cell), cell);


      // const auto tree = pack_rtree<bgi::rstar<max_elem_per_node>>(boxes);
      auto start = std::chrono::system_clock::now();
      auto tree  = pack_rtree<bgi::rstar<max_elem_per_node>>(boxes);
      std::cout << "Total number of available levels: " << n_levels(tree)
                << std::endl;

#ifdef AGGLO_DEBUG
      boost::geometry::index::detail::rtree::utilities::print(std::cout, tree);
      Assert(n_levels(tree) >= 2,
             ExcMessage("At least two levels are needed."));
#endif

      CellsAgglomerator<dim, decltype(tree)> agglomerator{tree,
                                                          extraction_level};
      const auto agglomerates = agglomerator.extract_agglomerates();

      std::size_t agglo_index = 0;
      for (std::size_t i = 0; i < agglomerates.size(); ++i)
        {
          // std::cout << "AGGLO " + std::to_string(i) << std::endl;
          const auto &agglo = agglomerates[i];
          for (const auto &el : agglo)
            {
              el->set_subdomain_id(agglo_index);
              // std::cout << el->active_cell_index() << std::endl;
            }
          ++agglo_index; // one agglomerate has been processed, increment
                         // counter
        }
      std::chrono::duration<double> wctduration =
        (std::chrono::system_clock::now() - start);
      std::cout << "R-tree agglomerates built in " << wctduration.count()
                << " seconds [Wall Clock]" << std::endl;
      n_subdomains = agglo_index;

      std::cout << "N subdomains = " << n_subdomains << std::endl;


      // Check number of agglomerates
      if constexpr (dim == 2)
        {
#ifdef AGGLO_DEBUG
          for (unsigned int j = 0; j < n_subdomains; ++j)
            std::cout << GridTools::count_cells_with_subdomain_association(tria,
                                                                           j)
                      << " cells have subdomain " + std::to_string(j)
                      << std::endl;
#endif
          GridOut           grid_out_svg;
          GridOutFlags::Svg svg_flags;
          svg_flags.background     = GridOutFlags::Svg::Background::transparent;
          svg_flags.line_thickness = 1;
          svg_flags.boundary_line_thickness = 1;
          svg_flags.label_subdomain_id      = true;
          svg_flags.coloring =
            GridOutFlags::Svg::subdomain_id; // GridOutFlags::Svg::none
          grid_out_svg.set_flags(svg_flags);
          std::string   grid_type = "agglomerated_grid";
          std::ofstream out(grid_type + ".svg");
          grid_out_svg.write_svg(tria, out);
        }
    }
}



template <int dim>
void
AgglomerationBenchmark<dim>::run()
{
  make_grid();
}



int
main()
{
  // Benchmarks 3D

  std::cout << "Benchmarking with Rtree:" << std::endl;
  // for (unsigned int i = 0; i < 10; ++i)
  //   {
  for (const unsigned int extraction_level : {2, 3, 4, 5, 6})
    {
      AgglomerationBenchmark<3> poisson_problem{GridType::unstructured,
                                                PartitionerType::rtree,
                                                extraction_level}; // 16, 64
      poisson_problem.run();
    }
  std::cout << std::endl;
  //   }

  std::cout << "Benchmarking with METIS:" << std::endl;
  // // for (unsigned int i = 0; i < 10; ++i)
  // //   {
  // // for (const unsigned int target_partitions : {12, 90, 715, 5715})
  // //piston
  // for (const unsigned int target_partitions : {47, 372, 2976, 23804}) //
  // brain
  for (const unsigned int target_partitions :
       {9, 70, 556, 4441, 29000}) // liver
    {
      AgglomerationBenchmark<3> poisson_problem{GridType::unstructured,
                                                PartitionerType::metis,
                                                0,
                                                target_partitions};
      poisson_problem.run();
    }
}
