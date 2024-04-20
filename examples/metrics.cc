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



enum class MeshType
{
  square,      // square
  ball,        //  ball
  unstructured // square generated with gmsh, unstructured
};



enum class PartitionerType
{
  metis,
  rtree,
  no_partition
};



template <int dim>
class Mesh
{
private:
  void
  setup_agglomerated_grid();
  void
  define_agglomerates();
  void
  compute_metrics();
  void
  export_mesh() const;


  Triangulation<dim>                         tria;
  std::unique_ptr<GridTools::Cache<dim>>     cached_tria;
  MappingQ<dim>                              mapping;
  FE_DGQ<dim>                                dg_fe;
  std::unique_ptr<AgglomerationHandler<dim>> ah;


public:
  Mesh(const MeshType        &grid_type        = MeshType::square,
       const PartitionerType &partitioner_type = PartitionerType::rtree,
       const unsigned int                      = 0,
       const unsigned int                      = 0);
  void
  run_analysis();

  MeshType        grid_type;
  PartitionerType partitioner_type;
  unsigned int    extraction_level;
  unsigned int    n_subdomains;
};



template <int dim>
Mesh<dim>::Mesh(const MeshType        &grid_type,
                const PartitionerType &partitioner_type,
                const unsigned int     extraction_level,
                const unsigned int     n_subdomains)
  : mapping(1)
  , dg_fe(0)
  , grid_type(grid_type)
  , partitioner_type(partitioner_type)
  , extraction_level(extraction_level)
  , n_subdomains(n_subdomains)
{
  static_assert(dim == 2);
}



template <int dim>
void
Mesh<dim>::setup_agglomerated_grid()
{
  GridIn<dim> grid_in;
  switch (grid_type)
    {
      case MeshType::square:
        {
          GridGenerator::hyper_cube(tria, 0., 1.);
          tria.refine_global(5); // 1024
          std::cout << "Number of background cells: " << tria.n_active_cells()
                    << std::endl;
          break;
        }
      case MeshType::ball:
        {
          GridGenerator::hyper_ball(tria);
          tria.refine_global(6); // 20480
          break;
        }
      case MeshType::unstructured:
        {
          grid_in.attach_triangulation(tria);
          std::ifstream gmsh_file(
            "../../meshes/t3.msh"); // unstructured square [0,1]^2
          grid_in.read_msh(gmsh_file);
          tria.refine_global(5); // 4
          std::cout << "Number of background cells: " << tria.n_active_cells()
                    << std::endl;
          break;
        }
      default:
        Assert(false, ExcNotImplemented());
        break;
    }

  cached_tria = std::make_unique<GridTools::Cache<dim>>(tria, mapping);

  if (partitioner_type == PartitionerType::metis)
    {
      Assert(extraction_level == 0, ExcInternalError());
      // Partition the triangulation with graph partitioner.
      auto start = std::chrono::system_clock::now();
      GridTools::partition_triangulation(n_subdomains,
                                         tria,
                                         SparsityTools::Partitioner::metis);
      std::chrono::duration<double> wctduration =
        (std::chrono::system_clock::now() - start);
      std::cout << "METIS agglomerates built in " << wctduration.count()
                << " seconds [Wall Clock]" << std::endl;
      std::cout << "N agglomerates: " << n_subdomains << std::endl;
    }
  else if (partitioner_type == PartitionerType::rtree)
    {
      Assert(n_subdomains == 0, ExcInternalError());
      // Partition with Rtree
      namespace bgi = boost::geometry::index;
      static constexpr unsigned int max_elem_per_node =
        constexpr_pow(2, dim); // 2^dim
      std::vector<std::pair<BoundingBox<dim>,
                            typename Triangulation<dim>::active_cell_iterator>>
                   boxes(tria.n_active_cells());
      unsigned int i = 0;
      for (const auto &cell : tria.active_cell_iterators())
        boxes[i++] = std::make_pair(mapping.get_bounding_box(cell), cell);

      auto       start = std::chrono::system_clock::now();
      const auto tree  = pack_rtree<bgi::rstar<max_elem_per_node>>(boxes);

#ifdef AGGLO_DEBUG
      std::cout << "Total number of available levels: " << n_levels(tree)
                << std::endl;
      Assert(n_levels(tree) >= 2,
             ExcMessage("At least two levels are needed."));
#endif

      const auto &csr_and_agglomerates =
        PolyUtils::extract_children_of_level(tree, extraction_level);
      const auto &agglomerates = csr_and_agglomerates.second; // vec<vec<>>

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

      std::cout << "N agglomerates = " << n_subdomains << std::endl;

      // Check number of agglomerates
#ifdef FALSE
      for (unsigned int j = 0; j < n_subdomains; ++j)
        std::cout << GridTools::count_cells_with_subdomain_association(tria, j)
                  << " cells have subdomain " + std::to_string(j) << std::endl;
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
#endif
    }
  else if (partitioner_type == PartitionerType::no_partition)
    {
    }
  else
    {
      Assert(false, ExcMessage("Wrong partitioning."));
    }
}



template <int dim>
void
Mesh<dim>::define_agglomerates()
{
  ah = std::make_unique<AgglomerationHandler<dim>>(*cached_tria);

  if (partitioner_type != PartitionerType::no_partition)
    {
      std::vector<
        std::vector<typename Triangulation<dim>::active_cell_iterator>>
        cells_per_subdomain(n_subdomains);
      for (const auto &cell : tria.active_cell_iterators())
        cells_per_subdomain[cell->subdomain_id()].push_back(cell);

      // For every subdomain, agglomerate elements together
      for (std::size_t i = 0; i < cells_per_subdomain.size(); ++i)
        ah->define_agglomerate(cells_per_subdomain[i]);
    }
  else
    {
      // No partitioning means that each cell is a master cell
      for (const auto &cell : tria.active_cell_iterators())
        {
          ah->define_agglomerate({cell});
        }
    }

  ah->distribute_agglomerated_dofs(dg_fe);

  // Post processing
  {
    std::string partitioner;
    if (partitioner_type == PartitionerType::metis)
      partitioner = "metis";
    else if (partitioner_type == PartitionerType::rtree)
      partitioner = "rtree";
    else
      partitioner = "no_partitioning";

    const std::string filename =
      "grid_" + partitioner + "_" + std::to_string(n_subdomains) + ".vtu";
    std::ofstream output(filename);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(ah->agglo_dh);

    Vector<float> agglomerated(tria.n_active_cells());
    Vector<float> agglo_idx(tria.n_active_cells());
    for (const auto &cell : tria.active_cell_iterators())
      {
        agglomerated[cell->active_cell_index()] =
          ah->get_relationships()[cell->active_cell_index()];
        agglo_idx[cell->active_cell_index()] = cell->subdomain_id();
      }
    data_out.add_data_vector(agglomerated,
                             "agglo_relationships",
                             DataOut<dim>::type_cell_data);
    data_out.add_data_vector(agglo_idx,
                             "agglomerated_idx",
                             DataOut<dim>::type_cell_data);
    data_out.build_patches(mapping);
    data_out.write_vtu(output);
  }
}



template <int dim>
void
Mesh<dim>::compute_metrics()
{
  // Check metrics
  ah->initialize_fe_values(QGauss<dim>(2 * dg_fe.degree + 1),
                           update_JxW_values);
  auto metrics = PolyUtils::compute_quality_metrics(*ah);

  // uniformity factors
  const auto &uf = std::get<0>(metrics);
  double      average_uniformity_factor =
    std::accumulate(uf.begin(), uf.end(), 0.) / uf.size();

  std::cout << "Uniformity factor:\n"
            << "Min: " << *std::min_element(uf.begin(), uf.end()) << "\n"
            << "Max: " << *std::max_element(uf.begin(), uf.end()) << "\n"
            << "Average: " << average_uniformity_factor << std::endl;

  std::cout << std::endl;

  // circle ratios
  const auto &cr = std::get<1>(metrics);

  double average_circle_ratio =
    std::accumulate(cr.begin(), cr.end(), 0.) / cr.size();
  std::cout << "Circle ratio:\n"
            << "Min: " << *std::min_element(cr.begin(), cr.end()) << "\n"
            << "Max: " << *std::max_element(cr.begin(), cr.end()) << "\n"
            << "Average: " << average_circle_ratio << std::endl;
  std::cout << std::endl;

  // box ratios
  const auto &br = std::get<2>(metrics);

  double average_box_ratio =
    std::accumulate(br.begin(), br.end(), 0.) / br.size();
  std::cout << "Box ratio:\n"
            << "Min: " << *std::min_element(br.begin(), br.end()) << "\n"
            << "Max: " << *std::max_element(br.begin(), br.end()) << "\n"
            << "Average: " << average_box_ratio << std::endl;
  std::cout << std::endl;

  // coverage
  const double coverage = std::get<3>(metrics);

  std::cout << "Coverage factor: " << coverage << std::endl;
}



template <int dim>
void
Mesh<dim>::export_mesh() const
{
  std::string partitioner;
  if (partitioner_type == PartitionerType::metis)
    partitioner = "metis";
  else if (partitioner_type == PartitionerType::rtree)
    partitioner = "rtree";
  else
    partitioner = "no_partitioning";

  std::string polygon_boundaries{"polygon" + partitioner + "_" +
                                 std::to_string(n_subdomains)};
  PolyUtils::export_polygon_to_csv_file(*ah, polygon_boundaries);
}


template <int dim>
void
Mesh<dim>::run_analysis()
{
  setup_agglomerated_grid();
  define_agglomerates();
  compute_metrics();
  export_mesh();
}



int
main()
{
  static constexpr unsigned int dim = 2;
  std::cout << "Compute quality metrics for a set of polygonal meshes:"
            << std::endl;


  // Square
  {
    std::cout << "TEST: ***********SQUARE GRID***********" << std::endl;
    std::cout << "Rtree: " << std::endl;
    const unsigned int extraction_level = 2;
    Mesh<dim>          test_rtree(MeshType::square,
                         PartitionerType::rtree,
                         extraction_level,
                         0);
    test_rtree.run_analysis();
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Metis: " << std::endl;
    const unsigned int n_subdomains = 16;
    Mesh<dim>          test_metis(MeshType::square,
                         PartitionerType::metis,
                         0,
                         n_subdomains);
    test_metis.run_analysis();
  }
  std::cout
    << "--------------------------------------------------------------------------------"
    << std::endl;


  // Ball
  {
    std::cout << "TEST: ***********BALL***********" << std::endl;
    std::cout << "Rtree: " << std::endl;
    const unsigned int extraction_level = 3;
    Mesh<dim>          test_rtree(MeshType::ball,
                         PartitionerType::rtree,
                         extraction_level,
                         0);
    test_rtree.run_analysis();
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Metis: " << std::endl;
    const unsigned int n_subdomains = 20;
    Mesh<dim>          test_metis(MeshType::ball,
                         PartitionerType::metis,
                         0,
                         n_subdomains);
    test_metis.run_analysis();
  }
  std::cout
    << "--------------------------------------------------------------------------------"
    << std::endl;

  // Unstructured square
  {
    std::cout << "TEST: ***********UNSTRUCTURED SQUARE***********" << std::endl;
    std::cout << "Rtree: " << std::endl;
    const unsigned int extraction_level = 4;
    Mesh<dim>          test_rtree(MeshType::unstructured,
                         PartitionerType::rtree,
                         extraction_level,
                         0);
    test_rtree.run_analysis();
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Metis: " << std::endl;
    const unsigned int n_subdomains = 91;
    Mesh<dim>          test_metis(MeshType::unstructured,
                         PartitionerType::metis,
                         0,
                         n_subdomains);
    test_metis.run_analysis();
  }
  std::cout
    << "--------------------------------------------------------------------------------"
    << std::endl;

  // ********************One more refinement********************************//
  // Square
  {
    std::cout << "TEST: ***********SQUARE GRID***********" << std::endl;
    std::cout << "Rtree: " << std::endl;
    const unsigned int extraction_level = 3;
    Mesh<dim>          test_rtree(MeshType::square,
                         PartitionerType::rtree,
                         extraction_level,
                         0);
    test_rtree.run_analysis();
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Metis: " << std::endl;
    const unsigned int n_subdomains = 64;
    Mesh<dim>          test_metis(MeshType::square,
                         PartitionerType::metis,
                         0,
                         n_subdomains);
    test_metis.run_analysis();
  }
  std::cout
    << "--------------------------------------------------------------------------------"
    << std::endl;


  // Ball
  {
    std::cout << "TEST: ***********BALL***********" << std::endl;
    std::cout << "Rtree: " << std::endl;
    const unsigned int extraction_level = 4;
    Mesh<dim>          test_rtree(MeshType::ball,
                         PartitionerType::rtree,
                         extraction_level,
                         0);
    test_rtree.run_analysis();
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Metis: " << std::endl;
    const unsigned int n_subdomains = 80;
    Mesh<dim>          test_metis(MeshType::ball,
                         PartitionerType::metis,
                         0,
                         n_subdomains);
    test_metis.run_analysis();
  }
  std::cout
    << "--------------------------------------------------------------------------------"
    << std::endl;

  // Unstructured square
  {
    std::cout << "TEST: ***********UNSTRUCTURED SQUARE***********" << std::endl;
    std::cout << "Rtree: " << std::endl;
    const unsigned int extraction_level = 5;
    Mesh<dim>          test_rtree(MeshType::unstructured,
                         PartitionerType::rtree,
                         extraction_level,
                         0);
    test_rtree.run_analysis();
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Metis: " << std::endl;
    const unsigned int n_subdomains = 364;
    Mesh<dim>          test_metis(MeshType::unstructured,
                         PartitionerType::metis,
                         0,
                         n_subdomains);
    test_metis.run_analysis();
  }
  std::cout
    << "--------------------------------------------------------------------------------"
    << std::endl;
}
