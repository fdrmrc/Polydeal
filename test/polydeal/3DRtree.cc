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

// Check that the R-tree based agglomeration works also in 3D.


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
class Poisson
{
private:
  void
  make_grid();
  void
  setup_agglomeration();

  Triangulation<dim>                         tria;
  MappingQ<dim>                              mapping;
  FE_DGQ<dim>                                dg_fe;
  std::unique_ptr<AgglomerationHandler<dim>> ah;
  // no hanging node in DG discretization, we define an AffineConstraints object
  // so we can use the distribute_local_to_global() directly.
  AffineConstraints<double>              constraints;
  Vector<double>                         system_rhs;
  std::unique_ptr<GridTools::Cache<dim>> cached_tria;
  std::unique_ptr<const Function<dim>>   rhs_function;

public:
  Poisson(const GridType        &grid_type        = GridType::grid_generator,
          const PartitionerType &partitioner_type = PartitionerType::rtree,
          const unsigned int                      = 0,
          const unsigned int                      = 0,
          const unsigned int fe_degree            = 1);
  void
  run();

  double          penalty_constant = 10.;
  GridType        grid_type;
  PartitionerType partitioner_type;
  unsigned int    extraction_level;
  unsigned int    n_subdomains;
};



template <int dim>
Poisson<dim>::Poisson(const GridType        &grid_type,
                      const PartitionerType &partitioner_type,
                      const unsigned int     extraction_level,
                      const unsigned int     n_subdomains,
                      const unsigned int     fe_degree)
  : mapping(1)
  , dg_fe(fe_degree)
  , grid_type(grid_type)
  , partitioner_type(partitioner_type)
  , extraction_level(extraction_level)
  , n_subdomains(n_subdomains)
{}

template <int dim>
void
Poisson<dim>::make_grid()
{
  GridIn<dim> grid_in;
  if (grid_type == GridType::unstructured)
    {
      grid_in.attach_triangulation(tria);
      // std::ifstream gmsh_file("../../meshes/t2.msh"); // rectangular domain
      // grid_in.read_msh(gmsh_file);
      // tria.refine_global(4);
      std::ifstream gmsh_file(
        "../../meshes/piston_2.inp"); // rectangular domain
      grid_in.read_abaqus(gmsh_file);
    }
  else
    {
      GridGenerator::hyper_ball(tria);
      tria.refine_global(3); // 6
    }
  std::cout << "Size of tria: " << tria.n_active_cells() << std::endl;
  cached_tria = std::make_unique<GridTools::Cache<dim>>(tria, mapping);

  if (partitioner_type == PartitionerType::metis)
    {
      // Partition the triangulation with graph partitioner.
      GridTools::partition_triangulation(n_subdomains,
                                         tria,
                                         SparsityTools::Partitioner::metis);
      std::cout << "N subdomains: " << n_subdomains << std::endl;
    }
  else if (partitioner_type == PartitionerType::rtree)
    {
      // Partition with Rtree

      namespace bgi = boost::geometry::index;
      // const unsigned int            extraction_level  = 4; // 3 okay too
      static constexpr unsigned int max_elem_per_node = 8; // 2
      std::vector<std::pair<BoundingBox<dim>,
                            typename Triangulation<dim>::active_cell_iterator>>
                   boxes(tria.n_active_cells());
      unsigned int i = 0;
      for (const auto &cell : tria.active_cell_iterators())
        boxes[i++] = std::make_pair(mapping.get_bounding_box(cell), cell);


      // const auto tree = pack_rtree<bgi::rstar<max_elem_per_node>>(boxes);
      const auto tree = pack_rtree<bgi::rstar<max_elem_per_node>>(boxes);

      // boost::geometry::index::detail::rtree::utilities::print(std::cout,
      // tree);

      std::cout << "Total number of available levels: " << n_levels(tree)
                << std::endl;
      // Rough description of the tria with a tree of BBoxes
      const auto vec_boxes = extract_rtree_level(tree, extraction_level);
      std::vector<BoundingBox<dim>> bboxes;
      for (const auto &box : vec_boxes)
        bboxes.push_back(box);

      std::vector<
        std::pair<BoundingBox<dim>,
                  typename Triangulation<dim, dim>::active_cell_iterator>>
        cells;
      std::vector<typename Triangulation<dim, dim>::active_cell_iterator>
                                            cells_to_agglomerate;
      std::vector<types::global_cell_index> idxs_to_agglomerate;
      const auto                           &csr_and_agglomerates =
        PolyUtils::extract_children_of_level(tree, extraction_level);


      const auto &agglomerates   = csr_and_agglomerates.second; // vec<vec<>>
      [[maybe_unused]] auto csrs = csr_and_agglomerates.first;

      std::size_t agglo_index = 0;
      for (std::size_t i = 0; i < agglomerates.size(); ++i)
        {
          const auto &agglo = agglomerates[i];
          for (const auto &el : agglo)
            {
              el->set_subdomain_id(agglo_index);
            }
          ++agglo_index; // one agglomerate has been processed, increment
                         // counter
        }
      n_subdomains = agglo_index;

      std::cout << "N subdomains = " << n_subdomains << std::endl;
      constraints.close();
    }
}

template <int dim>
void
Poisson<dim>::setup_agglomeration()

{
  ah = std::make_unique<AgglomerationHandler<dim>>(*cached_tria);

  std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>>
    cells_per_subdomain(n_subdomains);
  for (const auto &cell : tria.active_cell_iterators())
    cells_per_subdomain[cell->subdomain_id()].push_back(cell);

  // For every subdomain, agglomerate elements together
  for (std::size_t i = 0; i < cells_per_subdomain.size(); ++i)
    {
      // std::cout << "Subdomain " << i << std::endl;
      std::vector<types::global_cell_index> idxs_to_be_agglomerated;
      std::vector<typename Triangulation<dim>::active_cell_iterator>
        cells_to_be_agglomerated;
      // Get all the elements associated with the present subdomain_id
      for (const auto element : cells_per_subdomain[i])
        {
          idxs_to_be_agglomerated.push_back(element->active_cell_index());
        }
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated,
                                                 cells_to_be_agglomerated);
      // Agglomerate the cells just stored
      ah->define_agglomerate(cells_to_be_agglomerated);
    }

  {
    const std::string &partitioner =
      (partitioner_type == PartitionerType::metis) ? "metis" : "rtree";

    const std::string filename =
      "grid" + partitioner + "_" + std::to_string(n_subdomains) + ".vtu";
    std::ofstream output(filename);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(ah->agglo_dh);
    // data_out.attach_dof_handler(ah->output_dh);

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
Poisson<dim>::run()
{
  make_grid();
  setup_agglomeration();
}

int
main()
{
  {
    Poisson<3> poisson_problem{GridType::grid_generator,
                               PartitionerType::rtree,
                               2};
    poisson_problem.run();
  }



  return 0;
}
