#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/data_out.h>

#include <agglomeration_handler.h>
#include <poly_utils.h>


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
class Test
{
private:
  void
  make_grid();
  void
  setup_agglomeration();

  Triangulation<dim>                         tria;
  MappingFE<dim>                             mapping;
  FE_DGQ<dim>                                dg_fe;
  std::unique_ptr<AgglomerationHandler<dim>> ah;
  // no hanging node in DG discretization, we define an AffineConstraints object
  // so we can use the distribute_local_to_global() directly.
  AffineConstraints<double>              constraints;
  Vector<double>                         system_rhs;
  std::unique_ptr<GridTools::Cache<dim>> cached_tria;
  std::unique_ptr<const Function<dim>>   rhs_function;

public:
  Test(const GridType        &grid_type        = GridType::grid_generator,
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
Test<dim>::Test(const GridType        &grid_type,
                const PartitionerType &partitioner_type,
                const unsigned int     extraction_level,
                const unsigned int     n_subdomains,
                const unsigned int     fe_degree)
  : mapping(FE_SimplexDGP<dim>(fe_degree))
  , dg_fe(fe_degree)
  , grid_type(grid_type)
  , partitioner_type(partitioner_type)
  , extraction_level(extraction_level)
  , n_subdomains(n_subdomains)
{}

template <int dim>
void
Test<dim>::make_grid()
{
  GridIn<dim> grid_in;
  if (grid_type == GridType::unstructured)
    {
      grid_in.attach_triangulation(tria);
      std::cout << "########### Reading mesh file... ###########" << std::endl;
      std::ifstream filename("../../meshes/ernie.msh"); // liver or brain domain
      grid_in.read_msh(filename);
      std::cout << "########### Done ###########" << std::endl;
    }
  else
    {
      GridGenerator::hyper_cube(tria, 0, 1);
      tria.refine_global(2);
      GridTools::distort_random(0.25, tria);
    }
  std::cout << "Size of tria: " << tria.n_active_cells() << std::endl;
  cached_tria = std::make_unique<GridTools::Cache<dim>>(tria, mapping);
}

template <int dim>
void
Test<dim>::setup_agglomeration()

{
  ah = std::make_unique<AgglomerationHandler<dim>>(*cached_tria);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated = {0, 1, 2, 7};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated,
                                             cells_to_be_agglomerated);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated2 = {4, 5, 9};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated2;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated2,
                                             cells_to_be_agglomerated2);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated3 = {3, 6};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated3;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated3,
                                             cells_to_be_agglomerated3);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated4 = {8,
                                                                    9,
                                                                    10,
                                                                    15};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated4;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated4,
                                             cells_to_be_agglomerated4);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated5 = {11,
                                                                    12,
                                                                    13,
                                                                    14};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated5;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated5,
                                             cells_to_be_agglomerated5);

  // Agglomerate the cells just stored
  ah->define_agglomerate_with_check(cells_to_be_agglomerated);
  ah->define_agglomerate_with_check(cells_to_be_agglomerated2);
  ah->define_agglomerate_with_check(cells_to_be_agglomerated3);
  ah->define_agglomerate_with_check(cells_to_be_agglomerated4);
  ah->define_agglomerate_with_check(cells_to_be_agglomerated5);

  std::cout << "Number of generated agglomerates: " << ah->n_agglomerates()
            << std::endl;

  //   std::cout << "########### Repairing the grid... ###########" <<
  //   std::endl; ah->repair_grid();
  std::cout << "Defined all agglomerates" << std::endl;

  //   Check local bboxes
  for (const auto &box : ah->get_local_bboxes())
    {
      std::cout << "p0: " << box.get_boundary_points().first << std::endl;
      std::cout << "p1: " << box.get_boundary_points().second << std::endl;
      std::cout << std::endl;
    }

  std::cout << "########### Create output for visualization... ###########"
            << std::endl;
  {
    const std::string &partitioner =
      (partitioner_type == PartitionerType::metis) ? "metis" : "rtree";

    const std::string filename = "square_repaired" + partitioner + ".vtu";
    std::ofstream     output(filename);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(ah->agglo_dh);
    // data_out.attach_dof_handler(ah->output_dh);

    Vector<float> agglo_idx(tria.n_active_cells());
    for (const auto &polytope : ah->polytope_iterators())
      {
        const types::global_cell_index polytopal_idx = polytope->index();
        const auto                    &deal_cells = polytope->get_agglomerate();
        for (const auto &cell : deal_cells)
          agglo_idx[cell->active_cell_index()] = polytopal_idx;
      }
    data_out.add_data_vector(agglo_idx,
                             "agglomerated_idx",
                             DataOut<dim>::type_cell_data);
    data_out.build_patches(mapping);
    data_out.write_vtu(output);
  }
  std::cout << "########### Done ###########" << std::endl;
}

template <int dim>
void
Test<dim>::run()
{
  make_grid();
  setup_agglomeration();
}

int
main()
{
  {
    Test<2> test{GridType::grid_generator,
                 PartitionerType::rtree,
                 4 /*extraction_level*/};
    test.run();
  }



  return 0;
}
