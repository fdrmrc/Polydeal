/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2003 - 2023 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * Part of the source code is dual licensed under Apache-2.0 WITH
 * LLVM-exception OR LGPL-2.1-or-later. Detailed license information
 * governing the source code and code contributions can be found in
 * LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
 *
 * ------------------------------------------------------------------------
 */

// Check that constant and linear functions are prolonged exactly also with an
// unstructured external grid.


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
#include <multigrid_utils.h>
#include <poly_utils.h>

#include <algorithm>
#include <chrono>



template <int dim>
class LinearFunction : public Function<dim>
{
public:
  LinearFunction(const std::vector<int> &coeffs)
  {
    Assert(coeffs.size() <= dim, ExcMessage("Wrong size!"));
    coefficients.resize(coeffs.size());
    for (size_t i = 0; i < coeffs.size(); i++)
      coefficients[i] = coeffs[i];
  }
  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;
  std::vector<int> coefficients;
};

template <int dim>
double
LinearFunction<dim>::value(const Point<dim> &p, const unsigned int) const
{
  double value = 0.;
  for (size_t i = 0; i < coefficients.size(); i++)
    value += coefficients[i] * p[i];
  return value;
}



template <int dim>
class Test
{
private:
  void
  make_grid();
  void
  check_transfer();


  Triangulation<dim>                         tria;
  MappingQ<dim>                              mapping;
  FE_DGQ<dim>                                dg_fe;
  std::unique_ptr<AgglomerationHandler<dim>> ah_coarse;
  std::unique_ptr<AgglomerationHandler<dim>> ah_fine;
  std::unique_ptr<GridTools::Cache<dim>>     cached_tria;


public:
  Test(const unsigned int = 0, const unsigned int fe_degree = 1);
  void
  run();


  unsigned int extraction_level;
};



template <int dim>
Test<dim>::Test(const unsigned int extraction_level,
                const unsigned int fe_degree)
  : mapping(1)
  , dg_fe(fe_degree)
  , extraction_level(extraction_level)

{}

template <int dim>
void
Test<dim>::make_grid()
{
  GridIn<dim> grid_in;
  grid_in.attach_triangulation(tria);
  std::ifstream gmsh_file(SOURCE_DIR "/input_grids/square.msh");
  grid_in.read_msh(gmsh_file);
  tria.refine_global(3);
  cached_tria = std::make_unique<GridTools::Cache<dim>>(tria, mapping);
}



template <int dim>
void
Test<dim>::check_transfer()
{
  ah_coarse = std::make_unique<AgglomerationHandler<dim>>(*cached_tria);

  // build the tree
  namespace bgi = boost::geometry::index;
  static constexpr unsigned int max_elem_per_node =
    PolyUtils::constexpr_pow(2, dim); // 2^dim
  std::vector<std::pair<BoundingBox<dim>,
                        typename Triangulation<dim>::active_cell_iterator>>
               boxes(tria.n_active_cells());
  unsigned int i = 0;
  for (const auto &cell : tria.active_cell_iterators())
    boxes[i++] = std::make_pair(mapping.get_bounding_box(cell), cell);

  const auto tree = pack_rtree<bgi::rstar<max_elem_per_node>>(boxes);

  const auto &csr_and_agglomerates =
    PolyUtils::extract_children_of_level(tree, extraction_level);
  const auto &agglomerates = csr_and_agglomerates.second; // vec<vec<>>
  {
    std::size_t agglo_index = 0;
    for (std::size_t i = 0; i < agglomerates.size(); ++i)
      {
#ifdef FALSE
        std::cout << "AGGLO " + std::to_string(i) << std::endl;
#endif
        const auto &agglo = agglomerates[i];
        for (const auto &el : agglo)
          {
            el->set_subdomain_id(agglo_index);
#ifdef FALSE
            std::cout << el->active_cell_index() << std::endl;
#endif
          }
        ++agglo_index; // one agglomerate has been processed, increment
                       // counter
      }

    const unsigned int n_subdomains = agglo_index;

    std::cout << "N elements (coarse) = " << n_subdomains << std::endl;

    std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>>
      cells_per_subdomain(n_subdomains);
    for (const auto &cell : tria.active_cell_iterators())
      cells_per_subdomain[cell->subdomain_id()].push_back(cell);

    // For every subdomain, agglomerate elements together
    for (std::size_t i = 0; i < cells_per_subdomain.size(); ++i)
      ah_coarse->define_agglomerate(cells_per_subdomain[i]);


    ah_coarse->distribute_agglomerated_dofs(dg_fe);
  }



  // ****************do the same for the fine grid one****************
  ah_fine = std::make_unique<AgglomerationHandler<dim>>(*cached_tria);

  {
    const auto &csr_and_agglomerates_fine =
      PolyUtils::extract_children_of_level(tree,
                                           extraction_level + 1); //! level+1
    const auto &agglomerates = csr_and_agglomerates_fine.second;  // vec<vec<>>

    std::size_t agglo_index_fine  = 0;
    std::size_t n_subdomains_fine = 0;
    for (std::size_t i = 0; i < agglomerates.size(); ++i)
      {
#ifdef FALSE
        std::cout << "AGGLO FINE" + std::to_string(i) << std::endl;
#endif
        const auto &agglo = agglomerates[i];
        for (const auto &el : agglo)
          {
            el->set_material_id(agglo_index_fine);
#ifdef FALSE
            std::cout << el->active_cell_index() << std::endl;
#endif
          }
        ++agglo_index_fine; // one agglomerate has been processed, increment
                            // counter
      }

    n_subdomains_fine = agglo_index_fine;

    std::cout << "N elements (fine) = " << n_subdomains_fine << std::endl;


    std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>>
      cells_per_subdomain(n_subdomains_fine);
    for (const auto &cell : tria.active_cell_iterators())
      cells_per_subdomain[cell->material_id()].push_back(cell);

    // For every subdomain, agglomerate elements together
    for (std::size_t i = 0; i < cells_per_subdomain.size(); ++i)
      ah_fine->define_agglomerate(cells_per_subdomain[i]);

    ah_fine->distribute_agglomerated_dofs(dg_fe);
  }

#ifdef FALSE
  std::cout << "Master cells fine: " << std::endl;
  for (const auto &cell : ah_fine->master_cells_container)
    std::cout << cell->active_cell_index() << std::endl;
#endif


  // Check construction of transfer operator

  std::cout << "Construct transfer operator" << std::endl;
  RtreeInfo<2> rtree_info{csr_and_agglomerates.first,
                          csr_and_agglomerates.second};
  MGTwoLevelTransferAgglomeration<dim, Vector<double>> agglomeration_transfer(
    rtree_info);
  agglomeration_transfer.reinit(*ah_fine, *ah_coarse);


  const auto &do_test = [&](const Function<dim> &func) {
    // Test with linear function
    Vector<double>   interp_coarse(ah_coarse->agglo_dh.n_dofs());
    std::vector<int> coeffs{1, 1};
    VectorTools::interpolate(*(ah_coarse->euler_mapping),
                             ah_coarse->agglo_dh,
                             func,
                             interp_coarse);

    Vector<double> dst(ah_fine->agglo_dh.n_dofs());
    agglomeration_transfer.prolongate(dst, interp_coarse);


#ifdef FALSE
    DataOut<2> data_out;
    data_out.attach_dof_handler(ah_coarse->agglo_dh);
    data_out.add_data_vector(interp_coarse, "solution");
    data_out.build_patches(*(ah_coarse->euler_mapping));
    std::ofstream output_coarse("coarse_sol_linear.vtk");
    data_out.write_vtk(output_coarse);
    data_out.clear();
    std::ofstream output_fine("prolonged_solution_linear.vtk");
    data_out.attach_dof_handler(ah_fine->agglo_dh);
    data_out.add_data_vector(dst, "prolonged_solution_linear");
    data_out.build_patches(*(ah_fine->euler_mapping));
    data_out.write_vtk(output_fine);
#endif

    // Compute error:
    Vector<double> interp_fine(ah_fine->agglo_dh.n_dofs());
    VectorTools::interpolate(*(ah_fine->euler_mapping),
                             ah_fine->agglo_dh,
                             func,
                             interp_fine);

    Vector<double> err;
    dst -= interp_fine;
#ifdef FALSE
    std::cout << "Norm of error(L2): " << dst.l2_norm() << std::endl;
#endif
    // Simply check we are below a given treshold, in order to avoid diffs of
    // the order of machine eps.
    AssertThrow(dst.l2_norm() < 1e-14, ExcMessage("L2 norm too large."));
    std::cout << "L2 error: OK." << std::endl;
  };

  std::cout << "f= 1" << std::endl;
  do_test(Functions::ConstantFunction<dim>(1.));
  std::cout << "f= x+y" << std::endl;
  std::vector<int> coeffs_linear{1, 1};
  do_test(LinearFunction<dim>{coeffs_linear});
  std::cout << "f= x" << std::endl;
  std::vector<int> coeffs_x{1, 0};
  do_test(LinearFunction<dim>{coeffs_x});
  std::cout << "f= y" << std::endl;
  std::vector<int> coeffs_y{0, 1};
  do_test(LinearFunction<dim>{coeffs_y});
}



template <int dim>
void
Test<dim>::run()
{
  make_grid();
  check_transfer();
}



int
main()
{
  Test<2> prolongation_test{4 /*extaction_level*/, 1};
  prolongation_test.run();

  return 0;
}
