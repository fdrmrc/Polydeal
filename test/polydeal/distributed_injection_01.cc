// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) XXXX - YYYY by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------


// Check that we can interpolate from two different agglomeration handlers, one
// finer than the other one, in parallel and with different grid types and
// functions.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_integrate_difference.templates.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <agglomeration_handler.h>
#include <agglomerator.h>
#include <poly_utils.h>
#include <utils.h>

using namespace dealii;


template <int dim>
class SolutionLinear : public Function<dim>
{
public:
  SolutionLinear()
    : Function<dim>()
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

  virtual void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<double>           &values,
             const unsigned int /*component*/) const override;
};

template <int dim>
double
SolutionLinear<dim>::value(const Point<dim> &p, const unsigned int) const
{
  double sum = 0;
  for (unsigned int d = 0; d < dim; ++d)
    sum += p[d];

  return sum - 1;
}



template <int dim>
void
SolutionLinear<dim>::value_list(const std::vector<Point<dim>> &points,
                                std::vector<double>           &values,
                                const unsigned int /*component*/) const
{
  for (unsigned int i = 0; i < values.size(); ++i)
    values[i] = this->value(points[i]);
}



template <int dim>
class SolutionQuadratic : public Function<dim>
{
public:
  SolutionQuadratic()
    : Function<dim>()
  {
    Assert(dim == 2, ExcNotImplemented());
  }

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

  virtual void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<double>           &values,
             const unsigned int /*component*/) const override;
};



template <int dim>
double
SolutionQuadratic<dim>::value(const Point<dim> &p, const unsigned int) const
{
  return p[0] * p[0] + p[1] * p[1] - 1;
}



template <int dim>
void
SolutionQuadratic<dim>::value_list(const std::vector<Point<dim>> &points,
                                   std::vector<double>           &values,
                                   const unsigned int /*component*/) const
{
  for (unsigned int i = 0; i < values.size(); ++i)
    values[i] = this->value(points[i]);
}

/*--------------------------------------------------------------------------*/



enum class GridType
{
  grid_generator, // hyper_cube or hyper_ball
  unstructured    // square generated with gmsh, unstructured
};



template <int dim>
class DistributedHierarchyTest
{
public:
  DistributedHierarchyTest(const GridType      &grid_type,
                           const unsigned int   degree,
                           const Function<dim> &func,
                           const MPI_Comm       comm);
  void
  run();

private:
  void
  make_fine_grid(const unsigned int);
  void
  setup_agglomerated_problem();

  const MPI_Comm                                 comm;
  const unsigned int                             n_ranks;
  FE_DGQ<dim>                                    fe_dg;
  const GridType                                &grid_type;
  parallel::fullydistributed::Triangulation<dim> tria_pft;
  const Function<dim>                           &func;
  ConditionalOStream                             pcout;


  std::unique_ptr<AgglomerationHandler<dim>> ah_coarse;
  std::unique_ptr<AgglomerationHandler<dim>> ah_fine;

  TrilinosWrappers::MPI::Vector interpolated_dst;
};



template <int dim>
DistributedHierarchyTest<dim>::DistributedHierarchyTest(
  const GridType      &grid_type_,
  const unsigned int   degree,
  const Function<dim> &func_,
  const MPI_Comm       communicator)
  : comm(communicator)
  , n_ranks(Utilities::MPI::n_mpi_processes(comm))
  , fe_dg(degree)
  , grid_type(grid_type_)
  , tria_pft(comm)
  , func(func_)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(comm) == 0))
{
  pcout << "*** Running with " << n_ranks << " MPI ranks ***" << std::endl;
  pcout << "Grid type:";
  grid_type == GridType::grid_generator ?
    pcout << " Structured square" << std::endl :
    pcout << " Unstructured square" << std::endl;
}



template <int dim>
void
DistributedHierarchyTest<dim>::make_fine_grid(
  const unsigned int n_global_refinements)
{
  Triangulation<dim> tria;

  if (grid_type == GridType::unstructured)
    {
      GridIn<dim> grid_in;
      grid_in.attach_triangulation(tria);
      std::ifstream gmsh_file(SOURCE_DIR "/input_grids/square.msh");
      grid_in.read_msh(gmsh_file);
      tria.refine_global(5); // 4
    }
  else if (grid_type == GridType::grid_generator)
    {
      GridGenerator::hyper_cube(tria, 0., 1.);
      tria.refine_global(n_global_refinements);
    }
  else
    {
      Assert(false, ExcInternalError());
    }

  // Partition serial triangulation:
  GridTools::partition_triangulation(n_ranks, tria);

  // Create building blocks:
  const TriangulationDescription::Description<dim, dim> description =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      tria, comm);

  tria_pft.create_triangulation(description);
}



template <int dim>
void
DistributedHierarchyTest<dim>::setup_agglomerated_problem()
{
  GridTools::Cache<dim> cached_tria(tria_pft);

  ah_coarse = std::make_unique<AgglomerationHandler<dim>>(cached_tria);

  // Partition with Rtree locally to each partition.
  MappingQ1<dim> mapping; // use standard mapping

  namespace bgi = boost::geometry::index;
  static constexpr unsigned int max_elem_per_node =
    Utils::constexpr_pow(2, dim); // 2^dim
  std::vector<std::pair<BoundingBox<dim>,
                        typename Triangulation<dim>::active_cell_iterator>>
               boxes(tria_pft.n_locally_owned_active_cells());
  unsigned int i = 0;
  for (const auto &cell : tria_pft.active_cell_iterators())
    if (cell->is_locally_owned())
      boxes[i++] = std::make_pair(mapping.get_bounding_box(cell), cell);

  auto tree = pack_rtree<bgi::rstar<max_elem_per_node>>(boxes);
  Assert(n_levels(tree) >= 2, ExcMessage("At least two levels are needed."));
  pcout << "Total number of available levels: " << n_levels(tree) << std::endl;


  std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>>
                                         cells_per_material_id;
  const unsigned int                     extraction_level = 2;
  CellsAgglomerator<dim, decltype(tree)> agglomerator_coarse{tree,
                                                             extraction_level};
  const auto agglomerates_coarse = agglomerator_coarse.extract_agglomerates();
  ah_coarse->connect_hierarchy(agglomerator_coarse);

  std::size_t agglo_index = 0;
  for (std::size_t i = 0; i < agglomerates_coarse.size(); ++i)
    {
      const auto &agglo = agglomerates_coarse[i];
      for (const auto &el : agglo)
        el->set_material_id(agglo_index);

      ++agglo_index; // one agglomerate has been processed, increment
                     // counter
    }

  cells_per_material_id.resize(agglo_index);
  unsigned int total_agglomerates_coarse =
    Utilities::MPI::sum(agglo_index, comm);
  pcout << "Total coarse agglomerates: " << total_agglomerates_coarse
        << std::endl;

  for (const auto &cell : tria_pft.active_cell_iterators())
    if (cell->is_locally_owned())
      cells_per_material_id[cell->material_id()].push_back(cell);

  // Agglomerate elements with same id
  for (std::size_t i = 0; i < cells_per_material_id.size(); ++i)
    ah_coarse->define_agglomerate(cells_per_material_id[i]);

  ah_coarse->initialize_fe_values(QGauss<dim>(3),
                                  update_values | update_gradients |
                                    update_JxW_values |
                                    update_quadrature_points,
                                  QGauss<dim - 1>(3),
                                  update_JxW_values);
  ah_coarse->distribute_agglomerated_dofs(fe_dg);
  pcout << "Distributed coarse DoFs" << std::endl;

  /////////////////////////////////////////////////////////////////////////////
  // Do the same with finer level

  CellsAgglomerator<dim, decltype(tree)> agglomerator_fine{tree,
                                                           extraction_level +
                                                             1};
  const auto agglomerates_fine = agglomerator_fine.extract_agglomerates();
  ah_fine = std::make_unique<AgglomerationHandler<dim>>(cached_tria);

  std::size_t agglo_index_fine = 0;
  for (std::size_t i = 0; i < agglomerates_fine.size(); ++i)
    {
      const auto &agglo = agglomerates_fine[i]; // i-th agglomerate fine
      for (const auto &el : agglo)
        {
          el->set_material_id(agglo_index_fine);
        }
      ++agglo_index_fine;
    }

  const unsigned int n_subdomains_fine = agglo_index_fine;
  unsigned int       total_agglomerates_fine =
    Utilities::MPI::sum(n_subdomains_fine, comm);
  pcout << "Total fine agglomerates: " << total_agglomerates_fine << std::endl;

  std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>>
    cells_per_subdomain_fine(n_subdomains_fine);
  for (const auto &cell : tria_pft.active_cell_iterators())
    if (cell->is_locally_owned())
      cells_per_subdomain_fine[cell->material_id()].push_back(cell);

  // For every subdomain, agglomerate elements together
  for (std::size_t i = 0; i < cells_per_subdomain_fine.size(); ++i)
    ah_fine->define_agglomerate(cells_per_subdomain_fine[i]);

  ah_fine->initialize_fe_values(QGauss<dim>(3),
                                update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points,
                                QGauss<dim - 1>(3),
                                update_JxW_values);
  ah_fine->distribute_agglomerated_dofs(fe_dg);

  pcout << "Distributed fine DoFs" << std::endl;


  // Check injection matrix
  const IndexSet &locally_owned_dofs_coarse =
    ah_coarse->agglo_dh.locally_owned_dofs();
  TrilinosWrappers::MPI::Vector interp_coarse(locally_owned_dofs_coarse, comm);

  VectorTools::interpolate(*(ah_coarse->euler_mapping),
                           ah_coarse->agglo_dh,
                           func,
                           interp_coarse);

  const IndexSet &locally_owned_dofs_fine =
    ah_fine->agglo_dh.locally_owned_dofs();
  TrilinosWrappers::MPI::Vector  dst(locally_owned_dofs_fine, comm);
  SparsityPattern                embedding_sp;
  TrilinosWrappers::SparseMatrix embedding_matrix;
  Utils::fill_injection_matrix(*ah_coarse,
                               *ah_fine,
                               embedding_sp,
                               embedding_matrix);
  embedding_matrix.vmult(dst, interp_coarse);
  pcout << "Multiplication: done" << std::endl;


#ifdef FALSE
  {
    PolyUtils::interpolate_to_fine_grid(*ah_fine, interpolated_dst, dst);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(ah_fine->output_dh);

    data_out.add_data_vector(interpolated_dst,
                             "u",
                             DataOut<dim>::type_dof_data);

    Vector<float> subdomain(tria_pft.n_active_cells());

    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = tria_pft.locally_owned_subdomain();

    data_out.add_data_vector(subdomain, "subdomain");

    Vector<float> agglo_idx(tria_pft.n_active_cells());
    for (const auto &cell : tria_pft.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          agglo_idx[cell->active_cell_index()] = cell->material_id();
      }
    data_out.add_data_vector(agglo_idx,
                             "agglo_idx",
                             DataOut<dim>::type_cell_data);

    data_out.build_patches();
    const std::string filename =
      ("interpolated_dst." +
       Utilities::int_to_string(tria_pft.locally_owned_subdomain(), 4));

    std::ofstream output((filename + ".vtu").c_str());
    data_out.write_vtu(output);


    {
      std::vector<std::string> filenames;
      for (unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(comm); i++)
        {
          filenames.push_back("interpolated_dst." +
                              Utilities::int_to_string(i, 4) + ".vtu");
        }
      std::ofstream master_output("interpolated_dst.pvtu");
      data_out.write_pvtu_record(master_output, filenames);
    }
  }
#endif

  // Compute error:
  TrilinosWrappers::MPI::Vector interp_fine(
    dst.locally_owned_elements(), comm); // take parallel layout from dst
  VectorTools::interpolate(*(ah_fine->euler_mapping),
                           ah_fine->agglo_dh,
                           func,
                           interp_fine);

  TrilinosWrappers::MPI::Vector err(dst);
  dst -= interp_fine;
  const double l2_error = dst.l2_norm();
  AssertThrow(l2_error < 1e-14, ExcMessage("Injection didn't work properly."));
  pcout << "Norm of error(L2): " << l2_error << std::endl;
}



template <int dim>
void
DistributedHierarchyTest<dim>::run()
{
  make_fine_grid(6); // 6 global refinements of unit cube
  setup_agglomerated_problem();
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  const MPI_Comm                   comm = MPI_COMM_WORLD;
  static constexpr unsigned int    dim  = 2;


  // number of agglomerates in each local subdomain
  SolutionLinear<dim>            linear;
  SolutionQuadratic<dim>         quadratic;
  std::array<Function<dim> *, 2> functions{{&linear, &quadratic}};

  for (const Function<dim> *func : functions)
    for (unsigned int degree : {2})
      for (const auto &grid_type :
           {GridType::grid_generator, GridType::unstructured})
        {
          DistributedHierarchyTest<dim> problem(grid_type, degree, *func, comm);
          problem.run();
        }
}
