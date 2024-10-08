// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright ( ) XXXX - YYYY by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------


// Agglomerated multigrid with simplex meshes.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_integrate_difference.templates.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <agglomeration_handler.h>
#include <agglomerator.h>
#include <multigrid_amg.h>
#include <poly_utils.h>
#include <utils.h>

using namespace dealii;

static constexpr unsigned int degree_finite_element = 1;
static constexpr unsigned int n_components          = 1;
static constexpr bool         CHECK_AMG             = true;

#define GEOMETRIC_APPROACH true

enum class GridType
{
  grid_generator, // hyper_cube or hyper_ball
  unstructured    // square generated with gmsh, unstructured
};



template <typename VectorType, typename MatrixType, typename SolverType>
class MGCoarseDirect : public MGCoarseGridBase<VectorType>
{
public:
  MGCoarseDirect(const MatrixType &matrix)
  {
    coarse_matrix = &matrix;
    direct_solver.initialize(*coarse_matrix);
  }

  void
  initialize(const MatrixType &matrix)
  {}

  virtual void
  operator()(const unsigned int, VectorType &dst, const VectorType &src) const
  {
    if constexpr (std::is_same_v<SolverType, SparseDirectUMFPACK>)
      direct_solver.vmult(dst, src);
    else if constexpr (std::is_same_v<SolverType,
                                      TrilinosWrappers::SolverDirect>)
      const_cast<SolverType *>(&direct_solver)->solve(*coarse_matrix, dst, src);
    else
      AssertThrow(false, ExcNotImplemented());
  }

  SolverType        direct_solver;
  const MatrixType *coarse_matrix;
};



/**
 * The actual class aimed at solving the differential problem.
 */
template <int dim>
class AgglomeratedMultigridSimplex
{
public:
  AgglomeratedMultigridSimplex(const GridType    &grid_type,
                               const unsigned int degree,
                               const unsigned int starting_level,
                               const MPI_Comm     comm);
  void
  run();

private:
  void
  make_fine_grid(const unsigned int);
  void
  agglomerate_and_compute_level_matrices();

  const MPI_Comm                                 comm;
  const unsigned int                             n_ranks;
  FE_SimplexDGP<dim>                             fe_dg;
  const GridType                                &grid_type;
  parallel::fullydistributed::Triangulation<dim> tria_pft;
  DoFHandler<dim>                                dof_handler;
  ConditionalOStream                             pcout;
  unsigned int                                   starting_level;
  const MappingFE<dim>                           mapping;



  std::vector<std::unique_ptr<AgglomerationHandler<dim>>>
    agglomeration_handlers;

  std::vector<TrilinosWrappers::SparseMatrix> injection_matrices_two_level;
  std::unique_ptr<AgglomerationHandler<dim>>  agglomeration_handler_coarse;
};



template <int dim>
AgglomeratedMultigridSimplex<dim>::AgglomeratedMultigridSimplex(
  const GridType    &grid_type_,
  const unsigned int degree,
  const unsigned int starting_tree_level,
  const MPI_Comm     communicator)
  : comm(communicator)
  , n_ranks(Utilities::MPI::n_mpi_processes(comm))
  , fe_dg(degree)
  , grid_type(grid_type_)
  , tria_pft(comm)
  , dof_handler(tria_pft)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(comm) == 0))
  , mapping(FE_SimplexP<dim>{1})
{
  pcout << "Running with " << n_ranks << " MPI ranks." << std::endl;
  pcout << "Grid type:";
  starting_level = starting_tree_level;
  grid_type == GridType::grid_generator ?
    pcout << " Structured mesh" << std::endl :
    pcout << " Unstructured mesh" << std::endl;
}



template <int dim>
void
AgglomeratedMultigridSimplex<dim>::make_fine_grid(
  const unsigned int n_global_refinements)
{
  Triangulation<dim> tria;

  if (grid_type == GridType::unstructured)
    {
      GridIn<dim> grid_in;
      grid_in.attach_triangulation(tria);

      if constexpr (dim == 2)
        {
          std::ifstream gmsh_file(
            "../../meshes/square_simplex_coarser.msh"); // unstructured square
                                                        // made by triangles
          grid_in.read_msh(gmsh_file);
          tria.refine_global(n_global_refinements + 2);
        }
      else
        {
          // To decide
          DEAL_II_NOT_IMPLEMENTED();
        }
    }
  else if (grid_type == GridType::grid_generator)
    {
      Triangulation<dim> tria_hex;
      GridGenerator::hyper_cube(tria_hex, 0., 1.);
      tria_hex.refine_global(n_global_refinements);
      GridGenerator::convert_hypercube_to_simplex_mesh(tria_hex, tria);
    }
  else
    {
      Assert(false, ExcInternalError());
    }

  pcout << "Total number of fine cells: " << tria.n_active_cells() << std::endl;

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
AgglomeratedMultigridSimplex<dim>::agglomerate_and_compute_level_matrices()
{
  using VectorType = LinearAlgebra::distributed::Vector<
    double>; // TrilinosWrappers::MPI::Vector;
  GridTools::Cache<dim> cached_tria(tria_pft, mapping);

  // Define matrix free operator
  AffineConstraints constraints;
  constraints.close();
  dof_handler.distribute_dofs(fe_dg);

  // Start building R-tree
  namespace bgi = boost::geometry::index;
  static constexpr unsigned int max_elem_per_node =
    PolyUtils::constexpr_pow(2, dim); // 2^dim
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

  agglomeration_handler_coarse =
    std::make_unique<AgglomerationHandler<dim>>(cached_tria);


  pcout << "Starting level: " << starting_level << std::endl;
  const unsigned int total_tree_levels = n_levels(tree) - starting_level + 1;

  agglomeration_handlers.resize(total_tree_levels);


  // Loop through the available levels and set AgglomerationHandlers up.
  for (unsigned int extraction_level = starting_level;
       extraction_level <= n_levels(tree);
       ++extraction_level)
    {
      agglomeration_handlers[extraction_level - starting_level] =
        std::make_unique<AgglomerationHandler<dim>>(cached_tria);
      CellsAgglomerator<dim, decltype(tree)> agglomerator{tree,
                                                          extraction_level};
      const auto agglomerates = agglomerator.extract_agglomerates();
      agglomeration_handlers[extraction_level - starting_level]
        ->connect_hierarchy(agglomerator);

      // Flag elements for agglomeration
      unsigned int agglo_index = 0;
      for (unsigned int i = 0; i < agglomerates.size(); ++i)
        {
          const auto &agglo = agglomerates[i]; // i-th agglomerate
          for (const auto &el : agglo)
            {
              el->set_material_id(agglo_index);
            }
          ++agglo_index;
        }

      const unsigned int n_local_agglomerates = agglo_index;
      unsigned int       total_agglomerates =
        Utilities::MPI::sum(n_local_agglomerates, comm);
      pcout << "Total agglomerates per (tree) level: " << extraction_level
            << ": " << total_agglomerates << std::endl;

      // Now, perform agglomeration within each locally owned partition
      std::vector<
        std::vector<typename Triangulation<dim>::active_cell_iterator>>
        cells_per_subdomain(n_local_agglomerates);
      for (const auto &cell : tria_pft.active_cell_iterators())
        if (cell->is_locally_owned())
          cells_per_subdomain[cell->material_id()].push_back(cell);

      // For every subdomain, agglomerate elements together
      for (std::size_t i = 0; i < cells_per_subdomain.size(); ++i)
        agglomeration_handlers[extraction_level - starting_level]
          ->define_agglomerate(cells_per_subdomain[i]);

      agglomeration_handlers[extraction_level - starting_level]
        ->initialize_fe_values(QGaussSimplex<dim>(degree_finite_element + 1),
                               update_values | update_gradients |
                                 update_JxW_values | update_quadrature_points,
                               QGaussSimplex<dim - 1>(degree_finite_element +
                                                      1),
                               update_JxW_values);
      agglomeration_handlers[extraction_level - starting_level]
        ->distribute_agglomerated_dofs(fe_dg);
    }

  // Compute two-level transfers between agglomeration handlers
  pcout << "Fill injection matrices between agglomerated levels" << std::endl;
  injection_matrices_two_level.resize(total_tree_levels);
  pcout << "Number of injection matrices: "
        << injection_matrices_two_level.size() << std::endl;
  for (unsigned int l = 1; l < total_tree_levels; ++l)
    {
      pcout << "from level " << l - 1 << " to level " << l << std::endl;
      SparsityPattern sparsity;
      Utils::fill_injection_matrix(*agglomeration_handlers[l - 1],
                                   *agglomeration_handlers[l],
                                   sparsity,
                                   injection_matrices_two_level[l - 1]);
    }
  pcout << "Computed two-level matrices between agglomerated levels"
        << std::endl;


  // Define transfer between levels.
  std::vector<TrilinosWrappers::SparseMatrix *> transfer_matrices(
    total_tree_levels);
  for (unsigned int l = 0; l < total_tree_levels - 1; ++l)
    transfer_matrices[l] = &injection_matrices_two_level[l];


  // Last matrix, fill it by hand
  // add last two-level (which is an embedding)
  PolyUtils::fill_interpolation_matrix(*agglomeration_handlers.back(),
                                       injection_matrices_two_level.back());
  transfer_matrices[total_tree_levels - 1] =
    &injection_matrices_two_level.back();

  pcout << injection_matrices_two_level.back().m() << " and "
        << injection_matrices_two_level.back().n() << std::endl;


  VectorType      system_rhs;
  const IndexSet &locally_owned_dofs = dof_handler.locally_owned_dofs();
  system_rhs.reinit(locally_owned_dofs, comm);


  MGLevelObject<std::unique_ptr<TrilinosWrappers::SparseMatrix>>
    multigrid_matrices(0, total_tree_levels);

  multigrid_matrices[multigrid_matrices.max_level()] =
    std::make_unique<TrilinosWrappers::SparseMatrix>();

  PolyUtils::assemble_dg_matrix_on_standard_mesh(
    *multigrid_matrices[multigrid_matrices.max_level()],
    system_rhs,
    mapping,
    fe_dg,
    dof_handler);
  pcout << "Built finest operator" << std::endl;

#ifdef ALGEBRAIC_APPROACH
  AmgProjector<dim, TrilinosWrappers::SparseMatrix, double> amg_projector(
    injection_matrices_two_level);
  pcout << "Initialized projector" << std::endl;
  amg_projector.compute_level_matrices(multigrid_matrices);
  pcout << "Projected using transfer_matrices:" << std::endl;

#elif GEOMETRIC_APPROACH
  const unsigned int max_level = multigrid_matrices.max_level();
  // assemble level matrices on each level
  for (unsigned int level = max_level;
       level-- > multigrid_matrices.min_level();)
    {
      multigrid_matrices[level] =
        std::make_unique<TrilinosWrappers::SparseMatrix>();
      pcout << "Assemble matrix on level " << level << std::endl;
      PolyUtils::assemble_dg_matrix(*multigrid_matrices[level],
                                    fe_dg,
                                    *agglomeration_handlers[level]);
    }



#endif

  pcout << "Check dimensions of level operators" << std::endl;
  for (unsigned int l = 0; l < total_tree_levels + 1; ++l)
    pcout << "Number of rows and cols operator " << l << ":("
          << multigrid_matrices[l]->m() << "," << multigrid_matrices[l]->n()
          << ")" << std::endl;


  // Setup multigrid


  // Multigrid matrices
  using LevelMatrixType = TrilinosWrappers::SparseMatrix;
  mg::Matrix<VectorType> mg_matrix(multigrid_matrices);

  using SmootherType = PreconditionChebyshev<LevelMatrixType, VectorType>;
  mg::SmootherRelaxation<SmootherType, VectorType>     mg_smoother;
  MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
  smoother_data.resize(0, total_tree_levels + 1);

  // Fine level
  std::vector<VectorType> diag_inverses(total_tree_levels + 1);
  diag_inverses[total_tree_levels].reinit(locally_owned_dofs, comm);

  // Set exact diagonal for each operator
  for (unsigned int i =
         multigrid_matrices[total_tree_levels]->local_range().first;
       i < multigrid_matrices[total_tree_levels]->local_range().second;
       ++i)
    diag_inverses[total_tree_levels][i] =
      1. / multigrid_matrices[total_tree_levels]->diag_element(i);

  smoother_data[total_tree_levels].preconditioner =
    std::make_shared<DiagonalMatrix<VectorType>>(
      diag_inverses[total_tree_levels]);
  pcout << "Fine smoother data: done" << std::endl;

  pcout << "Start defining smoothers data" << std::endl;

  for (unsigned int l = 0; l < total_tree_levels; ++l)
    {
      pcout << "l = " << l << std::endl;
      diag_inverses[l].reinit(
        agglomeration_handlers[l]->agglo_dh.locally_owned_dofs(), comm);

      // Set exact diagonal for each operator
      for (unsigned int i = multigrid_matrices[l]->local_range().first;
           i < multigrid_matrices[l]->local_range().second;
           ++i)
        diag_inverses[l][i] = 1. / multigrid_matrices[l]->diag_element(i);

      smoother_data[l].preconditioner =
        std::make_shared<DiagonalMatrix<VectorType>>(diag_inverses[l]);
    }

  pcout << "Smoothers data initialized" << std::endl;

  for (unsigned int level = 0; level < total_tree_levels + 1; ++level)
    {
      if (level > 0)
        {
          smoother_data[level].smoothing_range     = 20.; // 15.;
          smoother_data[level].degree              = 5;   // 5;
          smoother_data[level].eig_cg_n_iterations = 20;
        }
      else
        {
          smoother_data[0].smoothing_range = 1e-3;
          smoother_data[0].degree = 3; // numbers::invalid_unsigned_int;
          smoother_data[0].eig_cg_n_iterations = dof_handler.n_dofs();
          smoother_data[0].eig_cg_n_iterations = multigrid_matrices[0]->m();
        }
    }

  mg_smoother.set_steps(5);
  mg_smoother.initialize(multigrid_matrices, smoother_data);

  pcout << "Smoothers initialized" << std::endl;

  // Define coarse grid solver
  const unsigned int min_level = 0;
  Utils::MGCoarseDirect<VectorType,
                        TrilinosWrappers::SparseMatrix,
                        TrilinosWrappers::SolverDirect>
    mg_coarse(*multigrid_matrices[min_level]);

  // Transfers
  MGLevelObject<TrilinosWrappers::SparseMatrix *> mg_level_transfers(
    0, total_tree_levels);
  for (unsigned int l = 0; l < total_tree_levels; ++l)
    mg_level_transfers[l] = transfer_matrices[l];


  std::vector<DoFHandler<dim> *> dof_handlers(total_tree_levels + 1);
  for (unsigned int l = 0; l < dof_handlers.size() - 1; ++l)
    dof_handlers[l] = &agglomeration_handlers[l]->agglo_dh;
  dof_handlers[dof_handlers.size() - 1] = &dof_handler; // fine

  unsigned int lev = 0;
  for (const auto &dh : dof_handlers)
    pcout << "Number of DoFs in level " << lev++ << ": " << dh->n_dofs()
          << std::endl;

  MGTransferAgglomeration<dim, VectorType> mg_transfer(mg_level_transfers,
                                                       dof_handlers);
  pcout << "MG transfers initialized" << std::endl;

  // Define multigrid object and convert to preconditioner.
  Multigrid<VectorType> mg(mg_matrix,
                           mg_coarse,
                           mg_transfer,
                           mg_smoother,
                           mg_smoother,
                           min_level,
                           numbers::invalid_unsigned_int,
                           Multigrid<VectorType>::v_cycle);

  PreconditionMG<dim, VectorType, MGTransferAgglomeration<dim, VectorType>>
    preconditioner(dof_handler, mg, mg_transfer);


  // Finally, solve.

  VectorType solution;
  if constexpr (std::is_same_v<VectorType,
                               LinearAlgebra::distributed::Vector<double>>)
    solution.reinit(locally_owned_dofs, comm);
  else if constexpr (std::is_same_v<VectorType, TrilinosWrappers::MPI::Vector>)
    solution.reinit(system_rhs.locally_owned_elements(), comm);
  else
    DEAL_II_NOT_IMPLEMENTED();

  ReductionControl     solver_control(10000, 1e-12, 1e-9);
  SolverCG<VectorType> cg(solver_control);
  double               start, stop;
  pcout << "Start solver" << std::endl;
  start = MPI_Wtime();
  cg.solve(*multigrid_matrices[multigrid_matrices.max_level()],
           solution,
           system_rhs,
           preconditioner);
  stop = MPI_Wtime();
  pcout << "Agglo AMG elapsed time: " << stop - start << "[s]" << std::endl;

  pcout << "Initial value: " << solver_control.initial_value() << std::endl;
  pcout << "Converged in " << solver_control.last_step()
        << " iterations with value " << solver_control.last_value()
        << std::endl;



  [[maybe_unused]] auto output_results = [&]() -> void {
    pcout << "Output results" << std::endl;
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution,
                             "interpolated_solution",
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

    data_out.build_patches(mapping);
    const std::string filename =
      ("agglo_mg_simplex." +
       Utilities::int_to_string(tria_pft.locally_owned_subdomain(), 4));
    std::ofstream output((filename + ".vtu").c_str());
    data_out.write_vtu(output);

    {
      std::vector<std::string> filenames;
      for (unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(comm); i++)
        {
          filenames.push_back("agglo_mg_simplex." +
                              Utilities::int_to_string(i, 4) + ".vtu");
        }
      std::ofstream master_output("agglo_mg_simplex.pvtu");
      data_out.write_pvtu_record(master_output, filenames);
    }
  };

  if (dof_handler.n_dofs() < 3e6)
    output_results();



  if constexpr (CHECK_AMG == true)
    {
      if (starting_level == 2)
        {
          pcout << "Classical way" << std::endl;
          TrilinosWrappers::PreconditionAMG                 prec_amg;
          TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
          amg_data.aggregation_threshold = 1e-3;
          amg_data.smoother_type         = "Chebyshev";
          amg_data.smoother_sweeps       = 5;
          amg_data.output_details        = true;
          if (degree_finite_element > 1)
            amg_data.higher_order_elements = true;
          pcout << "Initialized AMG prec matrix" << std::endl;
          prec_amg.initialize(*multigrid_matrices[max_level], amg_data);

          solution = 0.;
          SolverCG<VectorType> cg_check(solver_control);
          double               start_cg, stop_cg;
          start_cg = MPI_Wtime();
          cg_check.solve(*multigrid_matrices[max_level],
                         solution,
                         system_rhs,
                         prec_amg); // with id
          stop_cg = MPI_Wtime();

          pcout << "CG+AMG elapsed time: " << stop_cg - start_cg << "[s]"
                << std::endl;

          pcout << "Initial value: " << solver_control.initial_value()
                << std::endl;
          pcout << "Converged (CG+AMG) in " << solver_control.last_step()
                << " iterations with value " << solver_control.last_value()
                << std::endl;


          if (dof_handler.n_dofs() < 3e6)
            {
              DataOut<dim> data_out;
              data_out.attach_dof_handler(dof_handler);
              data_out.add_data_vector(solution,
                                       "interpolated_solution",
                                       DataOut<dim>::type_dof_data);

              const std::string filename = "check_multigrid_mf_amg.vtu";
              std::ofstream     output(filename);
              data_out.build_patches(mapping);
              data_out.write_vtu(output);
            }
        }
    }
}



template <int dim>
void
AgglomeratedMultigridSimplex<dim>::run()
{
  make_fine_grid(4);
  agglomerate_and_compute_level_matrices();
  pcout << std::endl;
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  const MPI_Comm                   comm = MPI_COMM_WORLD;
  static constexpr unsigned int    dim  = 2;

  if (Utilities::MPI::this_mpi_process(comm) == 0)
    std::cout << "Degree: " << degree_finite_element << std::endl;

  for (unsigned int starting_level = 1; starting_level < 3; ++starting_level)
    {
      AgglomeratedMultigridSimplex<dim> problem(GridType::grid_generator,
                                                degree_finite_element,
                                                starting_level,
                                                comm);
      problem.run();
    }
}