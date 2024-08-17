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


// Check the operator restriction on coarser, agglomerate, levels.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/la_parallel_vector.h>
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

static constexpr unsigned int degree_finite_element = 3;
static constexpr unsigned int n_components          = 1;
static constexpr bool         CHECK_AMG             = true;

enum class GridType
{
  grid_generator, // hyper_cube or hyper_ball
  unstructured    // square generated with gmsh, unstructured
};



template <int dim>
class AgglomeratedMG
{
public:
  AgglomeratedMG(const GridType    &grid_type,
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
  FE_DGQ<dim>                                    fe_dg;
  MappingQ1<dim>                                 mapping;
  const GridType                                &grid_type;
  parallel::fullydistributed::Triangulation<dim> tria_pft;
  DoFHandler<dim>                                dof_handler;
  ConditionalOStream                             pcout;
  unsigned int                                   starting_level;


  std::vector<std::unique_ptr<AgglomerationHandler<dim>>>
    agglomeration_handlers;
};



template <int dim>
AgglomeratedMG<dim>::AgglomeratedMG(const GridType    &grid_type_,
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
AgglomeratedMG<dim>::make_fine_grid(const unsigned int n_global_refinements)
{
  Triangulation<dim> tria;

  if (grid_type == GridType::unstructured)
    {
      GridIn<dim> grid_in;
      grid_in.attach_triangulation(tria);

      if constexpr (dim == 2)
        {
          std::ifstream gmsh_file(
            "../../meshes/t3.msh"); // unstructured square [0,1]^2
          grid_in.read_msh(gmsh_file);
          tria.refine_global(n_global_refinements + 2);
        }
      else
        {
          std::ifstream abaqus_file("../../meshes/piston_3.inp"); // piston mesh
          grid_in.read_abaqus(abaqus_file);
        }
    }
  else if (grid_type == GridType::grid_generator)
    {
      GridGenerator::hyper_cube(tria, 0., 1.);
      tria.refine_global(n_global_refinements + 3);
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
AgglomeratedMG<dim>::agglomerate_and_compute_level_matrices()
{
  using VectorType             = LinearAlgebra::distributed::Vector<double>;
  using LevelMatrixType        = LinearOperatorMG<VectorType, VectorType>;
  const unsigned int min_level = 0;


  // Define matrix free operator
  dof_handler.distribute_dofs(fe_dg);
  constexpr unsigned int n_qpoints = degree_finite_element + 1;
  Utils::LaplaceOperatorDG<dim,
                           degree_finite_element,
                           n_qpoints,
                           n_components,
                           double>
    system_matrix_dg;
  system_matrix_dg.reinit(mapping, dof_handler);



  // Start defining agglomerated quantities
  const unsigned int total_tree_levels =
    PolyUtils::construct_agglomerated_levels(
      tria_pft, agglomeration_handlers, fe_dg, mapping, starting_level);

  //  Start the multigrid setup

  // Transfers: compute two-level transfers between agglomeration handlers
  MGLevelObject<TrilinosWrappers::SparseMatrix> mg_level_transfers(
    0, total_tree_levels);

  pcout << "Fill injection matrices between agglomerated levels" << std::endl;
  for (unsigned int l = 1; l < total_tree_levels + 1; ++l)
    {
      pcout << "Construct interpolation matrix from level " << l - 1
            << " to level " << l << std::endl;
      SparsityPattern sparsity;
      if (l < total_tree_levels)
        Utils::fill_injection_matrix(*agglomeration_handlers[l - 1],
                                     *agglomeration_handlers[l],
                                     sparsity,
                                     mg_level_transfers[l - 1]);
      else
        PolyUtils::fill_interpolation_matrix(
          *agglomeration_handlers.back(),
          mg_level_transfers[total_tree_levels -
                             1]); // Last matrix, is an "interpolation" onto the
                                  // classical DoFHandler
    }
  pcout << "Computed two-level matrices between agglomerated levels"
        << std::endl;



  AmgProjector<dim, TrilinosWrappers::SparseMatrix, double> amg_projector(
    mg_level_transfers);


  MGLevelObject<std::unique_ptr<TrilinosWrappers::SparseMatrix>>
                     multigrid_matrices(0, total_tree_levels);
  const unsigned int max_level = multigrid_matrices.max_level();


  // Get fine operator and use it to build other levels.
  multigrid_matrices[max_level] =
    std::make_unique<TrilinosWrappers::SparseMatrix>();
  system_matrix_dg.get_system_matrix(*multigrid_matrices[max_level]);

  // Once the level operators are built, use the finest in a matrix-free way,
  // the others matrix-based.
  MGLevelObject<LevelMatrixType> multigrid_matrices_lo(0, max_level);
  amg_projector.compute_level_matrices_as_linear_operators(
    multigrid_matrices, multigrid_matrices_lo);
  pcout << "Projected using transfer_matrices:" << std::endl;

  multigrid_matrices_lo[max_level] =
    linear_operator_mg<VectorType, VectorType>(system_matrix_dg);
  multigrid_matrices_lo[max_level].n_rows = system_matrix_dg.m();
  multigrid_matrices_lo[max_level].n_cols = system_matrix_dg.n();


  pcout << "Check dimensions of level operators" << std::endl;
  for (unsigned int l = 0; l < total_tree_levels + 1; ++l)
    pcout << "Number of rows and cols operator " << l << ":("
          << multigrid_matrices_lo[l].n_rows << ","
          << multigrid_matrices_lo[l].n_cols << ")" << std::endl;


  // Multigrid matrices
  mg::Matrix<VectorType> mg_matrix_lo(multigrid_matrices_lo);

  // Define smoothers
  using SmootherType = PreconditionChebyshev<LevelMatrixType, VectorType>;

  mg::SmootherRelaxation<SmootherType, VectorType>     mg_smoother;
  MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
  smoother_data.resize(0, total_tree_levels + 1);

  // system_matrix.compute_diagonal();
  const VectorType &fine_diag_inverse_vector =
    system_matrix_dg.get_matrix_diagonal_inverse(); // get it from matrix_free

  // Fine level
  smoother_data[total_tree_levels].preconditioner =
    std::make_shared<DiagonalMatrix<VectorType>>(fine_diag_inverse_vector);
  std::vector<LinearAlgebra::distributed::Vector<double>> diag_inverses(
    total_tree_levels + 1);
  diag_inverses[total_tree_levels] = fine_diag_inverse_vector;


  pcout << "Start defining smoothers data" << std::endl;
  for (unsigned int l = 0; l < total_tree_levels; ++l)
    {
      pcout << "l = " << l << std::endl;
      diag_inverses[l].reinit(
        agglomeration_handlers[l]->agglo_dh.locally_owned_dofs(), comm);

      // Set exact for each operator
      for (unsigned int i = multigrid_matrices[l]->local_range().first;
           i < multigrid_matrices[l]->local_range().second;
           ++i)
        diag_inverses[l][i] = 1. / multigrid_matrices[l]->diag_element(i);

      smoother_data[l].preconditioner =
        std::make_shared<DiagonalMatrix<VectorType>>(diag_inverses[l]);
    }

  for (unsigned int level = 0; level < total_tree_levels + 1; ++level)
    {
      if (level > 0)
        {
          smoother_data[level].smoothing_range     = 20.; // 15.;
          smoother_data[level].degree              = 3;   // 5;
          smoother_data[level].eig_cg_n_iterations = 20;
        }
      else
        {
          smoother_data[0].smoothing_range = 1e-3;
          smoother_data[0].degree = 3; // numbers::invalid_unsigned_int;
          smoother_data[0].eig_cg_n_iterations = dof_handler.n_dofs();
        }
    }

  mg_smoother.set_steps(3);
  mg_smoother.initialize(multigrid_matrices_lo, smoother_data);

  pcout << "Smoothers initialized" << std::endl;

  // Define coarse grid solver. Avoid to use the LinearOperatorMG for this
  // level, use directly the matrix.
  Utils::MGCoarseDirect<VectorType,
                        TrilinosWrappers::SparseMatrix,
                        TrilinosWrappers::SolverDirect>
    mg_coarse(*multigrid_matrices[min_level]);



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
  Multigrid<VectorType> mg(
    mg_matrix_lo, mg_coarse, mg_transfer, mg_smoother, mg_smoother);

  PreconditionMG<dim, VectorType, MGTransferAgglomeration<dim, VectorType>>
    preconditioner(dof_handler, mg, mg_transfer);


  // Assemble system rhs
  VectorType system_rhs;
  system_matrix_dg.initialize_dof_vector(system_rhs);

  system_rhs = 0;
  FEEvaluation<dim, degree_finite_element> phi(
    *system_matrix_dg.get_matrix_free());
  for (unsigned int cell = 0;
       cell < system_matrix_dg.get_matrix_free()->n_cell_batches();
       ++cell)
    {
      phi.reinit(cell);
      for (const unsigned int q : phi.quadrature_point_indices())
        phi.submit_value(make_vectorized_array<double>(1.0), q);
      phi.integrate(EvaluationFlags::values);
      phi.distribute_local_to_global(system_rhs);
    }
  system_rhs.compress(VectorOperation::add);


  VectorType solution;
  system_matrix_dg.initialize_dof_vector(solution);
  ReductionControl     solver_control(10000, 1e-9, 1e-6);
  SolverCG<VectorType> cg(solver_control);
  double               start, stop;
  pcout << "Start solver" << std::endl;
  start = MPI_Wtime();
  cg.solve(system_matrix_dg, solution, system_rhs, preconditioner);
  stop = MPI_Wtime();
  pcout << "Agglo AMG elapsed time: " << stop - start << "[s]" << std::endl;

  pcout << "Initial value: " << solver_control.initial_value() << std::endl;
  pcout << "Converged in " << solver_control.last_step()
        << " iterations with value " << solver_control.last_value()
        << std::endl;



  [[maybe_unused]] const auto output_results = [&]() -> void {
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
      ("agglo_mg." +
       Utilities::int_to_string(tria_pft.locally_owned_subdomain(), 4));
    std::ofstream output((filename + ".vtu").c_str());
    data_out.write_vtu(output);

    {
      std::vector<std::string> filenames;
      for (unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(comm); i++)
        {
          filenames.push_back("agglo_mg." + Utilities::int_to_string(i, 4) +
                              ".vtu");
        }
      std::ofstream master_output("agglo_mg.pvtu");
      data_out.write_pvtu_record(master_output, filenames);
    }
  };

  if (dof_handler.n_dofs() < 3e6)
    output_results();



  if constexpr (CHECK_AMG == true)
    {
      if (starting_level == 1)
        {
          const TrilinosWrappers::SparseMatrix &fine_matrix =
            system_matrix_dg.get_system_matrix();
          pcout << "Built finest operator" << std::endl;
          pcout << "Classical way" << std::endl;
          TrilinosWrappers::PreconditionAMG                 prec_amg;
          TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
          amg_data.aggregation_threshold = 1e-3;
          amg_data.smoother_type         = "Chebyshev";
          amg_data.smoother_sweeps       = 3;
          amg_data.output_details        = true;
          if (degree_finite_element > 1)
            amg_data.higher_order_elements = true;
          pcout << "Initialized AMG prec matrix" << std::endl;
          prec_amg.initialize(fine_matrix, amg_data);

          solution = 0.;
          SolverCG<VectorType> cg_check(solver_control);
          double               start_cg, stop_cg;
          start_cg = MPI_Wtime();
          cg_check.solve(fine_matrix,
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
AgglomeratedMG<dim>::run()
{
  const unsigned int n_vect_doubles = VectorizedArray<double>::size();
  const unsigned int n_vect_bits    = 8 * sizeof(double) * n_vect_doubles;

  pcout << "Vectorization over " << n_vect_doubles
        << " doubles = " << n_vect_bits << " bits ("
        << Utilities::System::get_current_vectorization_level() << ')'
        << std::endl;
  make_fine_grid(3); // 6 global refinements of unit cube
  agglomerate_and_compute_level_matrices();
  pcout << std::endl;
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  const MPI_Comm                   comm = MPI_COMM_WORLD;
  static constexpr unsigned int    dim  = 3;

  if (Utilities::MPI::this_mpi_process(comm) == 0)
    std::cout << "Degree: " << degree_finite_element << std::endl;

  for (unsigned int starting_level = 0; starting_level < 2; ++starting_level)
    {
      AgglomeratedMG<dim> problem(GridType::unstructured,
                                  degree_finite_element,
                                  starting_level,
                                  comm);
      problem.run();
    }
}
