#include <deal.II/base/config.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>

// Trilinos linear algebra is employed for parallel computations
#include <deal.II/base/timer.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <boost/geometry/index/rtree.hpp>

#include <agglomeration_handler.h>
#include <agglomerator.h>
#include <multigrid_amg.h>
#include <poly_utils.h>
#include <utils.h>

#include <algorithm>

using namespace dealii;

// static constexpr double       TOL            = 2.5e-3;
static constexpr double       TOL            = 3e-3;
static constexpr unsigned int starting_level = 2;

namespace Utils
{
  double
  heaviside_sharp(const double &x, const double &x0)
  {
    return x > x0 ? 1.0 : 0.0;
  }

  double
  heaviside(const double &x, const double &x0, const double &k)
  {
    return (0.5 * (1 + std::tanh(k * (x - x0))));
  }

} // namespace Utils

// Model parameters for Bueno
struct ModelParameters
{
  SolverControl control;

  double       penalty_constant   = 10.;
  unsigned int fe_degree          = 1;
  double       dt                 = 1e-2;
  double       final_time         = 1.;
  double       final_time_current = 1.;
  double       chi                = 1;
  double       Cm                 = 1.;
  double       sigma              = 1e-4;
  double       V1                 = 0.3;
  double       V1m                = 0.015;
  double       V2                 = 0.015;
  double       V2m                = 0.03;
  double       V3                 = 0.9087;
  double       Vhat               = 1.58;
  double       Vo                 = 0.006;
  double       Vso                = 0.65;
  double       tauop              = 6e-3;
  double       tauopp             = 6e-3;
  double       tausop             = 43e-3;
  double       tausopp            = 0.2e-3;
  double       tausi              = 2.8723e-3;
  double       taufi              = 0.11e-3;
  double       tau1plus           = 1.4506e-3;
  double       tau2plus           = 0.28;
  double       tau2inf            = 0.07;
  double       tau1p              = 0.06;
  double       tau1pp             = 1.15;
  double       tau2p              = 0.07;
  double       tau2pp             = 0.02;
  double       tau3p              = 2.7342e-3;
  double       tau3pp             = 0.003;
  double       w_star_inf         = 0.94;
  double       k2                 = 65.0;
  double       k3                 = 2.0994;
  double       kso                = 2.0;
  bool         use_amg            = false;
  std::string  mesh_dir           = "../../meshes/idealized_lv.msh";

  // eventually compute activation map at each time step
  bool compute_activation_map = false;
};



template <int dim>
class AppliedCurrent : public Function<dim>
{
public:
  AppliedCurrent(const double final_time_current)
    : Function<dim>()
    // , p1{-0.015598, -0.0173368, 0.0307704}
    // , p2{0.0264292, -0.0043322, 0.0187656}
    // , p3{0.00155326, 0.0252701, 0.0248006}
    , p1{0.0981402, -0.0970197, -0.0406029}
    , p2{0.0981402, -0.0970197, -0.0406029}
    , p3{0.0981402, -0.0970197, -0.0406029}
  {
    t_end_current = final_time_current;
    p.push_back(p1);
    p.push_back(p2);
    p.push_back(p3);
  }

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

  virtual void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<double>           &values,
             const unsigned int /*component*/) const override;

private:
  double                  t_end_current;
  std::vector<Point<dim>> p;
  Point<dim>              p1;
  Point<dim>              p2;
  Point<dim>              p3;
};



template <int dim>
void
AppliedCurrent<dim>::value_list(const std::vector<Point<dim>> &points,
                                std::vector<double>           &values,
                                const unsigned int /*component*/) const
{
  for (unsigned int i = 0; i < values.size(); ++i)
    values[i] = this->value(points[i]);
}

template <int dim>
double
AppliedCurrent<dim>::value(const Point<dim> &point,
                           const unsigned int /*component*/) const
{
  const double t = this->get_time();
  if ((p1.distance(point) < TOL || p2.distance(point) < TOL ||
       p3.distance(point) < TOL) &&
      (t >= 0. && t <= t_end_current))
    {
#ifdef AGGLO_DEBUG
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Applying stimuli" << std::endl;
#endif
      return 300.;
      // return 34.28;
    }
  else
    return 0.;
}


template <int dim, int spacedim, typename MatrixType>
void
fill_interpolation_matrix(
  const AgglomerationHandler<dim, spacedim> &agglomeration_handler,
  MatrixType                                &interpolation_matrix)
{
  Assert((dim == spacedim), ExcNotImplemented());

  using NumberType = typename MatrixType::value_type;
  constexpr bool is_trilinos_matrix =
    std::is_same_v<MatrixType, TrilinosWrappers::MPI::Vector>;

  [[maybe_unused]]
  typename std::conditional_t<!is_trilinos_matrix, SparsityPattern, void *>
    sp;

  // Get some info from the handler
  const DoFHandler<dim, spacedim> &agglo_dh = agglomeration_handler.agglo_dh;

  DoFHandler<dim, spacedim> *output_dh =
    const_cast<DoFHandler<dim, spacedim> *>(&agglomeration_handler.output_dh);
  const FiniteElement<dim, spacedim> &fe = agglomeration_handler.get_fe();
  const Triangulation<dim, spacedim> &tria =
    agglomeration_handler.get_triangulation();
  const auto &bboxes = agglomeration_handler.get_local_bboxes();

  // Setup an auxiliary DoFHandler for output purposes
  output_dh->reinit(tria);
  output_dh->distribute_dofs(fe);

  const IndexSet &locally_owned_dofs = output_dh->locally_owned_dofs();
  const IndexSet  locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(*output_dh);

  const IndexSet &locally_owned_dofs_agglo = agglo_dh.locally_owned_dofs();


  DynamicSparsityPattern dsp(output_dh->n_dofs(),
                             agglo_dh.n_dofs(),
                             locally_relevant_dofs);

  std::vector<types::global_dof_index> agglo_dof_indices(fe.dofs_per_cell);
  std::vector<types::global_dof_index> standard_dof_indices(fe.dofs_per_cell);
  std::vector<types::global_dof_index> output_dof_indices(fe.dofs_per_cell);

  Quadrature<dim>         quad(fe.get_unit_support_points());
  FEValues<dim, spacedim> output_fe_values(fe, quad, update_quadrature_points);

  for (const auto &cell : agglo_dh.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        if (agglomeration_handler.is_master_cell(cell))
          {
            auto slaves = agglomeration_handler.get_slaves_of_idx(
              cell->active_cell_index());
            slaves.emplace_back(cell);

            cell->get_dof_indices(agglo_dof_indices);

            for (const auto &slave : slaves)
              {
                // addd master-slave relationship
                const auto slave_output =
                  slave->as_dof_handler_iterator(*output_dh);
                slave_output->get_dof_indices(output_dof_indices);
                for (const auto row : output_dof_indices)
                  dsp.add_entries(row,
                                  agglo_dof_indices.begin(),
                                  agglo_dof_indices.end());
              }
          }
      }


  const auto assemble_interpolation_matrix = [&]() {
    FullMatrix<NumberType>  local_matrix(fe.dofs_per_cell, fe.dofs_per_cell);
    std::vector<Point<dim>> reference_q_points(fe.dofs_per_cell);

    // Dummy AffineConstraints, only needed for loc2glb
    AffineConstraints<NumberType> c;
    c.close();

    for (const auto &cell : agglo_dh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          if (agglomeration_handler.is_master_cell(cell))
            {
              auto slaves = agglomeration_handler.get_slaves_of_idx(
                cell->active_cell_index());
              slaves.emplace_back(cell);

              cell->get_dof_indices(agglo_dof_indices);

              const types::global_cell_index polytope_index =
                agglomeration_handler.cell_to_polytope_index(cell);

              // Get the box of this agglomerate.
              const BoundingBox<dim> &box = bboxes[polytope_index];

              for (const auto &slave : slaves)
                {
                  // add master-slave relationship
                  const auto slave_output =
                    slave->as_dof_handler_iterator(*output_dh);

                  slave_output->get_dof_indices(output_dof_indices);
                  output_fe_values.reinit(slave_output);

                  local_matrix = 0.;

                  const auto &q_points =
                    output_fe_values.get_quadrature_points();
                  for (const auto i : output_fe_values.dof_indices())
                    {
                      const auto &p = box.real_to_unit(q_points[i]);
                      for (const auto j : output_fe_values.dof_indices())
                        {
                          local_matrix(i, j) = fe.shape_value(j, p);
                        }
                    }
                  c.distribute_local_to_global(local_matrix,
                                               output_dof_indices,
                                               agglo_dof_indices,
                                               interpolation_matrix);
                }
            }
        }
  };


  if constexpr (std::is_same_v<MatrixType, TrilinosWrappers::SparseMatrix>)
    {
      const MPI_Comm &communicator = tria.get_communicator();
      SparsityTools::distribute_sparsity_pattern(dsp,
                                                 locally_owned_dofs,
                                                 communicator,
                                                 locally_relevant_dofs);

      interpolation_matrix.reinit(locally_owned_dofs,
                                  locally_owned_dofs_agglo,
                                  dsp,
                                  communicator);
      assemble_interpolation_matrix();
    }
  else if constexpr (std::is_same_v<MatrixType, SparseMatrix<NumberType>>)
    {
      sp.copy_from(dsp);
      interpolation_matrix.reinit(sp);
      assemble_interpolation_matrix();
    }
  else
    {
      // PETSc, LA::d::v options not implemented.
      (void)agglomeration_handler;
      AssertThrow(false, ExcNotImplemented());
    }

  // If tria is distributed
  if (dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(&tria) !=
      nullptr)
    interpolation_matrix.compress(VectorOperation::add);
}



template <int dim>
class IonicModel
{
private:
  void
  setup_problem();

  void
  setup_multigrid();

  void
  assemble_time_independent_matrix();

  std::array<double, 3>
  alpha(const double u);

  std::array<double, 3>
  beta(const double u);

  std::array<double, 3>
  w_inf(const double u);

  double
  Iion(const double u_old, const std::vector<double> &w) const;

  void
  assemble_time_terms();
  void
  update_w_and_ion();
  void
  solve_w();
  void
  solve();
  void
  output_results();
  void
  compute_error() const;


  const MPI_Comm                                 communicator;
  parallel::fullydistributed::Triangulation<dim> tria;
  MappingQ<dim>                                  mapping;
  FE_DGQ<dim>                                    dg_fe;
  DoFHandler<dim>                                classical_dh;
  ConditionalOStream                             pcout;
  TimerOutput                                    computing_timer;
  SparsityPattern                                sparsity;
  AffineConstraints<double>                      constraints;
  TrilinosWrappers::PreconditionAMG              amg_preconditioner;
  TrilinosWrappers::SparseMatrix                 mass_matrix;
  TrilinosWrappers::SparseMatrix                 laplace_matrix;
  LinearAlgebra::distributed::Vector<double>     system_rhs;
  std::unique_ptr<Function<dim>>                 rhs_function;
  std::unique_ptr<Function<dim>>                 Iext;
  std::unique_ptr<Function<dim>>                 analytical_solution;

  std::unique_ptr<FEValues<dim>>     fe_values;
  std::unique_ptr<FEFaceValues<dim>> fe_faces0;
  std::unique_ptr<FEFaceValues<dim>> fe_faces1;

  IndexSet locally_owned_dofs;
  // Related to update solution
  LinearAlgebra::distributed::Vector<double> locally_relevant_solution_pre;
  LinearAlgebra::distributed::Vector<double> locally_relevant_solution_current;

  LinearAlgebra::distributed::Vector<double> locally_relevant_w0_pre;
  LinearAlgebra::distributed::Vector<double> locally_relevant_w0_current;
  LinearAlgebra::distributed::Vector<double> locally_relevant_w1_pre;
  LinearAlgebra::distributed::Vector<double> locally_relevant_w1_current;
  LinearAlgebra::distributed::Vector<double> locally_relevant_w2_pre;
  LinearAlgebra::distributed::Vector<double> locally_relevant_w2_current;

  LinearAlgebra::distributed::Vector<double> ion_at_dofs;

  // Activation map
  using VectorType = LinearAlgebra::distributed::Vector<double>;
  LinearAlgebra::distributed::Vector<double> activation_map;

  //   Time stepping parameters
  double       time;
  const double dt;
  const double end_time;
  const double end_time_current; // final time external application

  SolverCG<LinearAlgebra::distributed::Vector<double>> solver;
  const ModelParameters                               &param;

  // Agglomeration related
  dealii::RTree<
    std::pair<dealii::BoundingBox<dim, double>,
              dealii::TriaActiveIterator<dealii::CellAccessor<dim, dim>>>,
    boost::geometry::index::rstar<8, 2, 2, 32>,
    boost::geometry::index::indexable<
      std::pair<dealii::BoundingBox<dim, double>,
                dealii::TriaActiveIterator<dealii::CellAccessor<dim, dim>>>>>
    tree;

  unsigned int total_tree_levels;
  std::vector<std::unique_ptr<AgglomerationHandler<dim>>>
    agglomeration_handlers;

  std::vector<TrilinosWrappers::SparseMatrix> injection_matrices_two_level;
  std::unique_ptr<AgglomerationHandler<dim>>  agglomeration_handler_coarse;


  // Multigrid related

  using LevelMatrixType = TrilinosWrappers::SparseMatrix;
  using SmootherType    = PreconditionChebyshev<LevelMatrixType, VectorType>;

  MGLevelObject<TrilinosWrappers::SparseMatrix> multigrid_matrices;
  std::unique_ptr<Multigrid<VectorType>>        mg;
  std::unique_ptr<
    PreconditionMG<dim, VectorType, MGTransferAgglomeration<dim, VectorType>>>
    preconditioner;

  std::unique_ptr<MGLevelObject<TrilinosWrappers::SparseMatrix *>>
                                                            mg_level_transfers;
  std::unique_ptr<MGTransferAgglomeration<dim, VectorType>> mg_transfer;

  std::unique_ptr<mg::Matrix<VectorType>> mg_matrix;

  std::unique_ptr<Utils::MGCoarseDirect<VectorType,
                                        TrilinosWrappers::SparseMatrix,
                                        TrilinosWrappers::SolverDirect>>
    mg_coarse;

  std::unique_ptr<mg::SmootherRelaxation<SmootherType, VectorType>> mg_smoother;

  MGLevelObject<typename SmootherType::AdditionalData> smoother_data;

  std::vector<LinearAlgebra::distributed::Vector<double>> diag_inverses;
  std::vector<TrilinosWrappers::SparseMatrix *>           transfer_matrices;
  std::vector<DoFHandler<dim> *>                          dof_handlers;


public:
  IonicModel(const ModelParameters &parameters);
  void
  run();

  double penalty_constant;
};



template <int dim>
IonicModel<dim>::IonicModel(const ModelParameters &parameters)
  : communicator(MPI_COMM_WORLD)
  , tria(communicator)
  , mapping(1)
  , dg_fe(parameters.fe_degree)
  , classical_dh(tria)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(communicator) == 0)
  , computing_timer(communicator,
                    pcout,
                    TimerOutput::summary,
                    TimerOutput::wall_times)
  , dt(parameters.dt)
  , end_time(parameters.final_time)
  , end_time_current(parameters.final_time_current)
  , solver(const_cast<SolverControl &>(parameters.control))
  , param(parameters)
{
  static_assert(dim == 3);
  time = 0.;
  penalty_constant =
    param.penalty_constant * parameters.fe_degree * (parameters.fe_degree + 1);
  total_tree_levels = 0;
}



template <int dim>
std::array<double, 3>
IonicModel<dim>::alpha(const double u)
{
  std::array<double, 3> a;

  a[0] = (1.0 - Utils::heaviside_sharp(u, param.V1)) /
         (Utils::heaviside_sharp(u, param.V1m) * (param.tau1pp - param.tau1p) +
          param.tau1p);
  a[1] =
    (1.0 - Utils::heaviside_sharp(u, param.V2)) /
    (Utils::heaviside(u, param.V2m, param.k2) * (param.tau2pp - param.tau2p) +
     param.tau2p);
  a[2] =
    1.0 / (Utils::heaviside_sharp(u, param.V2) * (param.tau3pp - param.tau3p) +
           param.tau3p);

  return a;
}



template <int dim>
std::array<double, 3>
IonicModel<dim>::beta(const double u)
{
  std::array<double, 3> b;

  b[0] = -Utils::heaviside_sharp(u, param.V1) / param.tau1plus;
  b[1] = -Utils::heaviside_sharp(u, param.V2) / param.tau2plus;
  b[2] = 0;

  return b;
}



template <int dim>
std::array<double, 3>
IonicModel<dim>::w_inf(const double u)
{
  std::array<double, 3> wi;

  wi[0] = 1.0 - Utils::heaviside_sharp(u, param.V1m);
  wi[1] = Utils::heaviside_sharp(u, param.Vo) *
            (param.w_star_inf - 1.0 + u / param.tau2inf) +
          1.0 - u / param.tau2inf;
  wi[2] = Utils::heaviside(u, param.V3, param.k3);

  return wi;
}



template <int dim>
void
IonicModel<dim>::setup_problem()
{
  TimerOutput::Scope t(computing_timer, "Setup DoFs");

  const unsigned int quadrature_degree = 2 * dg_fe.degree + 1;
  fe_values =
    std::make_unique<FEValues<dim>>(mapping,
                                    dg_fe,
                                    QGauss<dim>(quadrature_degree),
                                    update_values | update_JxW_values |
                                      update_gradients |
                                      update_quadrature_points);
  fe_faces0 =
    std::make_unique<FEFaceValues<dim>>(mapping,
                                        dg_fe,
                                        QGauss<dim - 1>(quadrature_degree),
                                        update_values | update_JxW_values |
                                          update_gradients |
                                          update_quadrature_points |
                                          update_normal_vectors);
  fe_faces1 =
    std::make_unique<FEFaceValues<dim>>(mapping,
                                        dg_fe,
                                        QGauss<dim - 1>(quadrature_degree),
                                        update_values | update_JxW_values |
                                          update_gradients |
                                          update_quadrature_points |
                                          update_normal_vectors);
  classical_dh.distribute_dofs(dg_fe);
  locally_owned_dofs = classical_dh.locally_owned_dofs();
  const IndexSet locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(classical_dh);

  constraints.clear();
  constraints.close();

  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_flux_sparsity_pattern(classical_dh, dsp);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             classical_dh.locally_owned_dofs(),
                                             communicator,
                                             locally_relevant_dofs);

  mass_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, communicator);

  laplace_matrix.reinit(locally_owned_dofs,
                        locally_owned_dofs,
                        dsp,
                        communicator);

  locally_relevant_solution_pre.reinit(locally_owned_dofs, communicator);
  locally_relevant_solution_current.reinit(locally_owned_dofs, communicator);
  system_rhs.reinit(locally_owned_dofs, communicator);

  // Parallel layout of gating variables
  locally_relevant_w0_pre.reinit(locally_owned_dofs, communicator);
  locally_relevant_w0_current.reinit(locally_owned_dofs, communicator);
  locally_relevant_w1_pre.reinit(locally_owned_dofs, communicator);
  locally_relevant_w1_current.reinit(locally_owned_dofs, communicator);
  locally_relevant_w2_pre.reinit(locally_owned_dofs, communicator);
  locally_relevant_w2_current.reinit(locally_owned_dofs, communicator);

  ion_at_dofs.reinit(locally_owned_dofs, communicator);

  activation_map.reinit(locally_owned_dofs, communicator);

  Iext = std::make_unique<AppliedCurrent<dim>>(end_time_current);

  // // Start building R-tree
  namespace bgi = boost::geometry::index;
  static constexpr unsigned int max_elem_per_node =
    PolyUtils::constexpr_pow(2, dim); // 2^dim
  std::vector<std::pair<BoundingBox<dim>,
                        typename Triangulation<dim>::active_cell_iterator>>
               boxes(tria.n_locally_owned_active_cells());
  unsigned int i = 0;
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
      boxes[i++] = std::make_pair(mapping.get_bounding_box(cell), cell);

  tree = pack_rtree<bgi::rstar<max_elem_per_node>>(boxes);

  Assert(n_levels(tree) >= 2, ExcMessage("At least two levels are needed."));
  pcout << "Total number of available levels: " << n_levels(tree) << std::endl;

  pcout << "Starting level: " << starting_level << std::endl;
  total_tree_levels = n_levels(tree) - starting_level + 1;

  multigrid_matrices.resize(0, total_tree_levels);

  multigrid_matrices[multigrid_matrices.max_level()].reinit(locally_owned_dofs,
                                                            locally_owned_dofs,
                                                            dsp,
                                                            communicator);
}


template <int dim>
void
IonicModel<dim>::setup_multigrid()
{
  TimerOutput::Scope t(computing_timer, "Setup polytopal multigrid");

  GridTools::Cache<dim> cached_tria(tria);

  agglomeration_handler_coarse =
    std::make_unique<AgglomerationHandler<dim>>(cached_tria);

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
      // unsigned int       total_agglomerates =
      Utilities::MPI::sum(n_local_agglomerates, communicator);
      // pcout << "Total agglomerates per (tree) level: " << extraction_level
      //<< ": " << total_agglomerates << std::endl;

      // Now, perform agglomeration within each locally owned partition
      std::vector<
        std::vector<typename Triangulation<dim>::active_cell_iterator>>
        cells_per_subdomain(n_local_agglomerates);
      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned())
          cells_per_subdomain[cell->material_id()].push_back(cell);

      // For every subdomain, agglomerate elements together
      for (std::size_t i = 0; i < cells_per_subdomain.size(); ++i)
        agglomeration_handlers[extraction_level - starting_level]
          ->define_agglomerate(cells_per_subdomain[i]);

      agglomeration_handlers[extraction_level - starting_level]
        ->initialize_fe_values(QGauss<dim>(dg_fe.degree + 1),
                               update_values | update_gradients |
                                 update_JxW_values | update_quadrature_points,
                               QGauss<dim - 1>(dg_fe.degree + 1),
                               update_JxW_values);
      agglomeration_handlers[extraction_level - starting_level]
        ->distribute_agglomerated_dofs(dg_fe);
    }

  // Compute two-level transfers between agglomeration handlers
  // pcout << "Fill injection matrices between agglomerated levels" <<
  // std::endl;
  injection_matrices_two_level.resize(total_tree_levels);
  for (unsigned int l = 1; l < total_tree_levels; ++l)
    {
      // pcout << "from level " << l - 1 << " to level " << l << std::endl;
      SparsityPattern sparsity;
      Utils::fill_injection_matrix(*agglomeration_handlers[l - 1],
                                   *agglomeration_handlers[l],
                                   sparsity,
                                   injection_matrices_two_level[l - 1]);
    }
  // pcout << "Computed two-level matrices between agglomerated levels"
  //<< std::endl;


  // Define transfer between levels.
  transfer_matrices.resize(total_tree_levels);
  for (unsigned int l = 0; l < total_tree_levels - 1; ++l)
    transfer_matrices[l] = &injection_matrices_two_level[l];


  // Last matrix, fill it by hand
  // add last two-level (which is an embedding)
  fill_interpolation_matrix(*agglomeration_handlers.back(),
                            injection_matrices_two_level.back());
  transfer_matrices[total_tree_levels - 1] =
    &injection_matrices_two_level.back();

  // pcout << injection_matrices_two_level.back().m() << " and "
  //     << injection_matrices_two_level.back().n() << std::endl;

  AmgProjector<dim, TrilinosWrappers::SparseMatrix, double> amg_projector(
    injection_matrices_two_level);
  // pcout << "Initialized projector" << std::endl;

  // multigrid_matrices[multigrid_matrices.max_level()].release();
  // multigrid_matrices[multigrid_matrices.max_level()].reset(&system_matrix);


  amg_projector.compute_level_matrices(multigrid_matrices);

  pcout << "Projected using transfer_matrices:" << std::endl;

  TrilinosWrappers::SparseMatrix &system_matrix =
    multigrid_matrices[multigrid_matrices.max_level()];

  // Setup multigrid


  // Multigrid matrices
  using LevelMatrixType = TrilinosWrappers::SparseMatrix;
  using VectorType      = LinearAlgebra::distributed::Vector<double>;
  mg_matrix = std::make_unique<mg::Matrix<VectorType>>(multigrid_matrices);

  using SmootherType = PreconditionChebyshev<LevelMatrixType, VectorType>;

  smoother_data.resize(0, total_tree_levels + 1);

  // Fine level
  diag_inverses.resize(total_tree_levels + 1);
  diag_inverses.back().reinit(classical_dh.locally_owned_dofs(), communicator);

  // Set exact diagonal for each operator
  for (unsigned int i = system_matrix.local_range().first;
       i < system_matrix.local_range().second;
       ++i)
    diag_inverses.back()[i] = 1. / system_matrix.diag_element(i);

  smoother_data[total_tree_levels].preconditioner =
    std::make_shared<DiagonalMatrix<VectorType>>(diag_inverses.back());
  // pcout << "Start defining smoothers data" << std::endl;

  for (unsigned int l = 0; l < total_tree_levels; ++l)
    {
      // pcout << "l = " << l << std::endl;
      diag_inverses[l].reinit(
        agglomeration_handlers[l]->agglo_dh.locally_owned_dofs(), communicator);

      // Set exact diagonal for each operator
      for (unsigned int i = multigrid_matrices[l].local_range().first;
           i < multigrid_matrices[l].local_range().second;
           ++i)
        diag_inverses[l][i] = 1. / multigrid_matrices[l].diag_element(i);

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
          smoother_data[0].degree = 5; // numbers::invalid_unsigned_int;
          smoother_data[0].eig_cg_n_iterations = classical_dh.n_dofs();
          smoother_data[0].eig_cg_n_iterations = multigrid_matrices[0].m();
        }
    }

  mg_smoother =
    std::make_unique<mg::SmootherRelaxation<SmootherType, VectorType>>();
  mg_smoother->initialize(multigrid_matrices, smoother_data);

  pcout << "Smoothers initialized" << std::endl;

  // Define coarse grid solver
  const unsigned int min_level = 0;

  mg_coarse =
    std::make_unique<Utils::MGCoarseDirect<VectorType,
                                           TrilinosWrappers::SparseMatrix,
                                           TrilinosWrappers::SolverDirect>>(
      multigrid_matrices[min_level]);

  pcout << "Coarse solver initialized" << std::endl;

  // Transfers
  mg_level_transfers =
    std::make_unique<MGLevelObject<TrilinosWrappers::SparseMatrix *>>(
      0, total_tree_levels);
  for (unsigned int l = 0; l < total_tree_levels; ++l)
    (*mg_level_transfers)[l] = transfer_matrices[l];


  dof_handlers.resize(total_tree_levels + 1);
  for (unsigned int l = 0; l < dof_handlers.size() - 1; ++l)
    dof_handlers[l] = &agglomeration_handlers[l]->agglo_dh;
  dof_handlers[dof_handlers.size() - 1] = &classical_dh; // fine

  unsigned int lev = 0;
  for (const auto &dh : dof_handlers)
    pcout << "Number of DoFs in level " << lev++ << ": " << dh->n_dofs()
          << std::endl;

  mg_transfer = std::make_unique<MGTransferAgglomeration<dim, VectorType>>(
    *mg_level_transfers, dof_handlers);
  // pcout << "MG transfers initialized" << std::endl;

  // Define multigrid object and convert to preconditioner.
  mg = std::make_unique<Multigrid<VectorType>>(*mg_matrix,
                                               *mg_coarse,
                                               *mg_transfer,
                                               *mg_smoother,
                                               *mg_smoother,
                                               min_level,
                                               numbers::invalid_unsigned_int,
                                               Multigrid<VectorType>::v_cycle);


  preconditioner = std::make_unique<
    PreconditionMG<dim, VectorType, MGTransferAgglomeration<dim, VectorType>>>(
    classical_dh, *mg, *mg_transfer);
}



template <int dim>
double
IonicModel<dim>::Iion(const double u_old, const std::vector<double> &w) const
{
  double Iion_val =
    Utils::heaviside_sharp(u_old, param.V1) * (u_old - param.V1) *
      (param.Vhat - u_old) * w[0] / param.taufi -
    (1.0 - Utils::heaviside_sharp(u_old, param.V2)) * (u_old - 0.) /
      (Utils::heaviside_sharp(u_old, param.Vo) * (param.tauopp - param.tauop) +
       param.tauop) -
    Utils::heaviside_sharp(u_old, param.V2) /
      (Utils::heaviside(u_old, param.Vso, param.kso) *
         (param.tausopp - param.tausop) +
       param.tausop) +
    Utils::heaviside_sharp(u_old, param.V2) * w[1] * w[2] / param.tausi;

  Iion_val = -Iion_val;

  return Iion_val;
}



template <int dim>
void
IonicModel<dim>::update_w_and_ion()
{
  TimerOutput::Scope t(computing_timer, "Update w and ion at DoFs");

  // update w from t_n to t_{n+1} on the locally owned DoFs for all w's
  // On top of that, evaluate Iion at DoFs
  for (const types::global_dof_index i : locally_owned_dofs)
    {
      // First, update w's
      std::array<double, 3> a      = alpha(locally_relevant_solution_pre[i]);
      std::array<double, 3> b      = beta(locally_relevant_solution_pre[i]);
      std::array<double, 3> w_infs = w_inf(locally_relevant_solution_pre[i]);

      locally_relevant_w0_current[i] =
        locally_relevant_w0_pre[i] +
        dt * ((b[0] - a[0]) * locally_relevant_w0_pre[i] + a[0] * w_infs[0]);

      locally_relevant_w1_current[i] =
        locally_relevant_w1_pre[i] +
        dt * ((b[1] - a[1]) * locally_relevant_w1_pre[i] + a[1] * w_infs[1]);

      locally_relevant_w2_current[i] =
        locally_relevant_w2_pre[i] +
        dt * ((b[2] - a[2]) * locally_relevant_w2_pre[i] + a[2] * w_infs[2]);

      // Evaluate ion at u_n, w_{n+1}
      ion_at_dofs[i] = Iion(locally_relevant_solution_pre[i],
                            {locally_relevant_w0_current[i],
                             locally_relevant_w1_current[i],
                             locally_relevant_w2_current[i]});
    }
}



/*
 * Assemble the time independent block chi*c*M/dt + A
 */
template <int dim>
void
IonicModel<dim>::assemble_time_independent_matrix()
{
  TimerOutput::Scope t(computing_timer, "Assemble time independent terms");

  const unsigned int dofs_per_cell = dg_fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);

  //   Jump matrices needed for DG
  FullMatrix<double> M11(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> M12(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> M21(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> M22(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


  // Loop over standard deal.II cells
  for (const auto &cell : classical_dh.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          cell_matrix      = 0.;
          cell_mass_matrix = 0.;
          fe_values->reinit(cell);

          cell->get_dof_indices(local_dof_indices);

          for (unsigned int q_index : fe_values->quadrature_point_indices())
            {
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      cell_matrix(i, j) += param.sigma *
                                           fe_values->shape_grad(i, q_index) *
                                           fe_values->shape_grad(j, q_index) *
                                           fe_values->JxW(q_index);

                      cell_mass_matrix(i, j) +=
                        (1. / dt) * fe_values->shape_value(i, q_index) *
                        fe_values->shape_value(j, q_index) *
                        fe_values->JxW(q_index);
                    }
                }
            }

          for (const auto f : cell->face_indices())
            {
              double       hf      = 0.;
              const double extent1 = cell->measure() / cell->face(f)->measure();

              // Do nothing at boundary, we have Neumann homogeneous BC
              if (!cell->face(f)->at_boundary())
                {
                  const auto &neigh_cell = cell->neighbor(f);

                  if (cell->global_active_cell_index() <
                      neigh_cell->global_active_cell_index())
                    {
                      const double extent2 =
                        neigh_cell->measure() /
                        neigh_cell->face(cell->neighbor_of_neighbor(f))
                          ->measure();
                      hf = (1. / extent1 + 1. / extent2);
                      fe_faces0->reinit(cell, f);
                      fe_faces1->reinit(neigh_cell,
                                        cell->neighbor_of_neighbor(f));

                      std::vector<types::global_dof_index>
                        local_dof_indices_neighbor(dofs_per_cell);

                      M11 = 0.;
                      M12 = 0.;
                      M21 = 0.;
                      M22 = 0.;

                      const auto &normals = fe_faces0->get_normal_vectors();
                      // M11
                      for (unsigned int q_index :
                           fe_faces0->quadrature_point_indices())
                        {
                          for (unsigned int i = 0; i < dofs_per_cell; ++i)
                            {
                              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                                {
                                  M11(i, j) +=
                                    +param.sigma *
                                    (-0.5 * fe_faces0->shape_grad(i, q_index) *
                                       normals[q_index] *
                                       fe_faces0->shape_value(j, q_index) -
                                     0.5 * fe_faces0->shape_grad(j, q_index) *
                                       normals[q_index] *
                                       fe_faces0->shape_value(i, q_index) +
                                     (penalty_constant * hf) *
                                       fe_faces0->shape_value(i, q_index) *
                                       fe_faces0->shape_value(j, q_index)) *
                                    fe_faces0->JxW(q_index);

                                  M12(i, j) +=
                                    +param.sigma *
                                    (0.5 * fe_faces0->shape_grad(i, q_index) *
                                       normals[q_index] *
                                       fe_faces1->shape_value(j, q_index) -
                                     0.5 * fe_faces1->shape_grad(j, q_index) *
                                       normals[q_index] *
                                       fe_faces0->shape_value(i, q_index) -
                                     (penalty_constant * hf) *
                                       fe_faces0->shape_value(i, q_index) *
                                       fe_faces1->shape_value(j, q_index)) *
                                    fe_faces1->JxW(q_index);

                                  // A10
                                  M21(i, j) +=
                                    +param.sigma *
                                    (-0.5 * fe_faces1->shape_grad(i, q_index) *
                                       normals[q_index] *
                                       fe_faces0->shape_value(j, q_index) +
                                     0.5 * fe_faces0->shape_grad(j, q_index) *
                                       normals[q_index] *
                                       fe_faces1->shape_value(i, q_index) -
                                     (penalty_constant * hf) *
                                       fe_faces1->shape_value(i, q_index) *
                                       fe_faces0->shape_value(j, q_index)) *
                                    fe_faces1->JxW(q_index);

                                  // A11
                                  M22(i, j) +=
                                    +param.sigma *
                                    (0.5 * fe_faces1->shape_grad(i, q_index) *
                                       normals[q_index] *
                                       fe_faces1->shape_value(j, q_index) +
                                     0.5 * fe_faces1->shape_grad(j, q_index) *
                                       normals[q_index] *
                                       fe_faces1->shape_value(i, q_index) +
                                     (penalty_constant * hf) *
                                       fe_faces1->shape_value(i, q_index) *
                                       fe_faces1->shape_value(j, q_index)) *
                                    fe_faces1->JxW(q_index);
                                }
                            }
                        }

                      // distribute DoFs accordingly

                      neigh_cell->get_dof_indices(local_dof_indices_neighbor);

                      constraints.distribute_local_to_global(M11,
                                                             local_dof_indices,
                                                             laplace_matrix);
                      constraints.distribute_local_to_global(
                        M12,
                        local_dof_indices,
                        local_dof_indices_neighbor,
                        laplace_matrix);
                      constraints.distribute_local_to_global(
                        M21,
                        local_dof_indices_neighbor,
                        local_dof_indices,
                        laplace_matrix);
                      constraints.distribute_local_to_global(
                        M22, local_dof_indices_neighbor, laplace_matrix);

                    } // check idx neighbors
                }     // over faces
            }
          constraints.distribute_local_to_global(cell_matrix,
                                                 local_dof_indices,
                                                 laplace_matrix);
          constraints.distribute_local_to_global(cell_mass_matrix,
                                                 local_dof_indices,
                                                 mass_matrix);
        }
    }
  mass_matrix.compress(VectorOperation::add);
  laplace_matrix.compress(VectorOperation::add);
}



template <int dim>
void
IonicModel<dim>::assemble_time_terms()
{
  TimerOutput::Scope t(computing_timer, "Assemble time dependent terms");

  const unsigned int dofs_per_cell = dg_fe.n_dofs_per_cell();
  Vector<double>     cell_rhs(dofs_per_cell);
  Vector<double>     cell_ion(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Loop over standard deal.II cells
  for (const auto &cell : classical_dh.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          cell_rhs = 0.;
          cell_ion = 0.;

          fe_values->reinit(cell);

          const auto        &q_points  = fe_values->get_quadrature_points();
          const unsigned int n_qpoints = q_points.size();

          std::vector<double> applied_currents(n_qpoints);
          Iext->value_list(q_points, applied_currents);

          std::vector<double> ion_at_qpoints(n_qpoints);
          fe_values->get_function_values(ion_at_dofs, ion_at_qpoints);

          // Get local values of current solution, to be used inside the non
          // linear reaction matrix

          cell->get_dof_indices(local_dof_indices);

          for (unsigned int q_index : fe_values->quadrature_point_indices())
            {
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  cell_rhs(i) +=
                    (applied_currents[q_index] - ion_at_qpoints[q_index]) *
                    fe_values->shape_value(i, q_index) *
                    fe_values->JxW(q_index);
                }
            }

          constraints.distribute_local_to_global(cell_rhs,
                                                 local_dof_indices,
                                                 system_rhs);
        }
    }
  system_rhs.compress(VectorOperation::add);
}



template <int dim>
void
IonicModel<dim>::solve()
{
  TimerOutput::Scope t(computing_timer, "Solve");

  TrilinosWrappers::SparseMatrix &system_matrix =
    multigrid_matrices[multigrid_matrices.max_level()];

  if (param.use_amg)
    solver.solve(system_matrix,
                 locally_relevant_solution_current,
                 system_rhs,
                 amg_preconditioner);
  else
    solver.solve(system_matrix,
                 locally_relevant_solution_current,
                 system_rhs,
                 *preconditioner);


  // #ifdef AGGLO_DEBUG
  pcout << "Number of outer iterations: " << param.control.last_step()
        << std::endl;
  // #endif
}



template <int dim>
void
IonicModel<dim>::output_results()
{
  TimerOutput::Scope t(computing_timer, "Output results");

  DataOut<dim> data_out;
  data_out.attach_dof_handler(classical_dh);
  data_out.add_data_vector(locally_relevant_solution_current,
                           "transmembrane_potential",
                           DataOut<dim>::type_dof_data);

  data_out.add_data_vector(locally_relevant_w0_current,
                           "gating_variable",
                           DataOut<dim>::type_dof_data);

  Vector<float> subdomain(tria.n_active_cells());

  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = tria.locally_owned_subdomain();

  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches(mapping);

  const std::string filename =
    ("3Dmonodomain_time_" + std::to_string(time) +
     Utilities::int_to_string(tria.locally_owned_subdomain(), 4));

  std::ofstream output((filename + ".vtu").c_str());
  data_out.write_vtu(output);

  if (Utilities::MPI::this_mpi_process(communicator) == 0)
    {
      std::vector<std::string> filenames;
      for (unsigned int i = 0;
           i < Utilities::MPI::n_mpi_processes(communicator);
           i++)
        {
          filenames.push_back("3Dmonodomain_time_" + std::to_string(time) +
                              Utilities::int_to_string(i, 4) + ".vtu");
        }
      std::ofstream master_output("3Dmonodomain_time_" + std::to_string(time) +
                                  ".pvtu");
      data_out.write_pvtu_record(master_output, filenames);
    }
}


template <int dim>
void
IonicModel<dim>::compute_error() const
{
  pcout << "Computing error: " << std::endl;
  Vector<double> cellwise_error(locally_relevant_solution_current.size());
  VectorTools::integrate_difference(mapping,
                                    classical_dh,
                                    locally_relevant_solution_current,
                                    *analytical_solution,
                                    cellwise_error,
                                    QGauss<dim>(2 * dg_fe.degree + 1),
                                    VectorTools::NormType::L2_norm);
  const double error =
    VectorTools::compute_global_error(tria,
                                      cellwise_error,
                                      VectorTools::NormType::L2_norm);

  pcout << "L2 norm of error at t= " << time << " is " << error << std::endl;
}

template <int dim>
void
IonicModel<dim>::run()
{
  pcout << "Running on " << Utilities::MPI::n_mpi_processes(communicator)
        << " MPI rank(s)." << std::endl;

  // Create mesh
  Triangulation<dim> tria_dummy;
  GridIn<dim>        grid_in;
  grid_in.attach_triangulation(tria_dummy);
  std::ifstream mesh_file(param.mesh_dir); // idealized mesh of left ventricle
  grid_in.read_msh(mesh_file);

  // scale triangulation by a suitable factor in order to work with mm
  const double scale_factor = 1e-3;
  GridTools::scale(scale_factor, tria_dummy);

  // Partition serial triangulation:
  const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(communicator);
  GridTools::partition_triangulation(n_ranks, tria_dummy);

  // Create building blocks:
  const TriangulationDescription::Description<dim, dim> description =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      tria_dummy, communicator);

  tria.create_triangulation(description);
  pcout << "   Number of active cells:       " << tria.n_global_active_cells()
        << std::endl;

  setup_problem();
  pcout << "   Number of degrees of freedom: " << classical_dh.n_dofs()
        << std::endl;

  // Set initial conditions
  locally_relevant_solution_pre = -84e-3;
  // locally_relevant_solution_pre     = 0.;
  locally_relevant_solution_current = locally_relevant_solution_pre;

  locally_relevant_w0_pre     = 1.;
  locally_relevant_w0_current = locally_relevant_w0_pre;

  locally_relevant_w1_pre     = 1.;
  locally_relevant_w1_current = locally_relevant_w1_pre;

  locally_relevant_w2_pre     = 0.;
  locally_relevant_w2_current = locally_relevant_w2_pre;
  output_results();

  // Assemble time independent term
  assemble_time_independent_matrix();
  pcout << "Assembled time independent term: done" << std::endl;

  unsigned int iter_count = 0;

  const auto mesh_size = [this]() -> double {
    double hmax = 0.;
    for (const auto &cell : classical_dh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const double diameter = cell->diameter();
          if (diameter > hmax)
            hmax = diameter;
        }
    return hmax;
  }();

  pcout << "Max mesh size: " << mesh_size << std::endl;


  TrilinosWrappers::SparseMatrix &system_matrix =
    multigrid_matrices[multigrid_matrices.max_level()];

  system_matrix.copy_from(mass_matrix);   // M/dt
  system_matrix.add(+1., laplace_matrix); // M/dt + A


  // Depending on the preconditioner type, use AMG or polytopal multigrid.
  if (param.use_amg)
    amg_preconditioner.initialize(system_matrix);
  else
    setup_multigrid();
  pcout << "Setup multigrid: done " << std::endl;

  while (time <= end_time)
    {
      time += dt;
      Iext->set_time(time);

      // Solve the ODEs for w
      update_w_and_ion();
      //   Assemble time dependent terms
      assemble_time_terms();

      // Build system matrix by adding the time term
      mass_matrix.vmult_add(
        system_rhs,
        locally_relevant_solution_pre); // Add to system_rhs (M/dt)un

      // Solver for transmembrane potential
      solve();
      pcout << "Solved at t= " << time << std::endl;
      ++iter_count;

      // output results every 5 time steps
      if ((iter_count % 10 == 0) || time < param.final_time_current)
        output_results();

      // update solutions
      locally_relevant_solution_pre = locally_relevant_solution_current;
      locally_relevant_w0_pre       = locally_relevant_w0_current;
      locally_relevant_w1_pre       = locally_relevant_w1_current;
      locally_relevant_w2_pre       = locally_relevant_w2_current;

      // reset time dependent terms
      system_rhs = 0.;
    }
  pcout << std::endl;
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  ModelParameters parameters;
  parameters.control.set_tolerance(1e-13); // used in CG solver
  parameters.control.set_max_steps(2000);

  parameters.use_amg = false;
  // parameters.mesh_dir           = "../../meshes/idealized_lv.msh";
  parameters.mesh_dir           = "../../meshes/realistic_lv.msh";
  parameters.fe_degree          = 1;
  parameters.dt                 = 1e-4;
  parameters.final_time         = 0.3;
  parameters.final_time_current = 3e-3;

  IonicModel<3> problem(parameters);
  problem.run();

  return 0;
}
