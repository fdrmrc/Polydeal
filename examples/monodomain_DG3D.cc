#include <deal.II/base/config.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/timer.h>

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

// Trilinos linear algebra is employed for parallel computations
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
#include <filesystem>

using namespace dealii;

#ifndef DEAL_II_WITH_MUMPS
class SparseDirectMUMPS
{};
#endif

// solver-related parameters
static constexpr double       TOL                 = 3e-3;
static constexpr bool         measure_solve_times = true;
static constexpr unsigned int starting_level      = 2;

// matrix-free related parameters
static constexpr bool         use_matrix_free_action = true;
static constexpr unsigned int degree_finite_element  = 1;
static constexpr unsigned int n_qpoints    = degree_finite_element + 1;
static constexpr unsigned int n_components = 1;


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

  template <typename MatrixType,
            typename Number,
            typename PreconditionerType = TrilinosWrappers::PreconditionAMG>
  class MGCoarseIterative
    : public MGCoarseGridBase<LinearAlgebra::distributed::Vector<Number>>
  {
  public:
    MGCoarseIterative()
    {
      reduction_control =
        std::make_unique<ReductionControl>(100, 1e-5, 1e-6, false, false);
      solver_coarse =
        std::make_unique<SolverCG<LinearAlgebra::distributed::Vector<double>>>(
          *reduction_control);
    }

    void
    initialize(const MatrixType &matrix)
    {
      coarse_matrix = &matrix;
      if constexpr (std::is_same_v<PreconditionerType,
                                   TrilinosWrappers::PreconditionAMG>)
        {
          precondition = std::make_unique<TrilinosWrappers::PreconditionAMG>();
          precondition->initialize(*coarse_matrix);
        }
      else
        {
          DEAL_II_NOT_IMPLEMENTED();
        }
    }

    virtual void
    operator()(
      const unsigned int                                level,
      LinearAlgebra::distributed::Vector<double>       &dst,
      const LinearAlgebra::distributed::Vector<double> &src) const override
    {
      (void)level;
      [[maybe_unused]] double start, stop;
      if constexpr (std::is_same_v<PreconditionerType,
                                   TrilinosWrappers::PreconditionAMG>)
        {
#ifdef AGGLO_DEBUG
          start = MPI_Wtime();
#endif
          solver_coarse->solve(*coarse_matrix, dst, src, *precondition);
          // precondition->vmult(dst, src);
#ifdef AGGLO_DEBUG
          stop = MPI_Wtime();
#endif
        }
      else
        {
          DEAL_II_NOT_IMPLEMENTED();
        }

#ifdef AGGLO_DEBUG
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Coarse solver elapsed time: " << stop - start << "[s]"
                  << std::endl;
#endif
    }

    const MatrixType                   *coarse_matrix;
    std::unique_ptr<PreconditionerType> precondition;
    std::unique_ptr<ReductionControl>   reduction_control;
    std::unique_ptr<SolverCG<LinearAlgebra::distributed::Vector<double>>>
      solver_coarse;
  };

} // namespace Utils



enum class TestCase
{
  Idealized,
  Realistic
};



enum class Preconditioner
{
  AMG,
  AGGLOMG
};


// Model parameters for Bueno Orovio
struct ModelParameters
{
  SolverControl control;

  double                       penalty_constant   = 10.;
  unsigned int                 fe_degree          = 1;
  double                       dt                 = 1e-2;
  double                       final_time         = 1.;
  double                       final_time_current = 1.;
  double                       chi                = 1;
  double                       Cm                 = 1.;
  double                       sigma              = 1e-4;
  double                       V1                 = 0.3;
  double                       V1m                = 0.015;
  double                       V2                 = 0.015;
  double                       V2m                = 0.03;
  double                       V3                 = 0.9087;
  double                       Vhat               = 1.58;
  double                       Vo                 = 0.006;
  double                       Vso                = 0.65;
  double                       tauop              = 6e-3;
  double                       tauopp             = 6e-3;
  double                       tausop             = 43e-3;
  double                       tausopp            = 0.2e-3;
  double                       tausi              = 2.8723e-3;
  double                       taufi              = 0.11e-3;
  double                       tau1plus           = 1.4506e-3;
  double                       tau2plus           = 0.28;
  double                       tau2inf            = 0.07;
  double                       tau1p              = 0.06;
  double                       tau1pp             = 1.15;
  double                       tau2p              = 0.07;
  double                       tau2pp             = 0.02;
  double                       tau3p              = 2.7342e-3;
  double                       tau3pp             = 0.003;
  double                       w_star_inf         = 0.94;
  double                       k2                 = 65.0;
  double                       k3                 = 2.0994;
  double                       kso                = 2.0;
  Preconditioner               preconditioner     = Preconditioner::AMG;
  Utils::Physics::TimeStepping time_stepping_scheme =
    Utils::Physics::TimeStepping::BDF2;
  std::string  output_directory          = ".";
  unsigned int output_frequency          = 1;
  bool         compute_min_value         = false;
  bool         estimate_condition_number = false;

  enum TestCase test_case = TestCase::Idealized;
};



template <int dim>
class AppliedCurrent : public Function<dim>
{
public:
  AppliedCurrent(const double final_time_current, const TestCase test_case)
    : Function<dim>()
    , p1{test_case == TestCase::Idealized ?
           Point<dim>{-0.015598, -0.0173368, 0.0307704} :
           Point<dim>{0.0981402, -0.0970197, -0.0406029}}
    , p2{test_case == TestCase::Idealized ?
           Point<dim>{0.0264292, -0.0043322, 0.0187656} :
           Point<dim>{0.0981402, -0.0580452, -0.000723225}}
    , p3{test_case == TestCase::Idealized ?
           Point<dim>{0.00155326, 0.0252701, 0.0248006} :
           Point<dim>{0.0770339, -0.101529, -0.00292254}}

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

  const std::vector<Point<dim>> &
  get_points() const;

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


template <int dim>
const std::vector<Point<dim>> &
AppliedCurrent<dim>::get_points() const
{
  return p;
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
      const MPI_Comm &communicator = tria.get_mpi_communicator();
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
class MonodomainProblem
{
private:
  void
  setup_problem();

  void
  setup_multigrid();

  void
  assemble_time_independent_matrix_bdf1();

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
  solve();
  void
  output_results();


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
  // TrilinosWrappers::SparseMatrix                 mass_matrix;
  TrilinosWrappers::SparseMatrix system_matrix;
  TrilinosWrappers::SparseMatrix bdf1_matrix;

  std::unique_ptr<
    Utils::MatrixFreeOperators::
      MonodomainOperatorDG<dim, degree_finite_element, n_qpoints, n_components>>
    monodomain_operator;
  LinearOperator<LinearAlgebra::distributed::Vector<double>> system_operator;

  LinearAlgebra::distributed::Vector<double> system_rhs;
  std::unique_ptr<Function<dim>>             rhs_function;
  std::unique_ptr<Function<dim>>             Iext;

  std::unique_ptr<FEValues<dim>>     fe_values;
  std::unique_ptr<FEFaceValues<dim>> fe_faces0;
  std::unique_ptr<FEFaceValues<dim>> fe_faces1;

  IndexSet locally_owned_dofs;

  // Related to update solution
  using VectorType = LinearAlgebra::distributed::Vector<double>;

  // u_n, u_{n-1}, u_{n+1}
  VectorType locally_relevant_solution_nm1;
  VectorType locally_relevant_solution_n;
  VectorType locally_relevant_solution_np1;

  VectorType extrapoled_solution;

  VectorType locally_relevant_w0_np1; // w_{n+1}
  VectorType locally_relevant_w0_n;   // w_n
  VectorType locally_relevant_w0_nm1; // w_{n-1}

  // Same for other gating variables
  VectorType locally_relevant_w1_n;
  VectorType locally_relevant_w1_np1;
  VectorType locally_relevant_w1_nm1;

  VectorType locally_relevant_w2_n;
  VectorType locally_relevant_w2_np1;
  VectorType locally_relevant_w2_nm1;

  VectorType ion_at_dofs;

  //   Time stepping parameters
  double       time;
  const double dt;
  const double end_time;
  const double end_time_current; // final time external application

  SolverCG<VectorType> solver;
  // SolverFlexibleCG<VectorType> solver;
  const ModelParameters &param;

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


  // Multigrid related types
  using LevelMatrixType  = LinearOperatorMG<VectorType, VectorType>;
  using SmootherType     = PreconditionChebyshev<LevelMatrixType, VectorType>;
  using CoarseSolverType = TrilinosWrappers::PreconditionAMG;

  MGLevelObject<TrilinosWrappers::SparseMatrix> multigrid_matrices;
  MGLevelObject<LevelMatrixType>                multigrid_matrices_lo;

  std::unique_ptr<Multigrid<VectorType>> mg;
  std::unique_ptr<
    PreconditionMG<dim, VectorType, MGTransferAgglomeration<dim, VectorType>>>
    preconditioner;

  std::unique_ptr<MGLevelObject<TrilinosWrappers::SparseMatrix *>>
                                                            mg_level_transfers;
  std::unique_ptr<MGTransferAgglomeration<dim, VectorType>> mg_transfer;

  std::unique_ptr<mg::Matrix<VectorType>> mg_matrix;

  std::unique_ptr<Utils::MGCoarseIterative<TrilinosWrappers::SparseMatrix,
                                           double,
                                           CoarseSolverType>>
    mg_coarse;

  std::unique_ptr<mg::SmootherRelaxation<SmootherType, VectorType>> mg_smoother;

  MGLevelObject<typename SmootherType::AdditionalData> smoother_data;

  std::vector<VectorType>                       diag_inverses;
  std::vector<TrilinosWrappers::SparseMatrix *> transfer_matrices;
  std::vector<DoFHandler<dim> *>                dof_handlers;

  Utils::Physics::BilinearFormParameters bilinear_form_parameters;

  TableHandler              statistics_table;
  std::vector<unsigned int> iterations;
  std::vector<double>       iteration_times;
  double                    start_solver, stop_solver;

public:
  MonodomainProblem(const ModelParameters &parameters);
  void
  run();

  double penalty_constant;
};



template <int dim>
MonodomainProblem<dim>::MonodomainProblem(const ModelParameters &parameters)
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
    (double)(std::max(1u, degree_finite_element) * (degree_finite_element + 1));
  // param.penalty_constant * parameters.fe_degree * (parameters.fe_degree + 1);
  total_tree_levels = 0;

  bilinear_form_parameters.dt               = dt;
  bilinear_form_parameters.penalty_constant = penalty_constant;
  bilinear_form_parameters.chi              = param.chi;
  bilinear_form_parameters.Cm               = param.Cm;
  bilinear_form_parameters.sigma            = param.sigma;
  bilinear_form_parameters.time_stepping    = param.time_stepping_scheme;

  monodomain_operator = std::make_unique<
    Utils::MatrixFreeOperators::MonodomainOperatorDG<dim,
                                                     degree_finite_element,
                                                     n_qpoints,
                                                     n_components>>(
    bilinear_form_parameters);

  iterations.reserve(static_cast<unsigned int>(end_time / dt));
  iteration_times.reserve(static_cast<unsigned int>(end_time / dt));

  if (std::filesystem::exists(param.output_directory))
    {
      Assert(std::filesystem::is_directory(param.output_directory),
             ExcMessage("You specified <" + param.output_directory +
                        "> as the output directory in the input file, "
                        "but this is not in fact a directory."));
    }
  else
    std::filesystem::create_directory(param.output_directory);

  if (param.estimate_condition_number)
    solver.connect_condition_number_slot(std::bind(
      [](double input, const std::string &text) {
        std::cout << text << input << std::endl;
      },
      std::placeholders::_1,
      "Condition number estimate: "));
}



template <int dim>
std::array<double, 3>
MonodomainProblem<dim>::alpha(const double u)
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
MonodomainProblem<dim>::beta(const double u)
{
  std::array<double, 3> b;

  b[0] = -Utils::heaviside_sharp(u, param.V1) / param.tau1plus;
  b[1] = -Utils::heaviside_sharp(u, param.V2) / param.tau2plus;
  b[2] = 0;

  return b;
}



template <int dim>
std::array<double, 3>
MonodomainProblem<dim>::w_inf(const double u)
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
MonodomainProblem<dim>::setup_problem()
{
  TimerOutput::Scope t(computing_timer, "Setup DoFs");

  const unsigned int quadrature_degree = dg_fe.degree + 1;
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
                                          update_normal_vectors |
                                          update_inverse_jacobians);
  fe_faces1 =
    std::make_unique<FEFaceValues<dim>>(mapping,
                                        dg_fe,
                                        QGauss<dim - 1>(quadrature_degree),
                                        update_values | update_JxW_values |
                                          update_gradients |
                                          update_quadrature_points |
                                          update_normal_vectors |
                                          update_inverse_jacobians);
  classical_dh.distribute_dofs(dg_fe);
  locally_owned_dofs = classical_dh.locally_owned_dofs();
  const IndexSet locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(classical_dh);

  constraints.clear();
  constraints.close();

  statistics_table.add_value("N. MPI processes",
                             Utilities::MPI::n_mpi_processes(communicator));
  statistics_table.add_value("N. cells", tria.n_global_active_cells());
  statistics_table.add_value("N. DoFs", classical_dh.n_dofs());
  statistics_table.add_value("FE degree", degree_finite_element);

  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_flux_sparsity_pattern(classical_dh, dsp);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             classical_dh.locally_owned_dofs(),
                                             communicator,
                                             locally_relevant_dofs);

  // mass_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp,
  // communicator);
  if (param.preconditioner == Preconditioner::AMG)
    system_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         dsp,
                         communicator);

  bdf1_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, communicator);

  locally_relevant_solution_n.reinit(locally_owned_dofs, communicator);
  locally_relevant_solution_np1.reinit(locally_owned_dofs, communicator);
  extrapoled_solution.reinit(locally_owned_dofs, communicator);
  system_rhs.reinit(locally_owned_dofs, communicator);

  // Parallel layout of gating variables
  locally_relevant_w0_n.reinit(locally_owned_dofs, communicator);
  locally_relevant_w0_np1.reinit(locally_owned_dofs, communicator);
  locally_relevant_w0_nm1.reinit(locally_owned_dofs, communicator);

  locally_relevant_w1_n.reinit(locally_owned_dofs, communicator);
  locally_relevant_w1_np1.reinit(locally_owned_dofs, communicator);
  locally_relevant_w1_nm1.reinit(locally_owned_dofs, communicator);

  locally_relevant_w2_nm1.reinit(locally_owned_dofs, communicator);
  locally_relevant_w2_n.reinit(locally_owned_dofs, communicator);
  locally_relevant_w2_np1.reinit(locally_owned_dofs, communicator);

  ion_at_dofs.reinit(locally_owned_dofs, communicator);

  Iext =
    std::make_unique<AppliedCurrent<dim>>(end_time_current, param.test_case);

  // // Start building R-tree

  namespace bgi = boost::geometry::index;
  {
    TimerOutput::Scope t(computing_timer, "[R-tree]: Build data structure");
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
  }

  Assert(n_levels(tree) >= 2, ExcMessage("At least two levels are needed."));
  pcout << "Total number of available levels: " << n_levels(tree) << std::endl;

  pcout << "Starting level: " << starting_level << std::endl;
  total_tree_levels = n_levels(tree) - starting_level + 1;

  multigrid_matrices.resize(0, total_tree_levels);
  // The layout of the coarser level will be computed a'la AMG. Hence, we only
  // reinit the finest matrix of the problem
  multigrid_matrices[multigrid_matrices.max_level()].reinit(locally_owned_dofs,
                                                            locally_owned_dofs,
                                                            dsp,
                                                            communicator);
}


template <int dim>
void
MonodomainProblem<dim>::setup_multigrid()
{
  TimerOutput::Scope t(computing_timer, "[R-tree]: Setup polytopal multigrid");

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
      unsigned int                           agglo_index = 0;
      {
        TimerOutput::Scope t(computing_timer, "[R-tree]: Define agglomerates");
        const auto         agglomerates = agglomerator.extract_agglomerates();
        agglomeration_handlers[extraction_level - starting_level]
          ->connect_hierarchy(agglomerator);

        for (const auto &agglo : agglomerates)
          {
            agglomeration_handlers[extraction_level - starting_level]
              ->define_agglomerate(agglo);
            ++agglo_index;
          }
      }
      // Flag elements for agglomeration
      // unsigned int agglo_index = 0;
      // for (unsigned int i = 0; i < agglomerates.size(); ++i)
      //   {
      //     const auto &agglo = agglomerates[i]; // i-th agglomerate
      //     for (const auto &el : agglo)
      //       {
      //         el->set_material_id(agglo_index);
      //       }
      //     ++agglo_index;
      //   }

      const unsigned int n_local_agglomerates = agglo_index;
      unsigned int       total_agglomerates =
        Utilities::MPI::sum(n_local_agglomerates, communicator);
      pcout << "Total agglomerates per (tree) level: " << extraction_level
            << ": " << total_agglomerates << std::endl;

      // Now, perform agglomeration within each locally owned partition
      // std::vector<
      //   std::vector<typename Triangulation<dim>::active_cell_iterator>>
      //   cells_per_subdomain(n_local_agglomerates);
      // for (const auto &cell : tria.active_cell_iterators())
      //   if (cell->is_locally_owned())
      //     cells_per_subdomain[cell->material_id()].push_back(cell);

      // For every subdomain, agglomerate elements together
      // for (std::size_t i = 0; i < cells_per_subdomain.size(); ++i)
      //   agglomeration_handlers[extraction_level - starting_level]
      //     ->define_agglomerate(cells_per_subdomain[i]);
      {
        TimerOutput::Scope t(computing_timer,
                             "[R-tree]: Compute transfers matrices");
        agglomeration_handlers[extraction_level - starting_level]
          ->initialize_fe_values(QGauss<dim>(dg_fe.degree + 1),
                                 update_values | update_gradients |
                                   update_JxW_values | update_quadrature_points,
                                 QGauss<dim - 1>(dg_fe.degree + 1),
                                 update_JxW_values);
        agglomeration_handlers[extraction_level - starting_level]
          ->distribute_agglomerated_dofs(dg_fe);
      }
    }

  // Compute two-level transfers between agglomeration handlers
  // pcout << "Fill injection matrices between agglomerated levels" <<
  // std::endl;
  const unsigned int max_level = multigrid_matrices.max_level();

  {
    TimerOutput::Scope t(computing_timer,
                         "[R-tree]: Compute transfers matrices");
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
  }

  // pcout << injection_matrices_two_level.back().m() << " and "
  //     << injection_matrices_two_level.back().n() << std::endl;

  AmgProjector<dim, TrilinosWrappers::SparseMatrix, double> amg_projector(
    injection_matrices_two_level);


  // Get fine operator and use it to build other levels.
  // Once the level operators are built, use the finest in a matrix-free way,
  // the others matrix-based
  multigrid_matrices_lo.resize(0, max_level);


  {
    TimerOutput::Scope t(computing_timer, "[R-tree]: Compute level operators");
    amg_projector.compute_level_matrices_as_linear_operators(
      multigrid_matrices, multigrid_matrices_lo);
  }

  if constexpr (use_matrix_free_action)
    multigrid_matrices_lo[max_level] =
      linear_operator_mg<VectorType, VectorType>(*monodomain_operator);
  else
    multigrid_matrices_lo[max_level] =
      linear_operator_mg<VectorType, VectorType>(multigrid_matrices[max_level]);

  multigrid_matrices_lo[max_level].n_rows = monodomain_operator->m();
  multigrid_matrices_lo[max_level].n_cols = monodomain_operator->n();
  pcout << "Projected using transfer_matrices:" << std::endl;


  // Setup multigrid

  // Multigrid matrices
  mg_matrix = std::make_unique<mg::Matrix<VectorType>>(multigrid_matrices_lo);

  smoother_data.resize(0, total_tree_levels + 1);

  // Fine level
  diag_inverses.resize(total_tree_levels + 1);
  diag_inverses.back().reinit(classical_dh.locally_owned_dofs(), communicator);

  // Set exact diagonal for each operator
  for (unsigned int i = multigrid_matrices[max_level].local_range().first;
       i < multigrid_matrices[max_level].local_range().second;
       ++i)
    diag_inverses.back()[i] =
      1. / multigrid_matrices[max_level].diag_element(i);

  smoother_data[total_tree_levels].preconditioner =
    std::make_shared<DiagonalMatrix<VectorType>>(diag_inverses.back());
  // Start defining smoothers data

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
          smoother_data[level].degree              = 3;   // 5;
          smoother_data[level].eig_cg_n_iterations = 20;
        }
      else
        {
          smoother_data[0].smoothing_range = 1e-3;
          smoother_data[0].degree = 3; // numbers::invalid_unsigned_int;
          smoother_data[0].eig_cg_n_iterations = classical_dh.n_dofs();
        }
    }

  mg_smoother =
    std::make_unique<mg::SmootherRelaxation<SmootherType, VectorType>>();
  mg_smoother->set_steps(3);
  mg_smoother->initialize(multigrid_matrices_lo, smoother_data);

  pcout << "Smoothers initialized" << std::endl;

  // Define coarse grid solver
  const unsigned int min_level = 0;
  mg_coarse =
    std::make_unique<Utils::MGCoarseIterative<TrilinosWrappers::SparseMatrix,
                                              double,
                                              CoarseSolverType>>();
  mg_coarse->initialize(multigrid_matrices[min_level]);

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

  // compute operator complexity for AGGLO MG
  double op_complexity = 0.;
  for (unsigned int level = 0; level < total_tree_levels + 1; ++level)
    {
      pcout << "l" << level << std::endl;
      const double nnz_level = multigrid_matrices[level].n_nonzero_elements();
      op_complexity += nnz_level;
    }
  op_complexity /= multigrid_matrices[max_level].n_nonzero_elements();
  pcout << "Operator complexity (AGGLO MG): " << op_complexity << std::endl;

  statistics_table.add_value("N. MG levels", total_tree_levels + 1);
  statistics_table.add_value("MG Op. complexity", op_complexity);
}



template <int dim>
double
MonodomainProblem<dim>::Iion(const double               u_old,
                             const std::vector<double> &w) const
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
MonodomainProblem<dim>::update_w_and_ion()
{
  TimerOutput::Scope t(computing_timer, "Update w and ion at DoFs");

  // update w from t_n to t_{n+1} on the locally owned DoFs for all w's
  // On top of that, evaluate Iion at DoFs

  // We use BDF1 or BDF2 formula for time integration of gating variables
  for (const types::global_dof_index i : locally_owned_dofs)
    {
      // First, update w's

      if (param.time_stepping_scheme == Utils::Physics::TimeStepping::BDF1)
        {
          std::array<double, 3> a      = alpha(locally_relevant_solution_n[i]);
          std::array<double, 3> b      = beta(locally_relevant_solution_n[i]);
          std::array<double, 3> w_infs = w_inf(locally_relevant_solution_n[i]);

          locally_relevant_w0_np1[i] =
            locally_relevant_w0_n[i] +
            dt * ((b[0] - a[0]) * locally_relevant_w0_n[i] + a[0] * w_infs[0]);

          locally_relevant_w1_np1[i] =
            locally_relevant_w1_n[i] +
            dt * ((b[1] - a[1]) * locally_relevant_w1_n[i] + a[1] * w_infs[1]);

          locally_relevant_w2_np1[i] =
            locally_relevant_w2_n[i] +
            dt * ((b[2] - a[2]) * locally_relevant_w2_n[i] + a[2] * w_infs[2]);

          // Evaluate ion at u_n, w_{n+1}
          ion_at_dofs[i] = Iion(locally_relevant_solution_n[i],
                                {locally_relevant_w0_np1[i],
                                 locally_relevant_w1_np1[i],
                                 locally_relevant_w2_np1[i]});
        }
      else if (param.time_stepping_scheme == Utils::Physics::TimeStepping::BDF2)
        {
          std::array<double, 3> a      = alpha(extrapoled_solution[i]);
          std::array<double, 3> b      = beta(extrapoled_solution[i]);
          std::array<double, 3> w_infs = w_inf(extrapoled_solution[i]);

          // Backward Euler for the first step
          if (time == 0)
            {
              locally_relevant_w0_np1[i] =
                (locally_relevant_w0_n[i] + dt * a[0] * w_infs[0]) /
                (1.0 - dt * (b[0] - a[0]));

              locally_relevant_w1_np1[i] =
                (locally_relevant_w1_n[i] + dt * a[1] * w_infs[1]) /
                (1.0 - dt * (b[1] - a[1]));

              locally_relevant_w2_np1[i] =
                (locally_relevant_w2_n[i] + dt * a[2] * w_infs[2]) /
                (1.0 - dt * (b[2] - a[2]));
            }
          else
            {
              locally_relevant_w0_np1[i] = (4.0 * locally_relevant_w0_n[i] -
                                            1.0 * locally_relevant_w0_nm1[i] +
                                            2.0 * dt * (a[0] * w_infs[0])) /
                                           (3. - 2.0 * dt * (b[0] - a[0]));

              locally_relevant_w1_np1[i] = (4.0 * locally_relevant_w1_n[i] -
                                            1.0 * locally_relevant_w1_nm1[i] +
                                            2.0 * dt * (a[1] * w_infs[1])) /
                                           (3. - 2.0 * dt * (b[1] - a[1]));

              locally_relevant_w2_np1[i] = (4.0 * locally_relevant_w2_n[i] -
                                            1.0 * locally_relevant_w2_nm1[i] +
                                            2.0 * dt * (a[2] * w_infs[2])) /
                                           (3. - 2.0 * dt * (b[2] - a[2]));
            }

          // Evaluate ion at u*, w_{n+1}
          ion_at_dofs[i] = Iion(extrapoled_solution[i],
                                {locally_relevant_w0_np1[i],
                                 locally_relevant_w1_np1[i],
                                 locally_relevant_w2_np1[i]});
        }
    }
}



/*
 * Assemble the time independent block chi*c*M/dt + A
 */
template <int dim>
void
MonodomainProblem<dim>::assemble_time_independent_matrix()
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
                  const double Aii =
                    (param.sigma * fe_values->shape_grad(i, q_index) *
                       fe_values->shape_grad(i, q_index) +
                     (param.chi * param.Cm * 1.5 / dt) *
                       fe_values->shape_value(i, q_index) *
                       fe_values->shape_value(i, q_index)) *
                    fe_values->JxW(q_index);

                  const double Mii = (param.chi * param.Cm * 1.5 / dt) *
                                     fe_values->shape_value(i, q_index) *
                                     fe_values->shape_value(i, q_index) *
                                     fe_values->JxW(q_index);

                  cell_matrix(i, i) += Aii;
                  cell_mass_matrix(i, i) += Mii;

                  for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
                    {
                      const double Aij =
                        (param.sigma * fe_values->shape_grad(i, q_index) *
                           fe_values->shape_grad(j, q_index) +
                         (param.chi * param.Cm * 1.5 / dt) *
                           fe_values->shape_value(i, q_index) *
                           fe_values->shape_value(j, q_index)) *
                        fe_values->JxW(q_index);

                      const double Mij = (param.chi * param.Cm * 1.5 / dt) *
                                         fe_values->shape_value(i, q_index) *
                                         fe_values->shape_value(j, q_index) *
                                         fe_values->JxW(q_index);

                      cell_matrix(i, j) += Aij;
                      cell_matrix(j, i) += Aij;
                      cell_mass_matrix(i, j) += Mij;
                      cell_mass_matrix(j, i) += Mij;
                    }
                }
            }

          for (const auto f : cell->face_indices())
            {
              // Do nothing at boundary, we have Neumann homogeneous BC
              if (!cell->face(f)->at_boundary())
                {
                  const auto &neigh_cell = cell->neighbor(f);

                  if (cell->global_active_cell_index() <
                      neigh_cell->global_active_cell_index())
                    {
                      // const double extent1 = cell->measure() /
                      // cell->face(f)->measure(); const double extent2 =
                      // neigh_cell->measure() /
                      // neigh_cell->face(cell->neighbor_of_neighbor(f))
                      // ->measure();
                      fe_faces0->reinit(cell, f);
                      fe_faces1->reinit(neigh_cell,
                                        cell->neighbor_of_neighbor(f));

                      // FEFaceValues::inverse_jacobian returns J^{-1}. To
                      // emulate matrix-free (which sees J^{-T} with reference
                      // normal aligned to the last coordinate) pick the row
                      // of J^{-1} corresponding to this face’s reference
                      // normal and dot it with the physical normal.
                      const unsigned int ref_normal_0 =
                        GeometryInfo<dim>::unit_normal_direction[f];
                      const unsigned int ref_normal_1 = GeometryInfo<dim>::
                        unit_normal_direction[cell->neighbor_of_neighbor(f)];

                      double h_inverse_0 = 0.;
                      double h_inverse_1 = 0.;
                      for (unsigned int j = 0; j < dim; ++j)
                        {
                          h_inverse_0 +=
                            fe_faces0->inverse_jacobian(0)[ref_normal_0][j] *
                            fe_faces0->normal_vector(0)[j];
                          h_inverse_1 +=
                            fe_faces1->inverse_jacobian(0)[ref_normal_1][j] *
                            fe_faces1->normal_vector(0)[j];
                        }

                      // Notice: in case of cartesian meshes, this is
                      // equivalent to const double penalty =
                      // penalty_constant*(1. / extent1
                      // + 1. / extent2);
                      const double penalty =
                        penalty_constant *
                        (std::abs(h_inverse_0) + std::abs(h_inverse_1));

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
                                     (penalty)*fe_faces0->shape_value(i,
                                                                      q_index) *
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
                                     (penalty)*fe_faces0->shape_value(i,
                                                                      q_index) *
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
                                     (penalty)*fe_faces1->shape_value(i,
                                                                      q_index) *
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
                                     (penalty)*fe_faces1->shape_value(i,
                                                                      q_index) *
                                       fe_faces1->shape_value(j, q_index)) *
                                    fe_faces1->JxW(q_index);
                                }
                            }
                        }

                      // distribute DoFs accordingly

                      neigh_cell->get_dof_indices(local_dof_indices_neighbor);

                      constraints.distribute_local_to_global(
                        M11,
                        local_dof_indices,
                        param.preconditioner == Preconditioner::AMG ?
                          system_matrix :
                          multigrid_matrices[multigrid_matrices.max_level()]);
                      constraints.distribute_local_to_global(
                        M12,
                        local_dof_indices,
                        local_dof_indices_neighbor,
                        param.preconditioner == Preconditioner::AMG ?
                          system_matrix :
                          multigrid_matrices[multigrid_matrices.max_level()]);
                      constraints.distribute_local_to_global(
                        M21,
                        local_dof_indices_neighbor,
                        local_dof_indices,
                        param.preconditioner == Preconditioner::AMG ?
                          system_matrix :
                          multigrid_matrices[multigrid_matrices.max_level()]);
                      constraints.distribute_local_to_global(
                        M22,
                        local_dof_indices_neighbor,
                        param.preconditioner == Preconditioner::AMG ?
                          system_matrix :
                          multigrid_matrices[multigrid_matrices.max_level()]);

                    } // check idx neighbors
                }     // over faces
            }
          constraints.distribute_local_to_global(
            cell_matrix,
            local_dof_indices,
            param.preconditioner == Preconditioner::AMG ?
              system_matrix :
              multigrid_matrices[multigrid_matrices.max_level()]);
          // constraints.distribute_local_to_global(cell_mass_matrix,
          //                                        local_dof_indices,
          //                                        mass_matrix);
        }
    }
  // mass_matrix.compress(VectorOperation::add);

  if (param.preconditioner == Preconditioner::AMG)
    system_matrix.compress(VectorOperation::add);
  else
    multigrid_matrices[multigrid_matrices.max_level()].compress(
      VectorOperation::add);
}



/*
 * Assemble the time independent block chi*c*M/dt + A
 */
template <int dim>
void
MonodomainProblem<dim>::assemble_time_independent_matrix_bdf1()
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
                  const double Aii =
                    (param.sigma * fe_values->shape_grad(i, q_index) *
                       fe_values->shape_grad(i, q_index) +
                     (param.chi * param.Cm / dt) *
                       fe_values->shape_value(i, q_index) *
                       fe_values->shape_value(i, q_index)) *
                    fe_values->JxW(q_index);

                  const double Mii = (param.chi * param.Cm / dt) *
                                     fe_values->shape_value(i, q_index) *
                                     fe_values->shape_value(i, q_index) *
                                     fe_values->JxW(q_index);

                  cell_matrix(i, i) += Aii;
                  cell_mass_matrix(i, i) += Mii;

                  for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
                    {
                      const double Aij =
                        (param.sigma * fe_values->shape_grad(i, q_index) *
                           fe_values->shape_grad(j, q_index) +
                         (param.chi * param.Cm / dt) *
                           fe_values->shape_value(i, q_index) *
                           fe_values->shape_value(j, q_index)) *
                        fe_values->JxW(q_index);

                      const double Mij = (param.chi * param.Cm / dt) *
                                         fe_values->shape_value(i, q_index) *
                                         fe_values->shape_value(j, q_index) *
                                         fe_values->JxW(q_index);

                      cell_matrix(i, j) += Aij;
                      cell_matrix(j, i) += Aij;
                      cell_mass_matrix(i, j) += Mij;
                      cell_mass_matrix(j, i) += Mij;
                    }
                }
            }

          for (const auto f : cell->face_indices())
            {
              // Do nothing at boundary, we have Neumann homogeneous BC
              if (!cell->face(f)->at_boundary())
                {
                  const auto &neigh_cell = cell->neighbor(f);

                  if (cell->global_active_cell_index() <
                      neigh_cell->global_active_cell_index())
                    {
                      // const double extent1 = cell->measure() /
                      // cell->face(f)->measure(); const double extent2 =
                      // neigh_cell->measure() /
                      // neigh_cell->face(cell->neighbor_of_neighbor(f))
                      // ->measure();
                      fe_faces0->reinit(cell, f);
                      fe_faces1->reinit(neigh_cell,
                                        cell->neighbor_of_neighbor(f));

                      // FEFaceValues::inverse_jacobian returns J^{-1}. To
                      // emulate matrix-free (which sees J^{-T} with reference
                      // normal aligned to the last coordinate) pick the row
                      // of J^{-1} corresponding to this face’s reference
                      // normal and dot it with the physical normal.
                      const unsigned int ref_normal_0 =
                        GeometryInfo<dim>::unit_normal_direction[f];
                      const unsigned int ref_normal_1 = GeometryInfo<dim>::
                        unit_normal_direction[cell->neighbor_of_neighbor(f)];

                      double h_inverse_0 = 0.;
                      double h_inverse_1 = 0.;
                      for (unsigned int j = 0; j < dim; ++j)
                        {
                          h_inverse_0 +=
                            fe_faces0->inverse_jacobian(0)[ref_normal_0][j] *
                            fe_faces0->normal_vector(0)[j];
                          h_inverse_1 +=
                            fe_faces1->inverse_jacobian(0)[ref_normal_1][j] *
                            fe_faces1->normal_vector(0)[j];
                        }

                      // Notice: in case of cartesian meshes, this is
                      // equivalent to const double penalty =
                      // penalty_constant*(1. / extent1
                      // + 1. / extent2);
                      const double penalty =
                        penalty_constant *
                        (std::abs(h_inverse_0) + std::abs(h_inverse_1));

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
                                     (penalty)*fe_faces0->shape_value(i,
                                                                      q_index) *
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
                                     (penalty)*fe_faces0->shape_value(i,
                                                                      q_index) *
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
                                     (penalty)*fe_faces1->shape_value(i,
                                                                      q_index) *
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
                                     (penalty)*fe_faces1->shape_value(i,
                                                                      q_index) *
                                       fe_faces1->shape_value(j, q_index)) *
                                    fe_faces1->JxW(q_index);
                                }
                            }
                        }

                      // distribute DoFs accordingly

                      neigh_cell->get_dof_indices(local_dof_indices_neighbor);

                      constraints.distribute_local_to_global(M11,
                                                             local_dof_indices,
                                                             bdf1_matrix);
                      constraints.distribute_local_to_global(
                        M12,
                        local_dof_indices,
                        local_dof_indices_neighbor,
                        bdf1_matrix);
                      constraints.distribute_local_to_global(
                        M21,
                        local_dof_indices_neighbor,
                        local_dof_indices,
                        bdf1_matrix);
                      constraints.distribute_local_to_global(
                        M22, local_dof_indices_neighbor, bdf1_matrix);

                    } // check idx neighbors
                }     // over faces
            }
          constraints.distribute_local_to_global(cell_matrix,
                                                 local_dof_indices,
                                                 bdf1_matrix);
        }
    }

  bdf1_matrix.compress(VectorOperation::add);
}



template <int dim>
void
MonodomainProblem<dim>::assemble_time_terms()
{
  const unsigned int dofs_per_cell = dg_fe.n_dofs_per_cell();
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  const double                         factor = param.chi * param.Cm / dt;

  // Loop over standard deal.II cells
  for (const auto &cell : classical_dh.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          cell_rhs = 0.;

          fe_values->reinit(cell);

          const auto        &q_points  = fe_values->get_quadrature_points();
          const unsigned int n_qpoints = q_points.size();

          std::vector<double> applied_currents(n_qpoints);
          Iext->value_list(q_points, applied_currents);

          std::vector<double> ion_at_qpoints(n_qpoints);
          fe_values->get_function_values(ion_at_dofs, ion_at_qpoints);

          std::vector<double> solution_current(n_qpoints);
          fe_values->get_function_values(locally_relevant_solution_n,
                                         solution_current);

          // Get local values of current solution, to be used inside the non
          // linear reaction matrix

          cell->get_dof_indices(local_dof_indices);

          for (unsigned int q_index : fe_values->quadrature_point_indices())
            {
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  cell_rhs(i) += (applied_currents[q_index] -
                                  param.Cm * ion_at_qpoints[q_index] +
                                  factor * solution_current[q_index]) *
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
MonodomainProblem<dim>::solve()
{
  TimerOutput::Scope t(computing_timer, "Solve");

  if constexpr (measure_solve_times)
    start_solver = MPI_Wtime();
  if (param.preconditioner == Preconditioner::AMG)
    solver.solve(system_operator,
                 locally_relevant_solution_np1,
                 system_rhs,
                 amg_preconditioner);
  else
    solver.solve(system_operator,
                 locally_relevant_solution_np1,
                 system_rhs,
                 *preconditioner);
  if constexpr (measure_solve_times)
    {
      stop_solver = MPI_Wtime();
      iteration_times.push_back(stop_solver - start_solver);
    }

  pcout << "Number of outer iterations: " << param.control.last_step()
        << std::endl;
  iterations.push_back(param.control.last_step());
}



template <int dim>
void
MonodomainProblem<dim>::output_results()
{
  TimerOutput::Scope t(computing_timer, "Output results");

  DataOut<dim> data_out;
  data_out.attach_dof_handler(classical_dh);
  data_out.add_data_vector(locally_relevant_solution_np1,
                           "transmembrane_potential",
                           DataOut<dim>::type_dof_data);

  data_out.add_data_vector(locally_relevant_w0_np1,
                           "gating_variable",
                           DataOut<dim>::type_dof_data);

  Vector<float> subdomain(tria.n_active_cells());

  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = tria.locally_owned_subdomain();

  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches(mapping);

  const std::string filename =
    (param.output_directory + "/" + "3Dmonodomain_time_" +
     std::to_string(time) +
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
          filenames.push_back(param.output_directory + "/" +
                              "3Dmonodomain_time_" + std::to_string(time) +
                              Utilities::int_to_string(i, 4) + ".vtu");
        }
      std::ofstream master_output(param.output_directory + "/" +
                                  "3Dmonodomain_time_" + std::to_string(time) +
                                  ".pvtu");
      data_out.write_pvtu_record(master_output, filenames);
    }
}



template <int dim>
void
MonodomainProblem<dim>::run()
{
  pcout << "Running on " << Utilities::MPI::n_mpi_processes(communicator)
        << " MPI rank(s)." << std::endl;

  // Create mesh
  {
    TimerOutput::Scope t(computing_timer, "Import and partition mesh");
    Triangulation<dim> tria_dummy;
    GridIn<dim>        grid_in;
    grid_in.attach_triangulation(tria_dummy);
    std::string mesh_path;
    if (param.test_case == TestCase::Idealized)
      mesh_path = "../../meshes/idealized_lv.msh";
    else if (param.test_case == TestCase::Realistic)
      mesh_path = "../../meshes/realistic_lv.msh";
    std::ifstream mesh_file(mesh_path);
    grid_in.read_msh(mesh_file);

    // scale triangulation by a suitable factor in order to work with mm
    const double scale_factor = 1e-3;
    GridTools::scale(scale_factor, tria_dummy);

    // Partition serial triangulation:
    const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(communicator);
    GridTools::partition_triangulation(n_ranks, tria_dummy);

    // Create building blocks:
    const TriangulationDescription::Description<dim, dim> description =
      TriangulationDescription::Utilities::
        create_description_from_triangulation(tria_dummy, communicator);

    tria.create_triangulation(description);
    pcout << "   Number of active cells:       " << tria.n_global_active_cells()
          << std::endl;
  }

  setup_problem();
  pcout << "   Number of degrees of freedom: " << classical_dh.n_dofs()
        << std::endl;

  // Set initial conditions
  locally_relevant_solution_n   = -84e-3;
  locally_relevant_solution_nm1 = locally_relevant_solution_n;
  locally_relevant_solution_np1 = locally_relevant_solution_n;
  extrapoled_solution           = locally_relevant_solution_n;

  locally_relevant_w0_n   = 1.;
  locally_relevant_w0_nm1 = locally_relevant_w0_n;
  locally_relevant_w0_np1 = locally_relevant_w0_n;

  locally_relevant_w1_n   = 1.;
  locally_relevant_w1_nm1 = locally_relevant_w1_n;
  locally_relevant_w1_np1 = locally_relevant_w1_n;

  locally_relevant_w2_n   = 0.;
  locally_relevant_w2_nm1 = locally_relevant_w2_n;
  locally_relevant_w2_np1 = locally_relevant_w2_n;
  output_results();

  // Assemble time independent term
  assemble_time_independent_matrix();
  pcout << "Assembled time independent term: done" << std::endl;

  unsigned int iter_count = 0;

  const double mesh_size = [this]() -> double {
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

  pcout << "Max mesh size: " << Utilities::MPI::max(mesh_size, communicator)
        << std::endl;

  // Initialize matrix-free evaluator
  monodomain_operator->reinit(mapping, classical_dh);

  if constexpr (use_matrix_free_action)
    system_operator = linear_operator<VectorType>(*monodomain_operator);
  else
    system_operator = linear_operator<VectorType>(
      param.preconditioner == Preconditioner::AMG ?
        system_matrix :
        multigrid_matrices[multigrid_matrices.max_level()]);

  // Depending on the preconditioner type, use AMG or polytopal multigrid.
  if (param.preconditioner == Preconditioner::AMG)
    {
      TimerOutput::Scope t(computing_timer, "Initialize AMG preconditioner");
      typename TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
      if (dg_fe.degree > 1)
        amg_data.higher_order_elements = true;

      amg_data.smoother_type         = "Chebyshev";
      amg_data.smoother_sweeps       = 3;
      amg_data.output_details        = true;
      amg_data.aggregation_threshold = 0.2;
      amg_preconditioner.initialize(system_matrix, amg_data);
    }
  else
    setup_multigrid();

  pcout << "Setup preconditioner: done " << std::endl;

  double              min_value = std::numeric_limits<double>::min();
  std::vector<double> min_values;
  min_values.push_back(-84e-3);

  VectorType solution_minus_ion(classical_dh.locally_owned_dofs(),
                                communicator);

  while (time <= end_time)
    {
      // Solve the ODEs for w
      update_w_and_ion();

      // Assemble time dependent terms on the rhs
      if constexpr (use_matrix_free_action)
        {
          TimerOutput::Scope t(computing_timer,
                               "Assemble time dependent terms (matrix-free)");
          if (param.time_stepping_scheme == Utils::Physics::TimeStepping::BDF1)
            solution_minus_ion = locally_relevant_solution_n;
          else if (param.time_stepping_scheme ==
                   Utils::Physics::TimeStepping::BDF2)
            {
              if (time == 0)
                {
                  solution_minus_ion = locally_relevant_solution_n;
                  solution_minus_ion *= (param.Cm / dt);
                  solution_minus_ion -= ion_at_dofs;
                  monodomain_operator->rhs(system_rhs,
                                           solution_minus_ion,
                                           *Iext);

                  assemble_time_independent_matrix_bdf1();
                }
              else
                {
                  solution_minus_ion = locally_relevant_solution_n;
                  solution_minus_ion *= 4.;
                  solution_minus_ion -= locally_relevant_solution_nm1;


                  solution_minus_ion *= (param.Cm / (2. * dt));
                  solution_minus_ion -= ion_at_dofs;
                  monodomain_operator->rhs(system_rhs,
                                           solution_minus_ion,
                                           *Iext);
                }
            }
          else
            DEAL_II_NOT_IMPLEMENTED();
        }
      else
        {
          TimerOutput::Scope t(computing_timer,
                               "Assemble time dependent terms (matrix-based)");
          assemble_time_terms();
        }

      // Solver for transmembrane potential
      if (time == 0)
        {
          pcout << "Solving at t= " << time << " (initial step)" << std::endl;
          TrilinosWrappers::PreconditionAMG amg_preconditioner_bdf1;
          typename TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
          if (dg_fe.degree > 1)
            amg_data.higher_order_elements = true;

          amg_data.smoother_type         = "Chebyshev";
          amg_data.smoother_sweeps       = 3;
          amg_data.output_details        = true;
          amg_data.aggregation_threshold = 0.2;
          amg_preconditioner_bdf1.initialize(bdf1_matrix, amg_data);

          SolverCG<VectorType> solver_bdf1(
            const_cast<SolverControl &>(param.control));
          solver_bdf1.solve(bdf1_matrix,
                            locally_relevant_solution_np1,
                            system_rhs,
                            amg_preconditioner_bdf1);
        }
      else
        {
          solve();
        }

      pcout << "Solved at t= " << time << std::endl;
      ++iter_count;
      time += dt;
      Iext->set_time(time);

      // output results
      if (iter_count % param.output_frequency == 0)
        output_results();

      // Store minimum value current action potential on each processor
      if (param.compute_min_value)
        {
          min_value = *locally_relevant_solution_np1.begin();
          for (const double v : locally_relevant_solution_np1)
            if (v < min_value)
              min_value = v;

          min_values.push_back(Utilities::MPI::min(min_value, communicator));
          pcout << min_values.back() << std::endl;
        }

      // update solutions
      locally_relevant_solution_nm1 = locally_relevant_solution_n;
      locally_relevant_solution_n   = locally_relevant_solution_np1;

      // update extrapolated solution for BDF2
      extrapoled_solution = locally_relevant_solution_n;
      extrapoled_solution *= 2.;
      extrapoled_solution -= locally_relevant_solution_nm1;

      locally_relevant_w0_nm1 = locally_relevant_w0_n;
      locally_relevant_w0_n   = locally_relevant_w0_np1;

      locally_relevant_w1_nm1 = locally_relevant_w1_n;
      locally_relevant_w1_n   = locally_relevant_w1_np1;

      locally_relevant_w2_nm1 = locally_relevant_w2_n;
      locally_relevant_w2_n   = locally_relevant_w2_np1;

      // reset time dependent terms
      system_rhs = 0.;
    }
  pcout << std::endl;



  if (Utilities::MPI::this_mpi_process(communicator) == 0)
    {
      std::ofstream file_iterations;
      file_iterations.open(param.output_directory + "/" + "iterations_" +
                           (param.preconditioner == Preconditioner::AMG ?
                              std::string("AMG_") :
                              std::string("AGGLOMG_")) +
                           "degree_" + std::to_string(param.fe_degree) +
                           ".txt");

      for (size_t i = 0; i < iterations.size(); ++i)
        {
          file_iterations << iterations[i];
          if (i < iterations.size() - 1)
            file_iterations << ",\n";
        }
      file_iterations << "\n";
      file_iterations.close();

      std::ofstream file_iteration_times;
      file_iteration_times.open(
        param.output_directory + "/" + "iteration_times_" +
        (param.preconditioner == Preconditioner::AMG ?
           std::string("AMG_") :
           std::string("AGGLOMG_")) +
        "degree_" + std::to_string(param.fe_degree) + ".txt");

      for (size_t i = 0; i < iteration_times.size(); ++i)
        {
          file_iteration_times << iteration_times[i];
          if (i < iteration_times.size() - 1)
            file_iteration_times << ",\n";
        }
      file_iteration_times << "\n";
      file_iteration_times.close();

      std::cout
        << "---------------------------------------------------------------------SOLVER STATISTICS----"
           "--------------------------------------------------------------------------------------------------------------"
        << std::endl;
      std::cout
        << "+-------------------------------------------------------------------------------------"
           "-----------------------------------------------------------------------------------------------------------------+"
        << std::endl;

      const auto [min_iter, max_iter] =
        std::minmax_element(iterations.begin(), iterations.end());
      const double avg_iter =
        std::accumulate(iterations.begin(), iterations.end(), 0.0) /
        iterations.size();
      const double std_deviation_iter =
        [](const std::vector<unsigned int> &iters,
           const double                     average) -> double {
        double sum = 0.0;
        for (const unsigned int it : iters)
          sum += (it - average) * (it - average);
        return std::sqrt(sum / iters.size());
      }(iterations, avg_iter);

      statistics_table.add_value("Min. iterations", *min_iter);
      statistics_table.add_value("Max. iterations", *max_iter);
      statistics_table.add_value("Avg. iterations", avg_iter);
      statistics_table.add_value("Std. deviation iterations",
                                 std_deviation_iter);
      if constexpr (measure_solve_times)
        statistics_table.add_value("Avg. time per time-step [s]",
                                   std::accumulate(iteration_times.begin(),
                                                   iteration_times.end(),
                                                   0.0) /
                                     iterations.size());

      statistics_table.write_text(std::cout, TableHandler::org_mode_table);
      std::ofstream output_file(param.output_directory + "/" + "statistics_" +
                                (param.preconditioner == Preconditioner::AMG ?
                                   std::string("AMG_") :
                                   std::string("AGGLOMG_")) +
                                "degree_" + std::to_string(param.fe_degree) +
                                ".txt");
      statistics_table.write_text(output_file, TableHandler::org_mode_table);

      std::cout
        << "+-------------------------------------------------------------------------------------"
           "-----------------------------------------------------------------------------------------------------------------+"
        << std::endl;
    }
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  deallog.depth_console(
    Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 ? 10 : 0);

  if constexpr (use_matrix_free_action)
    {
      const unsigned int n_vect_doubles = VectorizedArray<double>::size();
      const unsigned int n_vect_bits    = 8 * sizeof(double) * n_vect_doubles;

      deallog << "Vectorization over " << n_vect_doubles
              << " doubles = " << n_vect_bits << " bits ("
              << Utilities::System::get_current_vectorization_level() << ')'
              << std::endl;
    }

  {
    ModelParameters parameters;
    parameters.control.set_tolerance(1e-13); // used in CG solver
    parameters.control.set_max_steps(2000);

    parameters.preconditioner            = Preconditioner::AMG;
    parameters.test_case                 = TestCase::Idealized;
    parameters.time_stepping_scheme      = Utils::Physics::TimeStepping::BDF2;
    parameters.fe_degree                 = degree_finite_element;
    parameters.dt                        = 1e-4;
    parameters.final_time                = 0.4;
    parameters.final_time_current        = 3e-3;
    parameters.compute_min_value         = true;
    parameters.estimate_condition_number = false;
    parameters.output_frequency          = 10;
    parameters.output_directory          = "./";

    MonodomainProblem<3> problem(parameters);
    problem.run();
  }
  return 0;
}
