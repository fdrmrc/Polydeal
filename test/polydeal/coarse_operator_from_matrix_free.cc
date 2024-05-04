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


// Check the Galerkin projection of a matrix-free operator on coarser,
// agglomerate, levels.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_integrate_difference.templates.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <agglomeration_handler.h>
#include <agglomerator.h>
#include <multigrid_amg.h>
#include <poly_utils.h>

using namespace dealii;

static constexpr unsigned int degree_finite_element = 1;

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


  DynamicSparsityPattern dsp(output_dh->n_dofs(), agglo_dh.n_dofs());

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



// Define coefficient needed by Matrix-free operator evaluation. Here, we set it
// to 1.
template <int dim>
class Coefficient : public Function<dim>
{
public:
  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

  template <typename number>
  number
  value(const Point<dim, number> &p, const unsigned int component = 0) const;
};



template <int dim>
template <typename number>
number
Coefficient<dim>::value(const Point<dim, number> &p,
                        const unsigned int /*component*/) const
{
  (void)p;
  return 1.;
}



template <int dim>
double
Coefficient<dim>::value(const Point<dim> &p, const unsigned int component) const
{
  return value<double>(p, component);
}

template <int dim, int fe_degree, typename number>
class LaplaceOperator
  : public MatrixFreeOperators::Base<dim,
                                     LinearAlgebra::distributed::Vector<number>>
{
public:
  using value_type = number;

  LaplaceOperator();

  void
  clear() override;

  void
  evaluate_coefficient(const Coefficient<dim> &coefficient_function);

  virtual void
  compute_diagonal() override;

private:
  virtual void
  apply_add(
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src) const override;

  void
  local_apply(const MatrixFree<dim, number>                    &data,
              LinearAlgebra::distributed::Vector<number>       &dst,
              const LinearAlgebra::distributed::Vector<number> &src,
              const std::pair<unsigned int, unsigned int> &cell_range) const;

  void
  local_compute_diagonal(
    const MatrixFree<dim, number>               &data,
    LinearAlgebra::distributed::Vector<number>  &dst,
    const unsigned int                          &dummy,
    const std::pair<unsigned int, unsigned int> &cell_range) const;

  Table<2, VectorizedArray<number>> coefficient;
};



template <int dim, int fe_degree, typename number>
LaplaceOperator<dim, fe_degree, number>::LaplaceOperator()
  : MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>()
{}



template <int dim, int fe_degree, typename number>
void
LaplaceOperator<dim, fe_degree, number>::clear()
{
  coefficient.reinit(0, 0);
  MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>::
    clear();
}



template <int dim, int fe_degree, typename number>
void
LaplaceOperator<dim, fe_degree, number>::evaluate_coefficient(
  const Coefficient<dim> &coefficient_function)
{
  const unsigned int n_cells = this->data->n_cell_batches();
  FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(*this->data);

  coefficient.reinit(n_cells, phi.n_q_points);
  for (unsigned int cell = 0; cell < n_cells; ++cell)
    {
      phi.reinit(cell);
      for (const unsigned int q : phi.quadrature_point_indices())
        coefficient(cell, q) =
          coefficient_function.value(phi.quadrature_point(q));
    }
}



template <int dim, int fe_degree, typename number>
void
LaplaceOperator<dim, fe_degree, number>::local_apply(
  const MatrixFree<dim, number>                    &data,
  LinearAlgebra::distributed::Vector<number>       &dst,
  const LinearAlgebra::distributed::Vector<number> &src,
  const std::pair<unsigned int, unsigned int>      &cell_range) const
{
  FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(data);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      AssertDimension(coefficient.size(0), data.n_cell_batches());
      AssertDimension(coefficient.size(1), phi.n_q_points);

      phi.reinit(cell);
      phi.read_dof_values(src);
      phi.evaluate(EvaluationFlags::gradients);
      for (const unsigned int q : phi.quadrature_point_indices())
        phi.submit_gradient(coefficient(cell, q) * phi.get_gradient(q), q);
      phi.integrate(EvaluationFlags::gradients);
      phi.distribute_local_to_global(dst);
    }
}



template <int dim, int fe_degree, typename number>
void
LaplaceOperator<dim, fe_degree, number>::apply_add(
  LinearAlgebra::distributed::Vector<number>       &dst,
  const LinearAlgebra::distributed::Vector<number> &src) const
{
  this->data->cell_loop(&LaplaceOperator::local_apply, this, dst, src);
}



template <int dim, int fe_degree, typename number>
void
LaplaceOperator<dim, fe_degree, number>::compute_diagonal()
{
  Assert(false, ExcInternalError());
}


// Then, define some functions for the test itself
template <int dim>
class XFunction : public Function<dim>
{
public:
  XFunction()
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
XFunction<dim>::value(const Point<dim> &p, const unsigned int) const
{
  return p[0];
}



template <int dim>
void
XFunction<dim>::value_list(const std::vector<Point<dim>> &points,
                           std::vector<double>           &values,
                           const unsigned int /*component*/) const
{
  for (unsigned int i = 0; i < values.size(); ++i)
    values[i] = this->value(points[i]);
}



template <int dim>
class XplusYFunction : public Function<dim>
{
public:
  XplusYFunction()
    : Function<dim>()
  {
    Assert(dim > 1, ExcNotImplemented());
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
XplusYFunction<dim>::value(const Point<dim> &p, const unsigned int) const
{
  return p[0] + p[1];
}



template <int dim>
void
XplusYFunction<dim>::value_list(const std::vector<Point<dim>> &points,
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
class TestMGMatrix
{
public:
  TestMGMatrix(const GridType      &grid_type,
               const unsigned int   degree,
               const Function<dim> &func,
               const MPI_Comm       comm);
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
  const GridType                                &grid_type;
  parallel::fullydistributed::Triangulation<dim> tria_pft;
  DoFHandler<dim>                                dof_handler;
  const Function<dim>                           &func;
  ConditionalOStream                             pcout;

  std::unique_ptr<AgglomerationHandler<dim>> agglomeration_handler;
};



template <int dim>
TestMGMatrix<dim>::TestMGMatrix(const GridType      &grid_type_,
                                const unsigned int   degree,
                                const Function<dim> &func_,
                                const MPI_Comm       communicator)
  : comm(communicator)
  , n_ranks(Utilities::MPI::n_mpi_processes(comm))
  , fe_dg(degree)
  , grid_type(grid_type_)
  , tria_pft(comm)
  , dof_handler(tria_pft)
  , func(func_)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(comm) == 0))
{
  pcout << "Running with " << n_ranks << " MPI ranks." << std::endl;
  pcout << "Grid type:";
  grid_type == GridType::grid_generator ?
    pcout << " Structured square" << std::endl :
    pcout << " Unstructured square" << std::endl;
}



template <int dim>
void
TestMGMatrix<dim>::make_fine_grid(const unsigned int n_global_refinements)
{
  Triangulation<dim> tria;

  if (grid_type == GridType::unstructured)
    {
      GridIn<dim> grid_in;
      grid_in.attach_triangulation(tria);
      //   std::ifstream abaqus_file("../../meshes/piston_3.inp"); // piston
      //   mesh
      std::ifstream gmsh_file(
        SOURCE_DIR "/input_grids/square.msh"); // unstructured square [0,1]^2
      grid_in.read_msh(gmsh_file);
      tria.refine_global(n_global_refinements - 2); // 4
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
TestMGMatrix<dim>::agglomerate_and_compute_level_matrices()
{
  GridTools::Cache<dim> cached_tria(tria_pft);



  // Partition with Rtree locally to each partition.
  MappingQ1<dim> mapping; // use standard mapping

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

  const unsigned int                     extraction_level = 2;
  CellsAgglomerator<dim, decltype(tree)> agglomerator{tree,
                                                      extraction_level + 1};
  const auto agglomerates = agglomerator.extract_agglomerates();
  agglomeration_handler =
    std::make_unique<AgglomerationHandler<dim>>(cached_tria);
  std::size_t agglo_index_fine = 0;
  for (std::size_t i = 0; i < agglomerates.size(); ++i)
    {
      const auto &agglo = agglomerates[i]; // i-th agglomerate fine
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
    agglomeration_handler->define_agglomerate(cells_per_subdomain_fine[i]);

  agglomeration_handler->initialize_fe_values(
    QGauss<dim>(degree_finite_element + 1),
    update_values | update_gradients | update_JxW_values |
      update_quadrature_points,
    QGauss<dim - 1>(degree_finite_element + 1),
    update_JxW_values);
  agglomeration_handler->distribute_agglomerated_dofs(fe_dg);

  pcout << "Distributed DoFs" << std::endl;


  // Check injection matrix
  TrilinosWrappers::SparseMatrix embedding_matrix;
  fill_interpolation_matrix(*agglomeration_handler, embedding_matrix);
  pcout << "Injection matrix has size: (" << embedding_matrix.m() << ","
        << embedding_matrix.n() << ")" << std::endl;

  // Setup matrix-free object and operator.
  AffineConstraints constraints;
  constraints.close();

  dof_handler.distribute_dofs(fe_dg);

  typename MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.tasks_parallel_scheme =
    MatrixFree<dim, double>::AdditionalData::none;
  additional_data.mapping_update_flags =
    (update_gradients | update_JxW_values | update_quadrature_points);
  std::shared_ptr<MatrixFree<dim, double>> system_mf_storage(
    new MatrixFree<dim, double>());
  system_mf_storage->reinit(mapping,
                            dof_handler,
                            constraints,
                            QGauss<1>(fe_dg.degree + 1),
                            additional_data);

  LaplaceOperator<dim, degree_finite_element, double> system_matrix;
  system_matrix.initialize(system_mf_storage);
  system_matrix.evaluate_coefficient(Coefficient<dim>());

  // Define transfer between levels.
  std::vector<TrilinosWrappers::SparseMatrix *> transfers(1);
  transfers[0] = &embedding_matrix;

  MatrixFreeProjector<dim, double> mf_amg(system_matrix, transfers);
  using VectorType = LinearAlgebra::distributed::Vector<double>;
  MGLevelObject<LinearOperator<VectorType, VectorType>> mg_matrices(0, 2);
  mf_amg.compute_level_matrices(mg_matrices);


  //  Define dst, src for the fine space.
  LinearAlgebra::distributed::Vector<double> fine_src;
  fine_src.reinit(dof_handler.locally_owned_dofs(), comm);
  LinearAlgebra::distributed::Vector<double> fine_dst;
  fine_dst.reinit(dof_handler.locally_owned_dofs(), comm);

  VectorTools::interpolate(mapping, dof_handler, func, fine_src);
  system_matrix.vmult(fine_dst, fine_src);
  const double test_operator_scalar_product = fine_src * fine_dst;
  pcout << "Scalar product induced by fine operator: "
        << test_operator_scalar_product << std::endl;


  // Define dst, src for the coarse agglomerated space.
  LinearAlgebra::distributed::Vector<double> src_coarse;
  src_coarse.reinit(agglomeration_handler->agglo_dh.locally_owned_dofs(), comm);
  LinearAlgebra::distributed::Vector<double> dst_coarse_op;
  dst_coarse_op.reinit(agglomeration_handler->agglo_dh.locally_owned_dofs(),
                       comm);
  VectorTools::interpolate(*(agglomeration_handler->euler_mapping),
                           agglomeration_handler->agglo_dh,
                           func,
                           src_coarse);
  dst_coarse_op = mg_matrices[1] * src_coarse;
  const double test_coarse_operator_scalar_product = src_coarse * dst_coarse_op;
  pcout << "Scalar product induced by agglomerated operator: "
        << test_coarse_operator_scalar_product << std::endl;

#ifdef FALSE
  pcout << "Size of resulting vector: " << dst_coarse_op.size() << std::endl;
  for (const double x : dst_coarse_op)
    pcout << x << std::endl;
  pcout << "L2 norm: " << dst_coarse_op.l2_norm() << std::endl;
  pcout << "L2 norm(fine): " << fine_dst.l2_norm() << std::endl;
#endif
}



template <int dim>
void
TestMGMatrix<dim>::run()
{
  make_fine_grid(6);
  agglomerate_and_compute_level_matrices();
  pcout << std::endl;
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  const MPI_Comm                   comm = MPI_COMM_WORLD;
  static constexpr unsigned int    dim  = 2;


  // number of agglomerates in each local subdomain
  XFunction<dim>                   x;
  XplusYFunction<dim>              xpy;
  Functions::ConstantFunction<dim> constant(1.);
  std::array<Function<dim> *, 3>   functions{{&constant, &x, &xpy}};

  for (const Function<dim> *func : functions)
    for (const GridType &grid_type :
         {GridType::grid_generator, GridType::unstructured})
      {
        TestMGMatrix<dim> problem(grid_type,
                                  degree_finite_element,
                                  *func,
                                  comm);
        problem.run();
      }
}
