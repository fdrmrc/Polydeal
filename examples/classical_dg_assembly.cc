#include <deal.II/base/config.h>

#include <deal.II/base/function.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/numerics/data_out.h>


// Trilinos linear algebra is employed for parallel computations
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <algorithm>


/**
 * The following example program solves the simplest elliptic problem with DG on
 * standard quad-hex meshes in parallel **without** employing the MeshWorker
 * framework of deal.II
 *
 */


using namespace dealii;


template <int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide()
    : Function<dim>()
  {}

  virtual void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<double> &          values,
             const unsigned int /*component*/) const override;
};


template <int dim>
void
RightHandSide<dim>::value_list(const std::vector<Point<dim>> &points,
                               std::vector<double> &          values,
                               const unsigned int /*component*/) const
{
  for (unsigned int i = 0; i < values.size(); ++i)
    values[i] = 8. * numbers::PI * numbers::PI *
                std::sin(2. * numbers::PI * points[i][0]) *
                std::sin(2. * numbers::PI * points[i][1]);
}


template <int dim>
class Poisson
{
private:
  void
  make_grid();

  void
  distribute_jumps_and_averages(
    FEFaceValues<dim> &                                   fe_face0,
    FEFaceValues<dim> &                                   fe_face1,
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    const unsigned int                                    f);
  void
  assemble_system();
  void
  solve();
  void
  output_results();


  parallel::distributed::Triangulation<dim> tria;
  MappingQ<dim>                             mapping;
  FE_DGQ<dim>                               dg_fe;
  DoFHandler<dim>                           classical_dh;
  ConditionalOStream                        pcout;
  SparsityPattern                           sparsity;
  AffineConstraints<double>                 constraints;
  TrilinosWrappers::SparseMatrix            system_matrix;
  TrilinosWrappers::MPI::Vector             locally_relevant_solution;
  TrilinosWrappers::MPI::Vector             system_rhs;
  SolverControl                             solver_control;
  TrilinosWrappers::SolverDirect            solver;
  std::unique_ptr<const Function<dim>>      rhs_function;

public:
  Poisson(const unsigned int fe_degree);
  void
  run();

  double penalty = 50.;
};



template <int dim>
Poisson<dim>::Poisson(const unsigned int fe_degree)
  : tria(MPI_COMM_WORLD)
  , mapping(1)
  , dg_fe(fe_degree)
  , classical_dh(tria)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  , solver_control(1)
  , solver(solver_control)
{}

template <int dim>
void
Poisson<dim>::make_grid()
{
  GridGenerator::hyper_cube(tria, -1, 1);
  tria.refine_global(6);

  classical_dh.distribute_dofs(dg_fe);
  const IndexSet &locally_owned_dofs = classical_dh.locally_owned_dofs();
  const IndexSet  locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(classical_dh);

  constraints.clear();
  constraints.close();

  DynamicSparsityPattern dsp(classical_dh.n_dofs());
  DoFTools::make_flux_sparsity_pattern(classical_dh, dsp);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             classical_dh.locally_owned_dofs(),
                                             MPI_COMM_WORLD,
                                             locally_relevant_dofs);

  system_matrix.reinit(locally_owned_dofs,
                       locally_owned_dofs,
                       dsp,
                       MPI_COMM_WORLD);
  locally_relevant_solution.reinit(locally_relevant_dofs, MPI_COMM_WORLD);
  system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);

  rhs_function = std::make_unique<const RightHandSide<dim>>();
}



template <int dim>
void
Poisson<dim>::assemble_system()
{
  const unsigned int quadrature_degree = 2 * dg_fe.degree + 1;
  FEFaceValues<dim>  fe_face0(mapping,
                             dg_fe,
                             QGauss<dim - 1>(quadrature_degree),
                             update_values | update_JxW_values |
                               update_gradients | update_quadrature_points |
                               update_normal_vectors);


  FEValues<dim> fe_values(mapping,
                          dg_fe,
                          QGauss<dim>(quadrature_degree),
                          update_values | update_JxW_values | update_gradients |
                            update_quadrature_points);

  FEFaceValues<dim>  fe_face1(mapping,
                             dg_fe,
                             QGauss<dim - 1>(quadrature_degree),
                             update_values | update_JxW_values |
                               update_gradients | update_quadrature_points |
                               update_normal_vectors);
  const unsigned int dofs_per_cell = dg_fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<unsigned int>            visited_internal_faces;
  std::vector<unsigned int>            visited_boundary_faces;

  // Loop over standard deal.II cells
  for (const auto &cell : classical_dh.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          cell_matrix = 0.;
          cell_rhs    = 0.;

          fe_values.reinit(cell);

          const auto &        q_points  = fe_values.get_quadrature_points();
          const unsigned int  n_qpoints = q_points.size();
          std::vector<double> rhs(n_qpoints);
          rhs_function->value_list(q_points, rhs);

          for (unsigned int q_index : fe_values.quadrature_point_indices())
            {
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      cell_matrix(i, j) += fe_values.shape_grad(i, q_index) *
                                           fe_values.shape_grad(j, q_index) *
                                           fe_values.JxW(q_index);
                    }
                  cell_rhs(i) += fe_values.shape_value(i, q_index) *
                                 rhs[q_index] * fe_values.JxW(q_index);
                }
            }

          // distribute volumetric DoFs
          cell->get_dof_indices(local_dof_indices);

          for (const auto f : cell->face_indices())
            {
              const double hf = cell->face(f)->measure();
              if (cell->face(f)->at_boundary())
                {
                  visited_boundary_faces.push_back(cell->face_index(f));
                  fe_face0.reinit(cell, f);

                  const auto &normals = fe_face0.get_normal_vectors();
                  for (unsigned int q_index :
                       fe_face0.quadrature_point_indices())
                    {
                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                              cell_matrix(i, j) +=
                                (-fe_face0.shape_value(i, q_index) *
                                   fe_face0.shape_grad(j, q_index) *
                                   normals[q_index] -
                                 fe_face0.shape_grad(i, q_index) *
                                   normals[q_index] *
                                   fe_face0.shape_value(j, q_index) +
                                 (penalty / hf) *
                                   fe_face0.shape_value(i, q_index) *
                                   fe_face0.shape_value(j, q_index)) *
                                fe_face0.JxW(q_index);
                            }
                        }
                    }
                }
              else if (cell->active_cell_index() <
                       cell->neighbor(f)->active_cell_index())
                {
                  visited_internal_faces.push_back(cell->face_index(f));

                  fe_face0.reinit(cell, f);
                  fe_face1.reinit(cell->neighbor(f),
                                  cell->neighbor_of_neighbor(f));

                  std::vector<types::global_dof_index>
                    local_dof_indices_neighbor(dofs_per_cell);

                  FullMatrix<double> M11(dofs_per_cell, dofs_per_cell);
                  FullMatrix<double> M12(dofs_per_cell, dofs_per_cell);
                  FullMatrix<double> M21(dofs_per_cell, dofs_per_cell);
                  FullMatrix<double> M22(dofs_per_cell, dofs_per_cell);

                  const auto &normals = fe_face1.get_normal_vectors();
                  // M11
                  for (unsigned int q_index :
                       fe_face0.quadrature_point_indices())
                    {
#ifdef AGGLO_DEBUG
                      pcout << normals[q_index] << std::endl;
#endif
                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                              M11(i, j) +=
                                (-0.5 * fe_face0.shape_grad(i, q_index) *
                                   normals[q_index] *
                                   fe_face0.shape_value(j, q_index) -
                                 0.5 * fe_face0.shape_grad(j, q_index) *
                                   normals[q_index] *
                                   fe_face0.shape_value(i, q_index) +
                                 (penalty / hf) *
                                   fe_face0.shape_value(i, q_index) *
                                   fe_face0.shape_value(j, q_index)) *
                                fe_face0.JxW(q_index);
                            }
                        }
                    }
                  // M12
                  for (unsigned int q_index :
                       fe_face0.quadrature_point_indices())
                    {
                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                              M12(i, j) +=
                                (0.5 * fe_face0.shape_grad(i, q_index) *
                                   normals[q_index] *
                                   fe_face1.shape_value(j, q_index) -
                                 0.5 * fe_face1.shape_grad(j, q_index) *
                                   normals[q_index] *
                                   fe_face0.shape_value(i, q_index) -
                                 (penalty / hf) *
                                   fe_face0.shape_value(i, q_index) *
                                   fe_face1.shape_value(j, q_index)) *
                                fe_face1.JxW(q_index);
                            }
                        }
                    }
                  // A10
                  for (unsigned int q_index :
                       fe_face0.quadrature_point_indices())
                    {
                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                              M21(i, j) +=
                                (-0.5 * fe_face1.shape_grad(i, q_index) *
                                   normals[q_index] *
                                   fe_face0.shape_value(j, q_index) +
                                 0.5 * fe_face0.shape_grad(j, q_index) *
                                   normals[q_index] *
                                   fe_face1.shape_value(i, q_index) -
                                 (penalty / hf) *
                                   fe_face1.shape_value(i, q_index) *
                                   fe_face0.shape_value(j, q_index)) *
                                fe_face1.JxW(q_index);
                            }
                        }
                    }
                  // A11
                  for (unsigned int q_index :
                       fe_face0.quadrature_point_indices())
                    {
                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                              M22(i, j) +=
                                (0.5 * fe_face1.shape_grad(i, q_index) *
                                   normals[q_index] *
                                   fe_face1.shape_value(j, q_index) +
                                 0.5 * fe_face1.shape_grad(j, q_index) *
                                   normals[q_index] *
                                   fe_face1.shape_value(i, q_index) +
                                 (penalty / hf) *
                                   fe_face1.shape_value(i, q_index) *
                                   fe_face1.shape_value(j, q_index)) *
                                fe_face1.JxW(q_index);
                            }
                        }
                    }

// distribute DoFs accordingly
#ifdef AGGLO_DEBUG
                  pcout << "Neighbor is " << cell->neighbor(f) << std::endl;
#endif
                  cell->neighbor(f)->get_dof_indices(
                    local_dof_indices_neighbor);

                  system_matrix.add(local_dof_indices, M11);
                  system_matrix.add(local_dof_indices,
                                    local_dof_indices_neighbor,
                                    M12);
                  system_matrix.add(local_dof_indices_neighbor,
                                    local_dof_indices,
                                    M21);
                  system_matrix.add(local_dof_indices_neighbor, M22);

                } // check idx neighbors
            }     // over faces

          constraints.distribute_local_to_global(cell_matrix,
                                                 cell_rhs,
                                                 local_dof_indices,
                                                 system_matrix,
                                                 system_rhs);
        }
    }
  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}


template <int dim>
void
Poisson<dim>::solve()
{
  TrilinosWrappers::MPI::Vector completely_distributed_solution(system_rhs);
  solver.solve(system_matrix, completely_distributed_solution, system_rhs);
  constraints.distribute(completely_distributed_solution);
  locally_relevant_solution = completely_distributed_solution;
}



template <int dim>
void
Poisson<dim>::output_results()
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(classical_dh);
  data_out.add_data_vector(locally_relevant_solution,
                           "u",
                           DataOut<dim>::type_dof_data);

  Vector<float> subdomain(tria.n_active_cells());

  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = tria.locally_owned_subdomain();

  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches(mapping);

  const std::string filename =
    ("agglomerated_Poisson_classic_DG." +
     Utilities::int_to_string(tria.locally_owned_subdomain(), 4));

  std::ofstream output((filename + ".vtu").c_str());
  data_out.write_vtu(output);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::vector<std::string> filenames;
      for (unsigned int i = 0;
           i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
           i++)
        {
          filenames.push_back("solution." + Utilities::int_to_string(i, 4) +
                              ".vtu");
        }
      std::ofstream master_output("solution.pvtu");
      data_out.write_pvtu_record(master_output, filenames);
    }
}

template <int dim>
void
Poisson<dim>::run()
{
  make_grid();
  assemble_system();
  solve();
  output_results();
}

int
main(int argc, char *argv[])
{
  const unsigned int               fe_degree = 1;
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  Poisson<2>                       poisson_problem(fe_degree);
  poisson_problem.run();
  return 0;
}
