/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2022 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 */


// Similar to exact_solutions.cc, but in the distributed setting.



#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_integrate_difference.templates.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <agglomeration_handler.h>
#include <poly_utils.h>


using namespace dealii;

static constexpr double TOL = 1e-14;

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

  virtual void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<double> &          values,
             const unsigned int /*component*/) const override;

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
void
LinearFunction<dim>::value_list(const std::vector<Point<dim>> &points,
                                std::vector<double> &          values,
                                const unsigned int /*component*/) const
{
  for (unsigned int i = 0; i < values.size(); ++i)
    values[i] = value(points[i]);
}

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
             const unsigned int /*component*/ = 0) const override;
};


template <int dim>
void
RightHandSide<dim>::value_list(const std::vector<Point<dim>> &points,
                               std::vector<double> &          values,
                               const unsigned int /*component*/) const
{
  (void)points;
  for (unsigned int i = 0; i < values.size(); ++i)
    values[i] = 0.;
}



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
             std::vector<double> &          values,
             const unsigned int /*component*/) const override;

  virtual Tensor<1, dim>
  gradient(const Point<dim> & p,
           const unsigned int component = 0) const override;
};

template <int dim>
double
SolutionLinear<dim>::value(const Point<dim> &p, const unsigned int) const
{
  return p[0] + p[1] - 1; // linear
}

template <int dim>
Tensor<1, dim>
SolutionLinear<dim>::gradient(const Point<dim> &p, const unsigned int) const
{
  Assert(dim == 2, ExcMessage("This test only works in 2D."));
  (void)p;
  Tensor<1, dim> return_value;
  return_value[0] = 1.;
  return_value[1] = 1.;
  return return_value;
}


template <int dim>
void
SolutionLinear<dim>::value_list(const std::vector<Point<dim>> &points,
                                std::vector<double> &          values,
                                const unsigned int /*component*/) const
{
  for (unsigned int i = 0; i < values.size(); ++i)
    values[i] = this->value(points[i]);
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  const MPI_Comm &                 comm = MPI_COMM_WORLD;
  const unsigned n_ranks                = Utilities::MPI::n_mpi_processes(comm);
  Assert(n_ranks == 3,
         ExcMessage("This test is meant to be run with 3 ranks only."));
  if (Utilities::MPI::this_mpi_process(comm) == 0)
    std::cout << "Running with " << n_ranks << " MPI ranks." << std::endl;

  parallel::distributed::Triangulation<2> tria(comm);

  GridGenerator::hyper_cube(tria);
  tria.refine_global(3);

  AffineConstraints<double> constraints;
  constraints.close();
  TrilinosWrappers::SparseMatrix system_matrix;
  TrilinosWrappers::MPI::Vector  system_rhs;

  GridTools::Cache<2>     cached_tria(tria);
  AgglomerationHandler<2> ah(cached_tria);

  const auto &get_all_locally_owned_indices = [&tria]() {
    std::vector<types::global_cell_index> local_indices;
    for (const auto &cell : tria.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          local_indices.push_back(cell->active_cell_index());
      }
    return local_indices;
  };

  // For each rank, store each locally owned cell as a polytope
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
      ah.define_agglomerate({cell});



  // Check ghosted indices
  FE_DGQ<2> fe_dg(1);

  const unsigned int quadrature_degree      = 2 * fe_dg.get_degree() + 1;
  const unsigned int face_quadrature_degree = 2 * fe_dg.get_degree() + 1;
  ah.initialize_fe_values(QGauss<2>(quadrature_degree),
                          update_values | update_gradients | update_JxW_values |
                            update_quadrature_points,
                          QGauss<1>(face_quadrature_degree),
                          update_JxW_values);

  ah.distribute_agglomerated_dofs(fe_dg);

  DynamicSparsityPattern sparsity_pattern;
  ah.create_agglomeration_sparsity_pattern(sparsity_pattern);

  const IndexSet &locally_owned_dofs = ah.agglo_dh.locally_owned_dofs();
  const IndexSet  locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(ah.agglo_dh);

  system_matrix.reinit(locally_owned_dofs,
                       locally_owned_dofs,
                       sparsity_pattern,
                       comm);
  system_rhs.reinit(locally_owned_dofs, comm);

  std::unique_ptr<const RightHandSide<2>> rhs_function;
  rhs_function = std::make_unique<const RightHandSide<2>>();



  const unsigned int dofs_per_cell = fe_dg.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  FullMatrix<double> M11(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> M12(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> M21(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> M22(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices_bdary_cell(
    dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices_neighbor(
    dofs_per_cell);


  SolutionLinear<2> analytical_solution;
  auto              polytope   = ah.begin();
  double            test_bdary = 0.;
  LinearFunction<2> linear_func{{1, 1}};
  double            test_integral = 0.;
  double            test_volume   = 0.;

  for (; polytope != ah.end(); ++polytope)
    {
      if (polytope->is_locally_owned())
        {
          cell_matrix = 0.;
          cell_rhs    = 0.;

          const auto &agglo_values = ah.reinit(polytope);

          const auto &        q_points  = agglo_values.get_quadrature_points();
          const unsigned int  n_qpoints = q_points.size();
          std::vector<double> rhs(n_qpoints);
          rhs_function->value_list(q_points, rhs);

          std::vector<double> linear_values(n_qpoints);
          linear_func.value_list(q_points, linear_values, 1);

          for (unsigned int q_index : agglo_values.quadrature_point_indices())
            {
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      cell_matrix(i, j) += agglo_values.shape_grad(i, q_index) *
                                           agglo_values.shape_grad(j, q_index) *
                                           agglo_values.JxW(q_index);
                    }
                  cell_rhs(i) += agglo_values.shape_value(i, q_index) *
                                 rhs[q_index] * agglo_values.JxW(q_index);
                }
              test_integral +=
                linear_values[q_index] * agglo_values.JxW(q_index);
              test_volume += agglo_values.JxW(q_index);
            }

          // distribute volumetric DoFs
          polytope->get_dof_indices(local_dof_indices);


          // Face terms
          unsigned int n_faces = polytope->n_faces();

          const double dummy_hf = polytope.master_cell()->face(0)->measure();
          const double dummy_penalty = 10.;
          for (unsigned int f = 0; f < n_faces; ++f)
            {
              if (polytope->at_boundary(f))
                {
                  // Get normal vectors seen from each agglomeration.
                  const auto &fe_face = ah.reinit(polytope, f);
                  const auto &normals = fe_face.get_normal_vectors();

                  const auto &face_q_points = fe_face.get_quadrature_points();

                  std::vector<double> analytical_solution_values(
                    face_q_points.size());
                  analytical_solution.value_list(face_q_points,
                                                 analytical_solution_values,
                                                 1);

                  for (unsigned int q_index :
                       fe_face.quadrature_point_indices())
                    {
                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                              cell_matrix(i, j) +=
                                (-fe_face.shape_value(i, q_index) *
                                   fe_face.shape_grad(j, q_index) *
                                   normals[q_index] -
                                 fe_face.shape_grad(i, q_index) *
                                   normals[q_index] *
                                   fe_face.shape_value(j, q_index) +
                                 (dummy_penalty / dummy_hf) *
                                   fe_face.shape_value(i, q_index) *
                                   fe_face.shape_value(j, q_index)) *
                                fe_face.JxW(q_index);
                            }
                          cell_rhs(i) +=
                            ((dummy_penalty / dummy_hf) *
                               analytical_solution_values[q_index] *
                               fe_face.shape_value(i, q_index) -
                             fe_face.shape_grad(i, q_index) * normals[q_index] *
                               analytical_solution_values[q_index]) *
                            fe_face.JxW(q_index);
                        }
                      test_bdary += fe_face.JxW(q_index);
                    }

                  // distribute DoFs
                  polytope->get_dof_indices(local_dof_indices_bdary_cell);
                }
              else
                {
                  const auto &neigh_polytope = polytope->neighbor(f);
                  if (polytope->id() < neigh_polytope->id())
                    {
                      unsigned int nofn =
                        polytope->neighbor_of_agglomerated_neighbor(f);

                      Assert(neigh_polytope->neighbor(nofn)->id() ==
                               polytope->id(),
                             ExcMessage("Impossible."));

                      const auto &fe_faces =
                        ah.reinit_interface(polytope, neigh_polytope, f, nofn);


                      const auto &fe_faces0 = fe_faces.first;
                      const auto &normals   = fe_faces0.get_normal_vectors();

                      if (neigh_polytope->is_locally_owned())
                        {
                          // use both fevalues
                          const auto &fe_faces1 = fe_faces.second;

                          M11 = 0.;
                          M12 = 0.;
                          M21 = 0.;
                          M22 = 0.;

                          // M11
                          for (unsigned int q_index :
                               fe_faces0.quadrature_point_indices())
                            {
                              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                                {
                                  for (unsigned int j = 0; j < dofs_per_cell;
                                       ++j)
                                    {
                                      M11(i, j) +=
                                        (-0.5 *
                                           fe_faces0.shape_grad(i, q_index) *
                                           normals[q_index] *
                                           fe_faces0.shape_value(j, q_index) -
                                         0.5 *
                                           fe_faces0.shape_grad(j, q_index) *
                                           normals[q_index] *
                                           fe_faces0.shape_value(i, q_index) +
                                         (dummy_penalty / dummy_hf) *
                                           fe_faces0.shape_value(i, q_index) *
                                           fe_faces0.shape_value(j, q_index)) *
                                        fe_faces0.JxW(q_index);

                                      M12(i, j) +=
                                        (0.5 *
                                           fe_faces0.shape_grad(i, q_index) *
                                           normals[q_index] *
                                           fe_faces1.shape_value(j, q_index) -
                                         0.5 *
                                           fe_faces1.shape_grad(j, q_index) *
                                           normals[q_index] *
                                           fe_faces0.shape_value(i, q_index) -
                                         (dummy_penalty / dummy_hf) *
                                           fe_faces0.shape_value(i, q_index) *
                                           fe_faces1.shape_value(j, q_index)) *
                                        fe_faces1.JxW(q_index);

                                      // A10
                                      M21(i, j) +=
                                        (-0.5 *
                                           fe_faces1.shape_grad(i, q_index) *
                                           normals[q_index] *
                                           fe_faces0.shape_value(j, q_index) +
                                         0.5 *
                                           fe_faces0.shape_grad(j, q_index) *
                                           normals[q_index] *
                                           fe_faces1.shape_value(i, q_index) -
                                         (dummy_penalty / dummy_hf) *
                                           fe_faces1.shape_value(i, q_index) *
                                           fe_faces0.shape_value(j, q_index)) *
                                        fe_faces1.JxW(q_index);

                                      // A11
                                      M22(i, j) +=
                                        (0.5 *
                                           fe_faces1.shape_grad(i, q_index) *
                                           normals[q_index] *
                                           fe_faces1.shape_value(j, q_index) +
                                         0.5 *
                                           fe_faces1.shape_grad(j, q_index) *
                                           normals[q_index] *
                                           fe_faces1.shape_value(i, q_index) +
                                         (dummy_penalty / dummy_hf) *
                                           fe_faces1.shape_value(i, q_index) *
                                           fe_faces1.shape_value(j, q_index)) *
                                        fe_faces1.JxW(q_index);
                                    }
                                }
                            }

                          // distribute DoFs accordingly
                          // fluxes
                          neigh_polytope->get_dof_indices(
                            local_dof_indices_neighbor);

                          constraints.distribute_local_to_global(
                            M11, local_dof_indices, system_matrix);
                          constraints.distribute_local_to_global(
                            M12,
                            local_dof_indices,
                            local_dof_indices_neighbor,
                            system_matrix);
                          constraints.distribute_local_to_global(
                            M21,
                            local_dof_indices_neighbor,
                            local_dof_indices,
                            system_matrix);
                          constraints.distribute_local_to_global(
                            M22, local_dof_indices_neighbor, system_matrix);
                        }
                      else
                        {
                          // neigh polytope is ghosted, assemble really by hand
                          // retrieving values and qpoints



                          types::subdomain_id neigh_rank =
                            neigh_polytope->subdomain_id();



                          const auto &test_jxws =
                            ah.recv_jxws.at(neigh_rank)
                              .at({neigh_polytope->id(), nofn});

                          const auto &recv_values =
                            ah.recv_values.at(neigh_rank)
                              .at({neigh_polytope->id(), nofn});

                          const auto &recv_gradients =
                            ah.recv_gradients.at(neigh_rank)
                              .at({neigh_polytope->id(), nofn});



                          {
                            const auto &test_points =
                              ah.recv_qpoints.at(neigh_rank)
                                .at({neigh_polytope->id(), nofn});

                            const auto &points0 =
                              fe_faces0.get_quadrature_points();
                            const auto &points1 = test_points;
                            for (size_t i = 0; i < points0.size(); ++i)
                              {
                                double d = (points0[i] - points1[i]).norm();
                                AssertThrow(
                                  d < 1e-15,
                                  ExcMessage(
                                    "Face qpoints at the interface do not match!"));
                              }


                            // Check on JxWs
                            const auto &jxws0 = fe_faces0.get_JxW_values();
                            const auto &jxws1 = test_jxws;
                            for (size_t i = 0; i < jxws0.size(); ++i)
                              {
                                double d = (jxws0[i] - jxws1[i]);
                                AssertThrow(
                                  std::fabs(d) < 1e-15,
                                  ExcMessage(
                                    "JxWs at the interface do not match!"));
                              }
                          }

                          M11 = 0.;
                          M12 = 0.;
                          M21 = 0.;
                          M22 = 0.;

                          // M11
                          for (unsigned int q_index :
                               fe_faces0.quadrature_point_indices())
                            {
                              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                                {
                                  for (unsigned int j = 0; j < dofs_per_cell;
                                       ++j)
                                    {
                                      M11(i, j) +=
                                        (-0.5 *
                                           fe_faces0.shape_grad(i, q_index) *
                                           normals[q_index] *
                                           fe_faces0.shape_value(j, q_index) -
                                         0.5 *
                                           fe_faces0.shape_grad(j, q_index) *
                                           normals[q_index] *
                                           fe_faces0.shape_value(i, q_index) +
                                         (dummy_penalty / dummy_hf) *
                                           fe_faces0.shape_value(i, q_index) *
                                           fe_faces0.shape_value(j, q_index)) *
                                        fe_faces0.JxW(q_index);

                                      M12(i, j) +=
                                        (0.5 *
                                           fe_faces0.shape_grad(i, q_index) *
                                           normals[q_index] *
                                           recv_values[j][q_index] -
                                         0.5 * recv_gradients[j][q_index] *
                                           normals[q_index] *
                                           fe_faces0.shape_value(i, q_index) -
                                         (dummy_penalty / dummy_hf) *
                                           fe_faces0.shape_value(i, q_index) *
                                           recv_values[j][q_index]) *
                                        test_jxws[q_index];

                                      // A10
                                      M21(i, j) +=
                                        (-0.5 * recv_gradients[i][q_index] *
                                           normals[q_index] *
                                           fe_faces0.shape_value(j, q_index) +
                                         0.5 *
                                           fe_faces0.shape_grad(j, q_index) *
                                           normals[q_index] *
                                           recv_values[i][q_index] -
                                         (dummy_penalty / dummy_hf) *
                                           recv_values[i][q_index] *
                                           fe_faces0.shape_value(j, q_index)) *
                                        test_jxws[q_index];

                                      // A11
                                      M22(i, j) +=
                                        (0.5 * recv_gradients[i][q_index] *
                                           normals[q_index] *
                                           recv_values[j][q_index] +
                                         0.5 * recv_gradients[j][q_index] *
                                           normals[q_index] *
                                           recv_values[i][q_index] +
                                         (dummy_penalty / dummy_hf) *
                                           recv_values[i][q_index] *
                                           recv_values[j][q_index]) *
                                        test_jxws[q_index];
                                    }
                                }
                            }
                          // distribute DoFs accordingly
                          // fluxes
                          neigh_polytope->get_dof_indices(
                            local_dof_indices_neighbor);

                          constraints.distribute_local_to_global(
                            M11, local_dof_indices, system_matrix);
                          constraints.distribute_local_to_global(
                            M12,
                            local_dof_indices,
                            local_dof_indices_neighbor,
                            system_matrix);
                          constraints.distribute_local_to_global(
                            M21,
                            local_dof_indices_neighbor,
                            local_dof_indices,
                            system_matrix);
                          constraints.distribute_local_to_global(
                            M22, local_dof_indices_neighbor, system_matrix);
                        } // ghosted polytope case


                    } // only once
                }     // internal face
            }         // face loop

          constraints.distribute_local_to_global(cell_matrix,
                                                 cell_rhs,
                                                 local_dof_indices,
                                                 system_matrix,
                                                 system_rhs);

        } // locally owned polytopes
    }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  TrilinosWrappers::MPI::Vector locally_relevant_solution;
  locally_relevant_solution.reinit(locally_relevant_dofs, comm);


  SolverControl                  control;
  TrilinosWrappers::SolverDirect solver(control);

  TrilinosWrappers::MPI::Vector completely_distributed_solution(system_rhs);
  solver.solve(system_matrix, completely_distributed_solution, system_rhs);
  constraints.distribute(completely_distributed_solution);
  locally_relevant_solution = completely_distributed_solution;


  // Compute L2 error
  {
    TrilinosWrappers::MPI::Vector interpolated_solution;

    PolyUtils::interpolate_to_fine_grid(ah,
                                        interpolated_solution,
                                        completely_distributed_solution);

    Vector<double> cellwise_error(locally_relevant_solution.size());
    VectorTools::integrate_difference(ah.output_dh,
                                      locally_relevant_solution,
                                      SolutionLinear<2>(),
                                      cellwise_error,
                                      QGauss<2>(2 * fe_dg.degree + 1),
                                      VectorTools::NormType::L2_norm);
    const double error =
      VectorTools::compute_global_error(tria,
                                        cellwise_error,
                                        VectorTools::NormType::L2_norm);

    double bdary_accumulated    = Utilities::MPI::sum(test_bdary, comm);
    double integral_accumulated = Utilities::MPI::sum(test_integral, comm);
    double volume               = Utilities::MPI::sum(test_volume, comm);
    AssertThrow(error < TOL, ExcMessage("L2 error too large."));
    if (Utilities::MPI::this_mpi_process(comm) == 0)
      {
        std::cout << "Volumetric check: " << volume << std::endl;
        std::cout << "Boundary check: " << bdary_accumulated << std::endl;
        std::cout << "Integral check: " << integral_accumulated << std::endl;

        std::cout << "Linear: OK" << std::endl;
      }
  }
}
