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


// Solve the diffusion reaction problem
// -Delta u + cu = f in \Omega
//             u = g on \partial Omega
// where Omega = [0,1]^3
//

#define HEX

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
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
#include <poly_utils.h>

using namespace dealii;


/**
 * Utility to compute jump terms when the interface is locally owned, i.e. both
 * elements are locally owned.
 */
template <int dim>
void
assemble_local_jumps_and_averages(FullMatrix<double>      &M11,
                                  FullMatrix<double>      &M12,
                                  FullMatrix<double>      &M21,
                                  FullMatrix<double>      &M22,
                                  const FEValuesBase<dim> &fe_faces0,
                                  const FEValuesBase<dim> &fe_faces1,
                                  const double             penalty_constant,
                                  const double             h_f)
{
  const std::vector<Tensor<1, dim>> &normals = fe_faces0.get_normal_vectors();
  const unsigned int                 dofs_per_cell =
    M11.m(); // size of local matrices equals the #DoFs

  for (unsigned int q_index : fe_faces0.quadrature_point_indices())
    {
      const Tensor<1, dim> &normal = normals[q_index];
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              M11(i, j) +=
                (-0.5 * fe_faces0.shape_grad(i, q_index) * normal *
                   fe_faces0.shape_value(j, q_index) -
                 0.5 * fe_faces0.shape_grad(j, q_index) * normal *
                   fe_faces0.shape_value(i, q_index) +
                 (penalty_constant / h_f) * fe_faces0.shape_value(i, q_index) *
                   fe_faces0.shape_value(j, q_index)) *
                fe_faces0.JxW(q_index);

              M12(i, j) +=
                (0.5 * fe_faces0.shape_grad(i, q_index) * normal *
                   fe_faces1.shape_value(j, q_index) -
                 0.5 * fe_faces1.shape_grad(j, q_index) * normal *
                   fe_faces0.shape_value(i, q_index) -
                 (penalty_constant / h_f) * fe_faces0.shape_value(i, q_index) *
                   fe_faces1.shape_value(j, q_index)) *
                fe_faces1.JxW(q_index);


              M21(i, j) +=
                (-0.5 * fe_faces1.shape_grad(i, q_index) * normal *
                   fe_faces0.shape_value(j, q_index) +
                 0.5 * fe_faces0.shape_grad(j, q_index) * normal *
                   fe_faces1.shape_value(i, q_index) -
                 (penalty_constant / h_f) * fe_faces1.shape_value(i, q_index) *
                   fe_faces0.shape_value(j, q_index)) *
                fe_faces1.JxW(q_index);


              M22(i, j) +=
                (0.5 * fe_faces1.shape_grad(i, q_index) * normal *
                   fe_faces1.shape_value(j, q_index) +
                 0.5 * fe_faces1.shape_grad(j, q_index) * normal *
                   fe_faces1.shape_value(i, q_index) +
                 (penalty_constant / h_f) * fe_faces1.shape_value(i, q_index) *
                   fe_faces1.shape_value(j, q_index)) *
                fe_faces1.JxW(q_index);
            }
        }
    }
}



/**
 * Same as above, but for a ghosted neighbor.
 */
template <int dim>
void
assemble_local_jumps_and_averages_ghost(
  FullMatrix<double>                             &M11,
  FullMatrix<double>                             &M12,
  FullMatrix<double>                             &M21,
  FullMatrix<double>                             &M22,
  const FEValuesBase<dim>                        &fe_faces0,
  const std::vector<std::vector<double>>         &recv_values,
  const std::vector<std::vector<Tensor<1, dim>>> &recv_gradients,
  const std::vector<double>                      &recv_jxws,
  const double                                    penalty_constant,
  const double                                    h_f)
{
  Assert(
    (recv_values.size() > 0 && recv_gradients.size() && recv_jxws.size()),
    ExcMessage(
      "Not possible to assemble jumps and averages at a ghosted interface."));
  const unsigned int dofs_per_cell = M11.m();

  const std::vector<Tensor<1, dim>> &normals = fe_faces0.get_normal_vectors();
  for (unsigned int q_index : fe_faces0.quadrature_point_indices())
    {
      const Tensor<1, dim> &normal = normals[q_index];
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              M11(i, j) +=
                (-0.5 * fe_faces0.shape_grad(i, q_index) * normal *
                   fe_faces0.shape_value(j, q_index) -
                 0.5 * fe_faces0.shape_grad(j, q_index) * normal *
                   fe_faces0.shape_value(i, q_index) +
                 (penalty_constant / h_f) * fe_faces0.shape_value(i, q_index) *
                   fe_faces0.shape_value(j, q_index)) *
                fe_faces0.JxW(q_index);

              M12(i, j) +=
                (0.5 * fe_faces0.shape_grad(i, q_index) * normal *
                   recv_values[j][q_index] -
                 0.5 * recv_gradients[j][q_index] * normal *
                   fe_faces0.shape_value(i, q_index) -
                 (penalty_constant / h_f) * fe_faces0.shape_value(i, q_index) *
                   recv_values[j][q_index]) *
                recv_jxws[q_index];


              M21(i, j) += (-0.5 * recv_gradients[i][q_index] * normal *
                              fe_faces0.shape_value(j, q_index) +
                            0.5 * fe_faces0.shape_grad(j, q_index) * normal *
                              recv_values[i][q_index] -
                            (penalty_constant / h_f) * recv_values[i][q_index] *
                              fe_faces0.shape_value(j, q_index)) *
                           recv_jxws[q_index];


              M22(i, j) += (0.5 * recv_gradients[i][q_index] * normal *
                              recv_values[j][q_index] +
                            0.5 * recv_gradients[j][q_index] * normal *
                              recv_values[i][q_index] +
                            (penalty_constant / h_f) * recv_values[i][q_index] *
                              recv_values[j][q_index]) *
                           recv_jxws[q_index];
            }
        }
    }
}



template <int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide(const double c = 0.5)
    : Function<dim>()
  {
    reaction_coefficient = c;
  }

  virtual void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<double>           &values,
             const unsigned int /*component*/ = 0) const override;

private:
  double reaction_coefficient;
};


template <int dim>
void
RightHandSide<dim>::value_list(const std::vector<Point<dim>> &points,
                               std::vector<double>           &values,
                               const unsigned int /*component*/) const
{
  for (unsigned int i = 0; i < values.size(); ++i)
    {
      const double x = points[i][0];
      const double y = points[i][1];
      const double z = points[i][2];
      values[i] =
        -std::exp(x * y * z) * ((x * y) * (x * y) + (x * z) * (x * z) +
                                (y * z) * (y * z) - reaction_coefficient);
    }
}



template <int dim>
class Solution : public Function<dim>
{
public:
  Solution()
    : Function<dim>()
  {
    static_assert(dim == 3, "Only 3D case is implemented.");
  }

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

  virtual void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<double>           &values,
             const unsigned int /*component*/) const override;

  virtual Tensor<1, dim>
  gradient(const Point<dim>  &p,
           const unsigned int component = 0) const override;
};

template <int dim>
double
Solution<dim>::value(const Point<dim> &p, const unsigned int) const
{
  return std::exp(p[0] * p[1] * p[2]);
}

template <int dim>
Tensor<1, dim>
Solution<dim>::gradient(const Point<dim> &p, const unsigned int) const
{
  Tensor<1, dim> grad;
  const double   sol_at_p = std::exp(p[0] * p[1] * p[2]);
  grad[0]                 = p[1] * p[2] * sol_at_p;
  grad[1]                 = p[0] * p[2] * sol_at_p;
  grad[2]                 = p[0] * p[1] * sol_at_p;
  return grad;
}


template <int dim>
void
Solution<dim>::value_list(const std::vector<Point<dim>> &points,
                          std::vector<double>           &values,
                          const unsigned int /*component*/) const
{
  for (unsigned int i = 0; i < values.size(); ++i)
    values[i] = this->value(points[i]);
}



/*--------------------------------------------------------------------------*/
// Class describing the 3D diffusion reaction problem.

template <int dim>
class DiffusionReactionProblem
{
public:
  DiffusionReactionProblem(const unsigned int n_local_agglomerates,
                           const unsigned int degree,
                           const double       reaction_coefficient,
                           const MPI_Comm     comm);
  void
  run();

private:
  void
  make_fine_grid(const unsigned int);
  void
  setup_agglomerated_problem(const unsigned int);
  void
  assemble_system();
  void
  solve();
  void
  output_results() const;

  const MPI_Comm comm;
#ifdef HEX
  MappingQ1<dim> mapping;
  FE_DGQ<dim>    fe_dg;
#else
  MappingFE<dim>     mapping;
  FE_SimplexDGP<dim> fe_dg;
#endif
  const unsigned int n_ranks;

  const unsigned int                             n_local_agglomerates;
  const double                                   reaction_coefficient;
  parallel::fullydistributed::Triangulation<dim> tria_pft;
  ConditionalOStream                             pcout;
  double                                         penalty_constant;


  std::unique_ptr<AgglomerationHandler<dim>> ah;
  AffineConstraints<double>                  constraints;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  TrilinosWrappers::SparseMatrix system_matrix;
  TrilinosWrappers::MPI::Vector  system_rhs;
  TrilinosWrappers::MPI::Vector  locally_relevant_solution;
  TrilinosWrappers::MPI::Vector  interpolated_solution;

  Solution<dim> analytical_solution;
};



template <int dim>
DiffusionReactionProblem<dim>::DiffusionReactionProblem(
  const unsigned int n_local_agglomerates,
  const unsigned int degree,
  const double       reaction_coefficient,
  const MPI_Comm     communicator)
  : comm(communicator)
#ifdef HEX
  , mapping()
#else
  , mapping(FE_SimplexP<dim>{1})
#endif
  , fe_dg(degree)
  , n_ranks(Utilities::MPI::n_mpi_processes(comm))
  , n_local_agglomerates(n_local_agglomerates)
  , reaction_coefficient(reaction_coefficient)
  , tria_pft(comm)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(comm) == 0))
{
  penalty_constant = 10. * degree * degree;
  pcout << "Running with " << n_ranks << " MPI ranks." << std::endl;
}



template <int dim>
void
DiffusionReactionProblem<dim>::make_fine_grid(
  const unsigned int n_global_refinements)
{
  Triangulation<dim> tria;
#ifdef HEX
  GridGenerator::hyper_cube(tria);
  tria.refine_global(n_global_refinements);
#else
  Triangulation<dim> tria_hex;
  GridGenerator::hyper_cube(tria_hex, 0., 1.);
  tria_hex.refine_global(n_global_refinements - 1);
  GridGenerator::convert_hypercube_to_simplex_mesh(tria_hex, tria);
#endif

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
DiffusionReactionProblem<dim>::setup_agglomerated_problem(
  const unsigned int n_local_agglomerates)
{
  GridTools::Cache<dim> cached_tria(tria_pft);

  ah = std::make_unique<AgglomerationHandler<dim>>(cached_tria);

  //   pcout << "Rank " << my_rank << " has "
  //         << tria_pft.n_locally_owned_active_cells() << std::endl;

  // Call the METIS partitioner to agglomerate within each processor.
  PolyUtils::partition_locally_owned_regions(n_local_agglomerates,
                                             tria_pft,
                                             SparsityTools::Partitioner::metis);
  pcout << "Number of cells: " << tria_pft.n_global_active_cells() << std::endl;

  // Agglomerate cells together based on their material id
  std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>>
    cells_per_material_id(n_local_agglomerates);
  for (const auto &cell : tria_pft.active_cell_iterators())
    if (cell->is_locally_owned())
      cells_per_material_id[cell->material_id()].push_back(cell);


  // Agglomerate elements with same id
  for (std::size_t i = 0; i < cells_per_material_id.size(); ++i)
    ah->define_agglomerate(cells_per_material_id[i]);
}



template <int dim>
void
DiffusionReactionProblem<dim>::assemble_system()
{
  constraints.close();

  const unsigned int quadrature_degree      = fe_dg.get_degree() + 1;
  const unsigned int face_quadrature_degree = fe_dg.get_degree() + 1;
#ifdef HEX
  ah->initialize_fe_values(QGauss<dim>(quadrature_degree),
                           update_gradients | update_JxW_values |
                             update_quadrature_points | update_JxW_values |
                             update_values,
                           QGauss<dim - 1>(face_quadrature_degree));
#else
  ah->initialize_fe_values(QGaussSimplex<dim>(quadrature_degree),
                           update_gradients | update_JxW_values |
                             update_quadrature_points | update_JxW_values |
                             update_values,
                           QGaussSimplex<dim - 1>(face_quadrature_degree));
#endif

  ah->distribute_agglomerated_dofs(fe_dg);

  TrilinosWrappers::SparsityPattern dsp;
  ah->create_agglomeration_sparsity_pattern(dsp);
  system_matrix.reinit(dsp);

  locally_owned_dofs    = ah->agglo_dh.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(ah->agglo_dh);
  system_rhs.reinit(locally_owned_dofs, comm);

  std::unique_ptr<const RightHandSide<dim>> rhs_function;
  rhs_function =
    std::make_unique<const RightHandSide<dim>>(reaction_coefficient);

  const unsigned int dofs_per_cell = fe_dg.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  FullMatrix<double> M11(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> M12(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> M21(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> M22(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices_neighbor(
    dofs_per_cell);

  double start_assembly, stop_assembly;
  start_assembly = MPI_Wtime();
  auto polytope  = ah->begin();
  for (; polytope != ah->end(); ++polytope)
    {
      if (polytope->is_locally_owned())
        {
          cell_matrix = 0.;
          cell_rhs    = 0.;

          const auto &agglo_values = ah->reinit(polytope);

          const auto         &q_points  = agglo_values.get_quadrature_points();
          const unsigned int  n_qpoints = q_points.size();
          std::vector<double> rhs(n_qpoints);
          rhs_function->value_list(q_points, rhs);



          for (unsigned int q_index : agglo_values.quadrature_point_indices())
            {
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      cell_matrix(i, j) +=
                        (agglo_values.shape_grad(i, q_index) *
                           agglo_values.shape_grad(j, q_index) +
                         reaction_coefficient *
                           agglo_values.shape_value(i, q_index) *
                           agglo_values.shape_value(j, q_index)) *
                        agglo_values.JxW(q_index);
                    }
                  cell_rhs(i) += agglo_values.shape_value(i, q_index) *
                                 rhs[q_index] * agglo_values.JxW(q_index);
                }
            }

          // get volumetric DoFs
          polytope->get_dof_indices(local_dof_indices);


          // Assemble face terms
          unsigned int n_faces = polytope->n_faces();

          const double h_f = polytope->diameter();
          for (unsigned int f = 0; f < n_faces; ++f)
            {
              if (polytope->at_boundary(f))
                {
                  // Get normal vectors seen from each agglomeration.
                  const auto &fe_face = ah->reinit(polytope, f);
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
                      const Tensor<1, dim> &normal = normals[q_index];
                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                              cell_matrix(i, j) +=
                                (-fe_face.shape_value(i, q_index) *
                                   fe_face.shape_grad(j, q_index) * normal -
                                 fe_face.shape_grad(i, q_index) * normal *
                                   fe_face.shape_value(j, q_index) +
                                 (penalty_constant / h_f) *
                                   fe_face.shape_value(i, q_index) *
                                   fe_face.shape_value(j, q_index)) *
                                fe_face.JxW(q_index);
                            }
                          cell_rhs(i) +=
                            ((penalty_constant / h_f) *
                               analytical_solution_values[q_index] *
                               fe_face.shape_value(i, q_index) -
                             fe_face.shape_grad(i, q_index) * normal *
                               analytical_solution_values[q_index]) *
                            fe_face.JxW(q_index);
                        }
                    }
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
                             ExcMessage("Mismatch."));

                      const auto &fe_faces =
                        ah->reinit_interface(polytope, neigh_polytope, f, nofn);
                      const auto &fe_faces0 = fe_faces.first;

                      if (neigh_polytope->is_locally_owned())
                        {
                          // use both fevalues
                          const auto &fe_faces1 = fe_faces.second;

                          M11 = 0.;
                          M12 = 0.;
                          M21 = 0.;
                          M22 = 0.;

                          assemble_local_jumps_and_averages(M11,
                                                            M12,
                                                            M21,
                                                            M22,
                                                            fe_faces0,
                                                            fe_faces1,
                                                            penalty_constant,
                                                            h_f);

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
                          // neigh polytope is ghosted, so retrieve necessary
                          // metadata.

                          types::subdomain_id neigh_rank =
                            neigh_polytope->subdomain_id();

                          const auto &recv_jxws =
                            ah->recv_jxws.at(neigh_rank)
                              .at({neigh_polytope->id(), nofn});

                          const auto &recv_values =
                            ah->recv_values.at(neigh_rank)
                              .at({neigh_polytope->id(), nofn});

                          const auto &recv_gradients =
                            ah->recv_gradients.at(neigh_rank)
                              .at({neigh_polytope->id(), nofn});

                          M11 = 0.;
                          M12 = 0.;
                          M21 = 0.;
                          M22 = 0.;

                          // there's no FEFaceValues on the other side (it's
                          // ghosted), so we just pass the actual data we have
                          // recevied from the neighboring ghosted polytope
                          assemble_local_jumps_and_averages_ghost(
                            M11,
                            M12,
                            M21,
                            M22,
                            fe_faces0,
                            recv_values,
                            recv_gradients,
                            recv_jxws,
                            penalty_constant,
                            h_f);


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
  stop_assembly = MPI_Wtime();
  pcout << "Assembled in: " << stop_assembly - start_assembly << "[s]."
        << std::endl;
}



template <int dim>
void
DiffusionReactionProblem<dim>::solve()
{
  locally_relevant_solution.reinit(locally_owned_dofs,
                                   locally_relevant_dofs,
                                   comm);

  pcout << "Start solver" << std::endl;
  SolverControl                                     control(10000, 1e-12);
  SolverCG<TrilinosWrappers::MPI::Vector>           solver(control);
  TrilinosWrappers::PreconditionAMG                 precondition;
  TrilinosWrappers::PreconditionAMG::AdditionalData additional_data;

  if (fe_dg.degree > 1)
    additional_data.higher_order_elements = true;
  precondition.initialize(system_matrix, additional_data);

  TrilinosWrappers::MPI::Vector completely_distributed_solution(system_rhs);
  double                        start_solver, stop_solver;
  start_solver = MPI_Wtime();
  solver.solve(system_matrix,
               completely_distributed_solution,
               system_rhs,
               precondition);
  stop_solver = MPI_Wtime();

  pcout << "Linear system solved in: " << stop_solver - start_solver << "[s]."
        << std::endl;
  pcout << "Number of outer iterations: " << control.last_step() << std::endl;

  constraints.distribute(completely_distributed_solution);
  locally_relevant_solution = completely_distributed_solution;

  PolyUtils::interpolate_to_fine_grid(*ah,
                                      interpolated_solution,
                                      completely_distributed_solution);

  // Use other method to compute the error
#ifdef FALSE
  Vector<double> cellwise_error(completely_distributed_solution.size());
  VectorTools::integrate_difference(ah->output_dh,
                                    interpolated_solution,
                                    Solution<dim>(),
                                    cellwise_error,
                                    QGauss<dim>(2 * fe_dg.degree + 1),
                                    VectorTools::NormType::L2_norm);
  const double error =
    VectorTools::compute_global_error(tria_pft,
                                      cellwise_error,
                                      VectorTools::NormType::L2_norm);

  VectorTools::integrate_difference(ah->output_dh,
                                    interpolated_solution,
                                    Solution<dim>(),
                                    cellwise_error,
                                    QGauss<dim>(2 * fe_dg.degree + 1),
                                    VectorTools::NormType::H1_seminorm);
  const double semiH1error =
    VectorTools::compute_global_error(tria_pft,
                                      cellwise_error,
                                      VectorTools::NormType::H1_seminorm);
#endif

  std::vector<double> global_errors;
  PolyUtils::compute_global_error(*ah,
                                  completely_distributed_solution,
                                  Solution<dim>(),
                                  {VectorTools::L2_norm,
                                   VectorTools::H1_seminorm},
                                  global_errors);

  pcout << "L2 error (exponential solution): " << global_errors[0] << std::endl;
  pcout << "Semi H1 error (exponential solution): " << global_errors[1]
        << std::endl;
}



template <int dim>
void
DiffusionReactionProblem<dim>::output_results() const
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(ah->output_dh);

  data_out.add_data_vector(interpolated_solution,
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
    ("3D_diffusion_reaction." +
     Utilities::int_to_string(tria_pft.locally_owned_subdomain(), 4));

  std::ofstream output((filename + ".vtu").c_str());
  data_out.write_vtu(output);


  {
    std::vector<std::string> filenames;
    for (unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(comm); i++)
      {
        filenames.push_back("3D_diffusion_reaction." +
                            Utilities::int_to_string(i, 4) + ".vtu");
      }
    std::ofstream master_output("3D_diffusion_reaction.pvtu");
    data_out.write_pvtu_record(master_output, filenames);
  }
}



template <int dim>
void
DiffusionReactionProblem<dim>::run()
{
  make_fine_grid(3); // 3 global refinements of unit cube
  setup_agglomerated_problem(n_local_agglomerates);
  assemble_system();
  solve();
  output_results();
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  const MPI_Comm                   comm = MPI_COMM_WORLD;
  static constexpr unsigned int    dim  = 3;


  // number of agglomerates in each local subdomain
  const unsigned int n_local_agglomerates = 10;
  const double       reaction_coefficient = .5;
#ifdef HEX
  for (unsigned int degree : {1, 2, 3, 4})
#else
  for (unsigned int degree : {1, 2, 3}) // degree > 4 not implemented
#endif
    {
      DiffusionReactionProblem<dim> problem(n_local_agglomerates,
                                            degree,
                                            reaction_coefficient,
                                            comm);
      problem.run();
    }
}
