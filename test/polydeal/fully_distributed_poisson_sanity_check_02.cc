/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2022 by the polyDEAL authors
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


// Similar to fully_distributed_poisson_sanity_check_01.cc, but with a fully
// distributed triangulation created out of an external mesh.



#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <agglomeration_handler.h>
#include <poly_utils.h>


using namespace dealii;


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
class RightHandSide : public Function<dim>
{
public:
  RightHandSide()
    : Function<dim>()
  {}

  virtual void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<double>           &values,
             const unsigned int /*component*/ = 0) const override;
};


template <int dim>
void
RightHandSide<dim>::value_list(const std::vector<Point<dim>> &points,
                               std::vector<double>           &values,
                               const unsigned int /*component*/) const
{
  for (unsigned int i = 0; i < values.size(); ++i)
    values[i] = 8. * numbers::PI * numbers::PI *
                std::sin(2. * numbers::PI * points[i][0]) *
                std::sin(2. * numbers::PI * points[i][1]);
}


int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  const MPI_Comm                  &comm = MPI_COMM_WORLD;
  unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);

  static constexpr int dim     = 2;
  const unsigned       n_ranks = Utilities::MPI::n_mpi_processes(comm);
  Assert(n_ranks == 3,
         ExcMessage("This test is meant to be run with 3 ranks only."));
  if (Utilities::MPI::this_mpi_process(comm) == 0)
    std::cout << "Running with " << n_ranks << " MPI ranks." << std::endl;

  AffineConstraints<double> constraints;
  constraints.close();

  TrilinosWrappers::SparseMatrix system_matrix;

  Triangulation<dim> tria_base;

  // Create a serial triangulation (by reading an external mesh):
  GridIn<dim>   grid_in;
  std::ifstream gmsh_file(SOURCE_DIR "/input_grids/square.msh");
  grid_in.attach_triangulation(tria_base);
  grid_in.read_msh(gmsh_file);
  tria_base.refine_global(1);

  // Partition serial triangulation:
  GridTools::partition_triangulation(n_ranks, tria_base);

  // Create building blocks:
  const TriangulationDescription::Description<dim, dim> description =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      tria_base, comm);

  // Create a fully distributed triangulation:
  parallel::fullydistributed::Triangulation<dim> tria_pft(comm);
  tria_pft.create_triangulation(description);

  if (Utilities::MPI::this_mpi_process(comm) == 0)
    std::cout << "Number of cells: " << tria_pft.n_global_active_cells()
              << std::endl;

  const unsigned int n_local_agglomerates =
    10; // number of agglomerates in each local subdomain

  // Call the METIS partitioner to agglomerate within each processor.
  PolyUtils::partition_locally_owned_regions(n_local_agglomerates,
                                             tria_pft,
                                             SparsityTools::Partitioner::metis);


  // The rest of the program is identical to
  // fully_distributed_poisson_sanity_check_01.cc
  /*------------------------*/

  // Setup the agglomeration handler.
  GridTools::Cache<dim>     cached_tria(tria_pft);
  AgglomerationHandler<dim> ah(cached_tria);

  // Agglomerate cells together based on their material id
  std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>>
    cells_per_subdomain(n_local_agglomerates);
  for (const auto &cell : tria_pft.active_cell_iterators())
    if (cell->is_locally_owned())
      cells_per_subdomain[cell->material_id()].push_back(cell);

  // Agglomerate elements with same id
  for (std::size_t i = 0; i < cells_per_subdomain.size(); ++i)
    ah.define_agglomerate(cells_per_subdomain[i]);


  FE_DGQ<2> fe_dg(1);

  const unsigned int quadrature_degree      = 2 * fe_dg.get_degree() + 1;
  const unsigned int face_quadrature_degree = 2 * fe_dg.get_degree() + 1;
  ah.initialize_fe_values(QGauss<2>(quadrature_degree),
                          update_values | update_gradients | update_JxW_values |
                            update_quadrature_points,
                          QGauss<1>(face_quadrature_degree),
                          update_JxW_values);

  ah.distribute_agglomerated_dofs(
    fe_dg); // setup_ghost_polytopes has been called here

  TrilinosWrappers::SparsityPattern dsp;
  ah.create_agglomeration_sparsity_pattern(dsp);

  system_matrix.reinit(dsp);
  std::ofstream out("sparsity_agglomeration_from_rank_" +
                    std::to_string(Utilities::MPI::this_mpi_process(comm)) +
                    ".svg");

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


  auto polytope = ah.begin();
  for (; polytope != ah.end(); ++polytope)
    {
      if (polytope->is_locally_owned())
        {
          cell_matrix = 0.;

          const auto &agglo_values = ah.reinit(polytope);

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
                }
            }

          // distribute volumetric DoFs
          polytope->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_matrix,
                                                 local_dof_indices,
                                                 system_matrix);

          // Face terms
          unsigned int n_faces = polytope->n_faces();

          const double dummy_hf      = 1.;
          const double dummy_penalty = 1.;
          for (unsigned int f = 0; f < n_faces; ++f)
            {
              if (polytope->at_boundary(f))
                {
                  //  Do nothing
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
                        }
                      else
                        {
                          // neigh polytope is ghosted, assemble really by hand
                          // retrieving values and qpoints

                          types::subdomain_id neigh_rank =
                            neigh_polytope->subdomain_id();

                          const auto &recv_jxws =
                            ah.recv_jxws.at(neigh_rank)
                              .at({neigh_polytope->id(), nofn});

                          const auto &recv_values =
                            ah.recv_values.at(neigh_rank)
                              .at({neigh_polytope->id(), nofn});

                          const auto &recv_gradients =
                            ah.recv_gradients.at(neigh_rank)
                              .at({neigh_polytope->id(), nofn});

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
                                        recv_jxws[q_index];

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
                                        recv_jxws[q_index];

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
                                        recv_jxws[q_index];
                                    }
                                }
                            }
                        } // ghosted polytope case

                      // distribute DoFs accordingly
                      neigh_polytope->get_dof_indices(
                        local_dof_indices_neighbor);

                      constraints.distribute_local_to_global(M11,
                                                             local_dof_indices,
                                                             system_matrix);
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


                    } // only once
                      // not on boundary
                }     // every face
            }
        } // locally owned
    }

  system_matrix.compress(VectorOperation::add);



  // Once matrix is assembled, perform the sanity checks
  {
    TrilinosWrappers::MPI::Vector interpx(
      system_matrix.locally_owned_domain_indices()); // f(x,y)=x
    TrilinosWrappers::MPI::Vector interpxplusy(
      system_matrix.locally_owned_domain_indices()); // f(x,y)=x+y
    LinearFunction<2> xfunction{{1, 0}};
    LinearFunction<2> xplusyfunction{{1, 1}};

    VectorTools::interpolate(*(ah.euler_mapping),
                             ah.get_dof_handler(),
                             xfunction,
                             interpx);

    VectorTools::interpolate(*(ah.euler_mapping),
                             ah.get_dof_handler(),
                             xplusyfunction,
                             interpxplusy);


    const double valuex = system_matrix.matrix_scalar_product(interpx, interpx);
    if (my_rank == 0)
      std::cout << "Test with f(x,y)=x: " << valuex << std::endl;

    const double valuexpy =
      system_matrix.matrix_scalar_product(interpxplusy, interpxplusy);
    if (my_rank == 0)
      std::cout << "Test with f(x,y)=x+y: " << valuexpy << std::endl;
  }
}