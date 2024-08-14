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


// Similar to poisson_sanity_check_01.cc, but in the distributed setting.



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

  GridTools::Cache<2>     cached_tria(tria);
  AgglomerationHandler<2> ah(cached_tria);

  /*
  {
    DataOut<2> data_out;
    data_out.attach_dof_handler(ah.agglo_dh);

    Vector<float> subdomain(tria.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      {
        subdomain(i) = tria.locally_owned_subdomain();
      }

    Vector<float> local_cell_index(tria.n_active_cells());
    unsigned int  i = 0;
    for (const auto &cell : tria.active_cell_iterators())
      {
        local_cell_index(i) = cell->active_cell_index();
        ++i;
      }

    data_out.add_data_vector(subdomain, "subdomain");
    data_out.add_data_vector(local_cell_index, "cell_index");

    data_out.build_patches();

    data_out.write_vtu_in_parallel("MPI_sandbox_2.vtu", comm);
  }
  */

  unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);

  const auto &get_all_locally_owned_indices = [&tria]() {
    std::vector<types::global_cell_index> local_indices;
    for (const auto &cell : tria.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          local_indices.push_back(cell->active_cell_index());
      }
    return local_indices;
  };


  if (my_rank == 0)
    {
      std::vector<types::global_cell_index> idxs_to_be_agglomerated0 = {5,
                                                                        6,
                                                                        7,
                                                                        8};
      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated0;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated0,
                                                 cells_to_be_agglomerated0);
      ah.define_agglomerate(cells_to_be_agglomerated0);

      std::vector<types::global_cell_index> idxs_to_be_agglomerated1 = {9,
                                                                        10,
                                                                        11,
                                                                        12};
      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated1;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated1,
                                                 cells_to_be_agglomerated1);
      ah.define_agglomerate(cells_to_be_agglomerated1);

      std::vector<types::global_cell_index> idxs_to_be_agglomerated2 = {21,
                                                                        22,
                                                                        23,
                                                                        24};
      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated2;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated2,
                                                 cells_to_be_agglomerated2);
      ah.define_agglomerate(cells_to_be_agglomerated2);

      std::vector<types::global_cell_index> idxs_to_be_agglomerated3 = {13,
                                                                        14,
                                                                        15,
                                                                        16};
      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated3;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated3,
                                                 cells_to_be_agglomerated3);
      ah.define_agglomerate(cells_to_be_agglomerated3);

      std::vector<types::global_cell_index> idxs_to_be_agglomerated4 = {17,
                                                                        18,
                                                                        19,
                                                                        20};
      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated4;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated4,
                                                 cells_to_be_agglomerated4);
      ah.define_agglomerate(cells_to_be_agglomerated4);
    }
  else if (my_rank == 1)
    {
      std::vector<types::global_cell_index> idxs_to_be_agglomerated0 = {18,
                                                                        19,
                                                                        20,
                                                                        21};
      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated0;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated0,
                                                 cells_to_be_agglomerated0);
      ah.define_agglomerate(cells_to_be_agglomerated0);

      std::vector<types::global_cell_index> idxs_to_be_agglomerated1 = {22,
                                                                        23,
                                                                        24,
                                                                        25};
      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated1;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated1,
                                                 cells_to_be_agglomerated1);
      ah.define_agglomerate(cells_to_be_agglomerated1);

      std::vector<types::global_cell_index> idxs_to_be_agglomerated2 = {26,
                                                                        27,
                                                                        28,
                                                                        29};
      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated2;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated2,
                                                 cells_to_be_agglomerated2);
      ah.define_agglomerate(cells_to_be_agglomerated2);

      std::vector<types::global_cell_index> idxs_to_be_agglomerated3 = {30,
                                                                        31,
                                                                        32,
                                                                        33};
      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated3;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated3,
                                                 cells_to_be_agglomerated3);
      ah.define_agglomerate(cells_to_be_agglomerated3);

      std::vector<types::global_cell_index> idxs_to_be_agglomerated4 = {34,
                                                                        35,
                                                                        36,
                                                                        37};
      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated4;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated4,
                                                 cells_to_be_agglomerated4);
      ah.define_agglomerate(cells_to_be_agglomerated4);

      std::vector<types::global_cell_index> idxs_to_be_agglomerated5 = {38,
                                                                        39,
                                                                        40,
                                                                        41};
      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated5;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated5,
                                                 cells_to_be_agglomerated5);
      ah.define_agglomerate(cells_to_be_agglomerated5);
    }
  else
    {
      const auto &idxs_to_be_agglomerated = get_all_locally_owned_indices();

      std::vector<typename Triangulation<2>::active_cell_iterator>
        cells_to_be_agglomerated;
      PolyUtils::collect_cells_for_agglomeration(tria,
                                                 idxs_to_be_agglomerated,
                                                 cells_to_be_agglomerated);
      ah.define_agglomerate(cells_to_be_agglomerated);
    }



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
