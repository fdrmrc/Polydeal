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


#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <agglomeration_handler.h>
#include <poly_utils.h>

#include <algorithm>

constexpr double entry_tol = 1e-14;

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
class MyFunction : public Function<dim>
{
public:
  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;
};

template <int dim>
double
MyFunction<dim>::value(const Point<dim> &p, const unsigned int) const
{
  return p[0];
}



template <int dim>
class Poisson
{
private:
  void
  make_grid();
  void
  setup_agglomeration();
  void
  assemble_system();
  void
  solve();
  void
  output_results();

  bool                                       to_agglomerate;
  Triangulation<dim>                         tria;
  MappingQ<dim>                              mapping;
  FE_DGQ<dim>                                dg_fe;
  std::unique_ptr<AgglomerationHandler<dim>> ah;
  // no hanging node in DG discretization, we define an AffineConstraints object
  // so we can use the distribute_local_to_global() directly.
  AffineConstraints<double>              constraints;
  SparsityPattern                        sparsity;
  DynamicSparsityPattern                 dsp;
  SparseMatrix<double>                   system_matrix;
  Vector<double>                         solution;
  Vector<double>                         system_rhs;
  std::unique_ptr<GridTools::Cache<dim>> cached_tria;
  std::unique_ptr<const Function<dim>>   rhs_function;

public:
  constexpr Poisson(const bool = true);
  void
  run();
  inline const SparseMatrix<double> &
  get_matrix()
  {
    return system_matrix;
  }

  double penalty = 20.;
};



template <int dim>
constexpr Poisson<dim>::Poisson(const bool agglomerated_or_not)
  : to_agglomerate(agglomerated_or_not)
  , mapping(1)
  , dg_fe(1)
{}

template <int dim>
void
Poisson<dim>::make_grid()
{
  GridGenerator::hyper_cube(tria, -1, 1);
  if constexpr (dim == 2)
    {
      if (to_agglomerate)
        {
          tria.refine_global(3);
          std::cout << "N elements " << tria.n_active_cells() << std::endl;
        }
      else
        {
          tria.refine_global(2);
          std::cout << "N elements " << tria.n_active_cells() << std::endl;
        }
    }
  else if constexpr (dim == 3)
    {
      if (to_agglomerate)
        {
          tria.refine_global(1);
        }
    }
  std::cout << "Number of cells: " << tria.n_active_cells() << std::endl;
  std::ofstream out("test_grid.vtk");
  GridOut       grid_out;
  grid_out.write_vtk(tria, out);
  std::cout << "Grid written " << std::endl;

  cached_tria  = std::make_unique<GridTools::Cache<dim>>(tria, mapping);
  rhs_function = std::make_unique<const RightHandSide<dim>>();

  constraints.close();
}



template <int dim>
void
Poisson<dim>::setup_agglomeration()

{
  ah = std::make_unique<AgglomerationHandler<dim>>(*cached_tria);

  if (to_agglomerate)
    {
      if constexpr (dim == 2)
        {
          std::vector<types::global_cell_index> idxs_to_be_agglomerated = {0,
                                                                           1,
                                                                           2,
                                                                           3};

          std::vector<typename Triangulation<dim>::active_cell_iterator>
            cells_to_be_agglomerated;
          PolyUtils::collect_cells_for_agglomeration(tria,
                                                     idxs_to_be_agglomerated,
                                                     cells_to_be_agglomerated);


          std::vector<types::global_cell_index> idxs_to_be_agglomerated2 = {4,
                                                                            5,
                                                                            6,
                                                                            7};

          std::vector<typename Triangulation<dim>::active_cell_iterator>
            cells_to_be_agglomerated2;
          PolyUtils::collect_cells_for_agglomeration(tria,
                                                     idxs_to_be_agglomerated2,
                                                     cells_to_be_agglomerated2);


          std::vector<types::global_cell_index> idxs_to_be_agglomerated3 = {8,
                                                                            9,
                                                                            10,
                                                                            11};
          std::vector<typename Triangulation<dim>::active_cell_iterator>
            cells_to_be_agglomerated3;
          PolyUtils::collect_cells_for_agglomeration(tria,
                                                     idxs_to_be_agglomerated3,
                                                     cells_to_be_agglomerated3);


          std::vector<types::global_cell_index> idxs_to_be_agglomerated4 = {
            12, 13, 14, 15}; //{36,37}
          std::vector<typename Triangulation<dim>::active_cell_iterator>
            cells_to_be_agglomerated4;
          PolyUtils::collect_cells_for_agglomeration(tria,
                                                     idxs_to_be_agglomerated4,
                                                     cells_to_be_agglomerated4);

          std::vector<types::global_cell_index> idxs_to_be_agglomerated5 = {
            16, 17, 18, 19}; //{36,37}
          std::vector<typename Triangulation<dim>::active_cell_iterator>
            cells_to_be_agglomerated5;
          PolyUtils::collect_cells_for_agglomeration(tria,
                                                     idxs_to_be_agglomerated5,
                                                     cells_to_be_agglomerated5);

          std::vector<types::global_cell_index> idxs_to_be_agglomerated6 = {
            20, 21, 22, 23}; //{36,37}
          std::vector<typename Triangulation<dim>::active_cell_iterator>
            cells_to_be_agglomerated6;
          PolyUtils::collect_cells_for_agglomeration(tria,
                                                     idxs_to_be_agglomerated6,
                                                     cells_to_be_agglomerated6);

          std::vector<types::global_cell_index> idxs_to_be_agglomerated7 = {
            24, 25, 26, 27}; //{36,37}
          std::vector<typename Triangulation<dim>::active_cell_iterator>
            cells_to_be_agglomerated7;
          PolyUtils::collect_cells_for_agglomeration(tria,
                                                     idxs_to_be_agglomerated7,
                                                     cells_to_be_agglomerated7);

          std::vector<types::global_cell_index> idxs_to_be_agglomerated8 = {
            28, 29, 30, 31}; //{36,37}
          std::vector<typename Triangulation<dim>::active_cell_iterator>
            cells_to_be_agglomerated8;
          PolyUtils::collect_cells_for_agglomeration(tria,
                                                     idxs_to_be_agglomerated8,
                                                     cells_to_be_agglomerated8);

          std::vector<types::global_cell_index> idxs_to_be_agglomerated9 = {
            32, 33, 34, 35}; //{36,37}
          std::vector<typename Triangulation<dim>::active_cell_iterator>
            cells_to_be_agglomerated9;
          PolyUtils::collect_cells_for_agglomeration(tria,
                                                     idxs_to_be_agglomerated9,
                                                     cells_to_be_agglomerated9);

          std::vector<types::global_cell_index> idxs_to_be_agglomerated10 = {
            36, 37, 38, 39}; //{36,37}
          std::vector<typename Triangulation<dim>::active_cell_iterator>
            cells_to_be_agglomerated10;
          PolyUtils::collect_cells_for_agglomeration(
            tria, idxs_to_be_agglomerated10, cells_to_be_agglomerated10);

          std::vector<types::global_cell_index> idxs_to_be_agglomerated11 = {
            40, 41, 42, 43}; //{36,37}
          std::vector<typename Triangulation<dim>::active_cell_iterator>
            cells_to_be_agglomerated11;
          PolyUtils::collect_cells_for_agglomeration(
            tria, idxs_to_be_agglomerated11, cells_to_be_agglomerated11);

          std::vector<types::global_cell_index> idxs_to_be_agglomerated12 = {
            44, 45, 46, 47}; //{36,37}
          std::vector<typename Triangulation<dim>::active_cell_iterator>
            cells_to_be_agglomerated12;
          PolyUtils::collect_cells_for_agglomeration(
            tria, idxs_to_be_agglomerated12, cells_to_be_agglomerated12);

          std::vector<types::global_cell_index> idxs_to_be_agglomerated13 = {
            48, 49, 50, 51}; //{36,37}
          std::vector<typename Triangulation<dim>::active_cell_iterator>
            cells_to_be_agglomerated13;
          PolyUtils::collect_cells_for_agglomeration(
            tria, idxs_to_be_agglomerated13, cells_to_be_agglomerated13);

          std::vector<types::global_cell_index> idxs_to_be_agglomerated14 = {
            52, 53, 54, 55}; //{36,37}
          std::vector<typename Triangulation<dim>::active_cell_iterator>
            cells_to_be_agglomerated14;
          PolyUtils::collect_cells_for_agglomeration(
            tria, idxs_to_be_agglomerated14, cells_to_be_agglomerated14);

          std::vector<types::global_cell_index> idxs_to_be_agglomerated15 = {
            56, 57, 58, 59}; //{36,37}
          std::vector<typename Triangulation<dim>::active_cell_iterator>
            cells_to_be_agglomerated15;
          PolyUtils::collect_cells_for_agglomeration(
            tria, idxs_to_be_agglomerated15, cells_to_be_agglomerated15);

          std::vector<types::global_cell_index> idxs_to_be_agglomerated16 = {
            60, 61, 62, 63}; //{36,37}
          std::vector<typename Triangulation<dim>::active_cell_iterator>
            cells_to_be_agglomerated16;
          PolyUtils::collect_cells_for_agglomeration(
            tria, idxs_to_be_agglomerated16, cells_to_be_agglomerated16);



          // Agglomerate the cells just stored
          ah->define_agglomerate(cells_to_be_agglomerated);
          ah->define_agglomerate(cells_to_be_agglomerated2);
          ah->define_agglomerate(cells_to_be_agglomerated3);
          ah->define_agglomerate(cells_to_be_agglomerated4);
          ah->define_agglomerate(cells_to_be_agglomerated5);
          ah->define_agglomerate(cells_to_be_agglomerated6);
          ah->define_agglomerate(cells_to_be_agglomerated7);
          ah->define_agglomerate(cells_to_be_agglomerated8);
          ah->define_agglomerate(cells_to_be_agglomerated9);
          ah->define_agglomerate(cells_to_be_agglomerated10);
          ah->define_agglomerate(cells_to_be_agglomerated11);
          ah->define_agglomerate(cells_to_be_agglomerated12);
          ah->define_agglomerate(cells_to_be_agglomerated13);
          ah->define_agglomerate(cells_to_be_agglomerated14);
          ah->define_agglomerate(cells_to_be_agglomerated15);
          ah->define_agglomerate(cells_to_be_agglomerated16);
        }
      else if constexpr (dim == 3)
        {
          std::vector<types::global_cell_index> idxs_to_be_agglomerated = {
            0, 1, 2, 3, 4, 5, 6, 7};

          std::vector<typename Triangulation<dim>::active_cell_iterator>
            cells_to_be_agglomerated;
          PolyUtils::collect_cells_for_agglomeration(tria,
                                                     idxs_to_be_agglomerated,
                                                     cells_to_be_agglomerated);

          // Agglomerate the cells just stored
          ah->define_agglomerate(cells_to_be_agglomerated);
        }
    }
  ah->distribute_agglomerated_dofs(dg_fe);
  ah->create_agglomeration_sparsity_pattern(dsp);
  sparsity.copy_from(dsp);
}



template <int dim>
void
Poisson<dim>::assemble_system()
{
  system_matrix.reinit(sparsity);
  solution.reinit(ah->n_dofs());
  system_rhs.reinit(ah->n_dofs());

  const unsigned int quadrature_degree      = 2 * dg_fe.get_degree() + 1;
  const unsigned int face_quadrature_degree = 2 * dg_fe.get_degree() + 1;
  ah->initialize_fe_values(QGauss<dim>(quadrature_degree),
                           update_gradients | update_JxW_values |
                             update_quadrature_points | update_JxW_values |
                             update_values,
                           QGauss<dim - 1>(face_quadrature_degree));

  const unsigned int dofs_per_cell = ah->n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  // Next, we define the four dofsxdofs matrices needed to assemble jumps and
  // averages.
  FullMatrix<double> M11(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> M12(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> M21(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> M22(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices_bdary_cell(
    dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices_neighbor(
    dofs_per_cell);

  for (const auto &polytope : ah->polytope_iterators())
    {
      std::cout << "Polytope with idx: " << polytope->index() << std::endl;
      cell_matrix              = 0;
      cell_rhs                 = 0;
      const auto &agglo_values = ah->reinit(polytope);

      const auto &        q_points  = agglo_values.get_quadrature_points();
      const unsigned int  n_qpoints = q_points.size();
      std::vector<double> rhs(n_qpoints);
      rhs_function->value_list(q_points, rhs);

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
        }

      polytope->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);

      // Face terms
      const unsigned int n_faces = polytope->n_faces();
      std::cout << "Number of (generalized) faces: " << n_faces << std::endl;

      for (unsigned int f = 0; f < n_faces; ++f)
        {
          const double hf = 1.; // cell->face(0)->measure();
#ifdef AGGLO_DEBUG
          std::cout << "Face measure: " << hf << std::endl;
#endif
          if (polytope->at_boundary(f))
            {
              std::cout << "at boundary!" << std::endl;
              const auto &fe_face = ah->reinit(polytope, f);

              {
                std::cout << "I qpoints sul boundary sono: " << std::endl;
                for (const auto &q : fe_face.get_quadrature_points())
                  std::cout << q << std::endl;
              }
              const unsigned int dofs_per_cell = fe_face.dofs_per_cell;
              std::cout << "With dofs_per_cell =" << fe_face.dofs_per_cell
                        << std::endl;


              // Get normal vectors seen from each agglomeration.
              const auto &normals = fe_face.get_normal_vectors();
              cell_matrix         = 0.;
              for (unsigned int q_index : fe_face.quadrature_point_indices())
                {
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                          cell_matrix(i, j) +=
                            0. *
                            (-fe_face.shape_value(i, q_index) *
                               fe_face.shape_grad(j, q_index) *
                               normals[q_index] -
                             fe_face.shape_grad(i, q_index) * normals[q_index] *
                               fe_face.shape_value(j, q_index) +
                             (penalty / hf) * fe_face.shape_value(i, q_index) *
                               fe_face.shape_value(j, q_index)) *
                            fe_face.JxW(q_index);
                        }
                    }
                }

              // distribute DoFs
              polytope->get_dof_indices(local_dof_indices_bdary_cell);
              system_matrix.add(local_dof_indices_bdary_cell, cell_matrix);
            }
          else
            {
              const auto &neigh_polytope = polytope->neighbor(f);
#ifdef AGGLO_DEBUG
              std::cout << "Neighbor is " << neigh_polytope->index()
                        << std::endl;
#endif

              // This is necessary to loop over internal faces only once.
              if (polytope->index() < neigh_polytope->index())
                {
                  unsigned int nofn =
                    polytope->neighbor_of_agglomerated_neighbor(f);
#ifdef AGGLO_DEBUG
                  std::cout << "Neighbor of neighbor is:" << nofn << std::endl;
#endif

                  Assert(neigh_polytope->neighbor(nofn)->index() ==
                           polytope->index(),
                         ExcMessage("Impossible."));

                  const auto &fe_faces =
                    ah->reinit_interface(polytope, neigh_polytope, f, nofn);
#ifdef AGGLO_DEBUG
                  std::cout << "Reinited the interface:" << nofn << std::endl;
#endif

                  const auto &fe_faces0 = fe_faces.first;
                  const auto &fe_faces1 = fe_faces.second;
#ifdef AGGLO_DEBUG
                  std::cout << "Local from current: " << f << std::endl;
                  std::cout << "Local from neighbor: " << nofn << std::endl;

                  std::cout << "Jump between " << polytope->index() << " and "
                            << neigh_polytope->index() << std::endl;
                  {
                    std::cout << "Qpoints from first polytope: " << std::endl;
                    for (const auto &q : fe_faces0.get_quadrature_points())
                      std::cout << q << std::endl;
                    std::cout << "Qpoints from secondo polytope: " << std::endl;
                    for (const auto &q : fe_faces1.get_quadrature_points())
                      std::cout << q << std::endl;
                  }
#endif

                  M11 = 0.;
                  M12 = 0.;
                  M21 = 0.;
                  M22 = 0.;

                  const auto &normals = fe_faces0.get_normal_vectors();
                  // M11
                  for (unsigned int q_index :
                       fe_faces0.quadrature_point_indices())
                    {
#ifdef AGGLO_DEBUG
                      std::cout << normals[q_index] << std::endl;
#endif
                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                              M11(i, j) +=
                                (-0.5 * fe_faces0.shape_grad(i, q_index) *
                                   normals[q_index] *
                                   fe_faces0.shape_value(j, q_index) -
                                 0.5 * fe_faces0.shape_grad(j, q_index) *
                                   normals[q_index] *
                                   fe_faces0.shape_value(i, q_index) +
                                 (penalty / hf) *
                                   fe_faces0.shape_value(i, q_index) *
                                   fe_faces0.shape_value(j, q_index)) *
                                fe_faces0.JxW(q_index);

                              M12(i, j) +=
                                (0.5 * fe_faces0.shape_grad(i, q_index) *
                                   normals[q_index] *
                                   fe_faces1.shape_value(j, q_index) -
                                 0.5 * fe_faces1.shape_grad(j, q_index) *
                                   normals[q_index] *
                                   fe_faces0.shape_value(i, q_index) -
                                 (penalty / hf) *
                                   fe_faces0.shape_value(i, q_index) *
                                   fe_faces1.shape_value(j, q_index)) *
                                fe_faces1.JxW(q_index);

                              // A10
                              M21(i, j) +=
                                (-0.5 * fe_faces1.shape_grad(i, q_index) *
                                   normals[q_index] *
                                   fe_faces0.shape_value(j, q_index) +
                                 0.5 * fe_faces0.shape_grad(j, q_index) *
                                   normals[q_index] *
                                   fe_faces1.shape_value(i, q_index) -
                                 (penalty / hf) *
                                   fe_faces1.shape_value(i, q_index) *
                                   fe_faces0.shape_value(j, q_index)) *
                                fe_faces1.JxW(q_index);

                              // A11
                              M22(i, j) +=
                                (0.5 * fe_faces1.shape_grad(i, q_index) *
                                   normals[q_index] *
                                   fe_faces1.shape_value(j, q_index) +
                                 0.5 * fe_faces1.shape_grad(j, q_index) *
                                   normals[q_index] *
                                   fe_faces1.shape_value(i, q_index) +
                                 (penalty / hf) *
                                   fe_faces1.shape_value(i, q_index) *
                                   fe_faces1.shape_value(j, q_index)) *
                                fe_faces1.JxW(q_index);
                            }
                        }
                    }

#ifdef AGGLO_DEBUG
                  std::cout << "Neighbor is " << neigh_polytope->index()
                            << std::endl;
#endif
                  // distribute DoFs accordingly
                  neigh_polytope->get_dof_indices(local_dof_indices_neighbor);

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
                } // Loop only once trough internal faces
            }
        } // Loop over faces of current cell
    }     // Loop over cells
}


template <int dim>
void
Poisson<dim>::solve()
{
  SparseDirectUMFPACK A_direct;
  {
    {
      std::cout << "Sanity check:" << std::endl;

      // Sanity check: v(x,y)=x
      Vector<double>  interpx(ah->get_dof_handler().n_dofs());
      MyFunction<dim> xfunction{};
      VectorTools::interpolate(*(ah->euler_mapping),
                               ah->get_dof_handler(),
                               xfunction,
                               interpx);

      double value = system_matrix.matrix_scalar_product(interpx, interpx);
      std::cout << "Test = " << value << std::endl;
    }

    if (to_agglomerate)
      {
        std::ofstream output_file("agglo_SIP_matrix.txt");
        system_matrix.print_formatted(output_file);
      }
    else
      {
        std::ofstream output_file("std_SIP_matrix.txt");
        system_matrix.print_formatted(output_file);
      }
  }
  A_direct.initialize(system_matrix);
  A_direct.vmult(solution, system_rhs);
}



template <int dim>
void
Poisson<dim>::output_results()
{
  {
    const std::string filename = "agglomerated_Poisson.vtu";
    std::ofstream     output(filename);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(ah->get_dof_handler());
    data_out.add_data_vector(solution, "u", DataOut<dim>::type_dof_data);
    data_out.build_patches(mapping);
    data_out.write_vtu(output);
  }
  {
    const std::string filename = "agglomerated_Poisson_test.vtu";
    std::ofstream     output(filename);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(ah->get_dof_handler());
    data_out.add_data_vector(solution, "u", DataOut<dim>::type_dof_data);
    data_out.build_patches(*(ah->euler_mapping));
    data_out.write_vtu(output);
  }
  {
    const std::string filename = "interpolated_solution.vtu";
    std::ofstream     output(filename);

    // Setup interpolation onto classical underlying DoFHandler
    Vector<double> interpolated_solution;
    PolyUtils::interpolate_to_fine_grid(*ah, interpolated_solution, solution);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(ah->output_dh);
    data_out.add_data_vector(interpolated_solution,
                             "u",
                             DataOut<dim>::type_dof_data);
    data_out.build_patches(mapping);
    data_out.write_vtu(output);
  }
  {
    const std::string filename = "interpolated_solution.vtu";
    std::ofstream     output(filename);

    DataOut<dim>   data_out;
    Vector<double> interpolated_solution;
    PolyUtils::interpolate_to_fine_grid(*ah, interpolated_solution, solution);
    data_out.attach_dof_handler(ah->output_dh);
    data_out.add_data_vector(interpolated_solution,
                             "u",
                             DataOut<dim>::type_dof_data);
    data_out.build_patches(mapping);
    data_out.write_vtu(output);
  }

  // {
  //   // Sanity check: v(x,y)=x
  //   Vector<double>  interpx(ah->agglo_dh.n_dofs());
  //   MyFunction<dim> xfunction{};
  //   VectorTools::interpolate(mapping, ah->agglo_dh, xfunction, interpx);

  //   double value = system_matrix.matrix_scalar_product(interpx, interpx);
  //   std::cout << "Test = " << value << std::endl;
  // }
}

template <int dim>
void
Poisson<dim>::run()
{
  make_grid();
  setup_agglomeration();
  assemble_system();
  solve();
  output_results();
}

int
main(int argc, char *argv[])
{
  constexpr unsigned int dim = 2;
  if (argc == 1)
    {
      std::cout << "Running SIP with agglomerations." << std::endl;
      Poisson<dim> poisson_problem;
      poisson_problem.run();
    }
  else if (argc == 2)
    {
      if (std::strcmp(argv[1], "agglo") == 0)
        {
          std::cout << "Running SIP with 4 agglomerated cells." << std::endl;
          Poisson<dim> poisson_problem(true); // agglomerate
          poisson_problem.run();
        }
      else if (std::strcmp(argv[1], "standard") == 0)
        {
          std::cout << "Running SIP with standard grid." << std::endl;
          Poisson<dim> poisson_problem(false); // standard SIPDG
          poisson_problem.run();
        }
      else if (std::strcmp(argv[1], "test_equality") == 0)
        {
          std::cout << "Running the two techniques and testing for equality"
                    << std::endl;
          Poisson<dim> standard_problem(false);
          standard_problem.run();
          const auto & standard_matrix = standard_problem.get_matrix();
          Poisson<dim> agglo_problem(true);
          agglo_problem.run();
          const auto &agglo_matrix = agglo_problem.get_matrix();

          std::cout
            << "Comparing entries of the two matrices by subtracting them:"
            << std::endl;
          for (unsigned int i = 0; i < standard_matrix.m(); ++i)
            for (unsigned int j = 0; j < standard_matrix.n(); ++j)
              {
                std::cout << std::fabs(standard_matrix.el(i, j) -
                                       agglo_matrix.el(i, j))
                          << "\t";
                Assert(
                  std::fabs(standard_matrix.el(i, j) - agglo_matrix.el(i, j)) <
                    entry_tol,
                  ExcMessage(
                    "Matrices are not equivalent up to machine precision. Code is buggy."));
              }
          std::cout << std::endl;
        }
      else
        {
          Assert(
            false,
            ExcMessage(
              "Please choose between `agglo` and `standard` way of assembling the matrix. If you want to test if the two approcahes are equal, use `test_equality` as command line argument."));
        }
    }
  else
    {
      Assert(
        false,
        ExcMessage(
          "Invalid number of command line arguments, aborting the program."));
    }

  return 0;
}