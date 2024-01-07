#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/data_out.h>

#include <agglomeration_handler.h>

#include <algorithm>


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


  Triangulation<dim>                   tria;
  MappingQ<dim>                        mapping;
  FE_DGQ<dim>                          dg_fe;
  DoFHandler<dim>                      classical_dh;
  SparsityPattern                      sparsity;
  SparseMatrix<double>                 system_matrix;
  Vector<double>                       solution;
  Vector<double>                       system_rhs;
  std::unique_ptr<const Function<dim>> rhs_function;

public:
  Poisson();
  void
  run();

  double penalty = 100.;
};



template <int dim>
Poisson<dim>::Poisson()
  : mapping(1)
  , dg_fe(1)
  , classical_dh(tria)
{}

template <int dim>
void
Poisson<dim>::make_grid()
{
  GridGenerator::hyper_cube(tria, -1, 1);
  tria.refine_global(6);
  std::cout << "Number of faces for the tria: " << tria.n_active_faces()
            << std::endl;
  classical_dh.distribute_dofs(dg_fe);
  rhs_function = std::make_unique<const RightHandSide<dim>>();
}



template <int dim>
void
Poisson<dim>::assemble_system()
{
  DynamicSparsityPattern dsp(classical_dh.n_dofs());
  DoFTools::make_flux_sparsity_pattern(classical_dh, dsp);
  sparsity.copy_from(dsp);

  system_matrix.reinit(sparsity);
  solution.reinit(classical_dh.n_dofs());
  system_rhs.reinit(classical_dh.n_dofs());

  AffineConstraints<double> constraints;
  constraints.close();
  const unsigned int quadrature_degree = 3;
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
              cell_rhs(i) += fe_values.shape_value(i, q_index) * rhs[q_index] *
                             fe_values.JxW(q_index);
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
              for (unsigned int q_index : fe_face0.quadrature_point_indices())
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
                             (penalty / hf) * fe_face0.shape_value(i, q_index) *
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
              fe_face1.reinit(cell->neighbor(f), cell->neighbor_of_neighbor(f));

              std::vector<types::global_dof_index> local_dof_indices_neighbor(
                dofs_per_cell);

              FullMatrix<double> M11(dofs_per_cell, dofs_per_cell);
              FullMatrix<double> M12(dofs_per_cell, dofs_per_cell);
              FullMatrix<double> M21(dofs_per_cell, dofs_per_cell);
              FullMatrix<double> M22(dofs_per_cell, dofs_per_cell);

              const auto &normals = fe_face1.get_normal_vectors();
              // M11
              for (unsigned int q_index : fe_face0.quadrature_point_indices())
                {
                  std::cout << normals[q_index] << std::endl;
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
                             (penalty / hf) * fe_face0.shape_value(i, q_index) *
                               fe_face0.shape_value(j, q_index)) *
                            fe_face0.JxW(q_index);
                        }
                    }
                }
              // M12
              for (unsigned int q_index : fe_face0.quadrature_point_indices())
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
                             (penalty / hf) * fe_face0.shape_value(i, q_index) *
                               fe_face1.shape_value(j, q_index)) *
                            fe_face1.JxW(q_index);
                        }
                    }
                }
              // A10
              for (unsigned int q_index : fe_face0.quadrature_point_indices())
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
                             (penalty / hf) * fe_face1.shape_value(i, q_index) *
                               fe_face0.shape_value(j, q_index)) *
                            fe_face1.JxW(q_index);
                        }
                    }
                }
              // A11
              for (unsigned int q_index : fe_face0.quadrature_point_indices())
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
                             (penalty / hf) * fe_face1.shape_value(i, q_index) *
                               fe_face1.shape_value(j, q_index)) *
                            fe_face1.JxW(q_index);
                        }
                    }
                }

              // distribute DoFs accordingly
              std::cout << "Neighbor is " << cell->neighbor(f) << std::endl;
              cell->neighbor(f)->get_dof_indices(local_dof_indices_neighbor);

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

      constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }
}


template <int dim>
void
Poisson<dim>::solve()
{
  SparseDirectUMFPACK A_direct;
  A_direct.initialize(system_matrix);
  A_direct.vmult(solution, system_rhs);
}



template <int dim>
void
Poisson<dim>::output_results()
{
  const std::string filename = "agglomerated_Poisson_classic_DG.vtu";
  std::ofstream     output(filename);

  DataOut<dim> data_out;
  data_out.attach_dof_handler(classical_dh);
  data_out.add_data_vector(solution, "u", DataOut<dim>::type_dof_data);
  data_out.build_patches(mapping);
  data_out.write_vtu(output);
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
main()
{
  Poisson<2> poisson_problem;
  poisson_problem.run();
  return 0;
}
