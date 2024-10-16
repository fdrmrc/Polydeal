#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>

#include <agglomeration_handler.h>
#include <poly_utils.h>

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
             std::vector<double>           &values,
             const unsigned int /*component*/) const override;
};

template <int dim>
class Solution : public Function<dim>
{
public:
  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

  virtual Tensor<1, dim>
  gradient(const Point<dim>  &p,
           const unsigned int component = 0) const override;
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


template <int dim>
double
Solution<dim>::value(const Point<dim> &p, const unsigned int) const
{
  return std::sin(2. * numbers::PI * p[0]) * std::sin(2. * numbers::PI * p[1]);
}

template <int dim>
Tensor<1, dim>
Solution<dim>::gradient(const Point<dim> &p, const unsigned int) const
{
  Tensor<1, dim> return_value;
  Assert(false, ExcNotImplemented());
  return return_value;
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
  Poisson();
  void
  run();

  double penalty = 20.;
};



template <int dim>
Poisson<dim>::Poisson()
  : mapping(1)
  , dg_fe(1)
{}

template <int dim>
void
Poisson<dim>::make_grid()
{
  GridGenerator::hyper_cube(tria, -1, 1);
  tria.refine_global(6);
  cached_tria  = std::make_unique<GridTools::Cache<dim>>(tria, mapping);
  rhs_function = std::make_unique<const RightHandSide<dim>>();

  constraints.close();
}



template <int dim>
void
Poisson<dim>::setup_agglomeration()

{
  // std::vector<types::global_cell_index> idxs_to_be_agglomerated = {
  //   3, 9}; //{8, 9, 10, 11};
  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells; // each cell = an agglomerate
  for (const auto &cell : tria.active_cell_iterators())
    cells.push_back(cell);


  std::vector<types::global_cell_index> flagged_cells;
  const auto                            store_flagged_cells =
    [&flagged_cells](
      const std::vector<types::global_cell_index> &idxs_to_be_agglomerated) {
      for (const int idx : idxs_to_be_agglomerated)
        flagged_cells.push_back(idx);
    };

  std::vector<types::global_cell_index> idxs_to_be_agglomerated = {
    3235, 3238}; //{3,9};
  store_flagged_cells(idxs_to_be_agglomerated);
  std::vector<typename Triangulation<dim>::active_cell_iterator>
    cells_to_be_agglomerated;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated,
                                             cells_to_be_agglomerated);


  std::vector<types::global_cell_index> idxs_to_be_agglomerated2 = {
    831, 874}; //{25,19}
  store_flagged_cells(idxs_to_be_agglomerated2);

  std::vector<typename Triangulation<dim>::active_cell_iterator>
    cells_to_be_agglomerated2;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated2,
                                             cells_to_be_agglomerated2);


  std::vector<types::global_cell_index> idxs_to_be_agglomerated3 = {1226, 1227};
  store_flagged_cells(idxs_to_be_agglomerated3);

  std::vector<typename Triangulation<dim>::active_cell_iterator>
    cells_to_be_agglomerated3;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated3,
                                             cells_to_be_agglomerated3);


  std::vector<types::global_cell_index> idxs_to_be_agglomerated4 = {
    2279, 2278}; //{36,37}
  store_flagged_cells(idxs_to_be_agglomerated4);

  std::vector<typename Triangulation<dim>::active_cell_iterator>
    cells_to_be_agglomerated4;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated4,
                                             cells_to_be_agglomerated4);


  std::vector<types::global_cell_index> idxs_to_be_agglomerated5 = {
    3760, 3761}; //{3772,3773}
  store_flagged_cells(idxs_to_be_agglomerated5);

  std::vector<typename Triangulation<dim>::active_cell_iterator>
    cells_to_be_agglomerated5;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated5,
                                             cells_to_be_agglomerated5);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated6 = {3648, 3306};
  store_flagged_cells(idxs_to_be_agglomerated6);

  std::vector<typename Triangulation<dim>::active_cell_iterator>
    cells_to_be_agglomerated6;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated6,
                                             cells_to_be_agglomerated6);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated7 = {3765, 3764};
  store_flagged_cells(idxs_to_be_agglomerated7);

  std::vector<typename Triangulation<dim>::active_cell_iterator>
    cells_to_be_agglomerated7;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated7,
                                             cells_to_be_agglomerated7);


  // Agglomerate the cells just stored
  ah = std::make_unique<AgglomerationHandler<dim>>(*cached_tria);
  ah->define_agglomerate(cells_to_be_agglomerated);
  ah->define_agglomerate(cells_to_be_agglomerated2);
  ah->define_agglomerate(cells_to_be_agglomerated3);
  ah->define_agglomerate(cells_to_be_agglomerated4);
  ah->define_agglomerate(cells_to_be_agglomerated5);
  ah->define_agglomerate(cells_to_be_agglomerated6);
  ah->define_agglomerate(cells_to_be_agglomerated7);

  // Agglomerate all the other singletons
  for (std::size_t i = 0; i < tria.n_active_cells(); ++i)
    {
      // If not present, agglomerate all the singletons
      if (std::find(flagged_cells.begin(),
                    flagged_cells.end(),
                    cells[i]->active_cell_index()) == std::end(flagged_cells))
        ah->define_agglomerate({cells[i]});
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

  for (const auto &polytope : ah->polytope_iterators())
    {
      cell_matrix              = 0;
      cell_rhs                 = 0;
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

      for (unsigned int f = 0; f < n_faces; ++f)
        {
          double hf = polytope.master_cell()->face(0)->measure();

          if (polytope->at_boundary(f))
            {
              const auto &fe_face = ah->reinit(polytope, f);

              const unsigned int dofs_per_cell = fe_face.dofs_per_cell;
              std::vector<types::global_dof_index> local_dof_indices_bdary_cell(
                dofs_per_cell);

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

              // This is necessary to loop over internal faces only once.
              if (polytope->index() < neigh_polytope->index())
                {
                  unsigned int nofn =
                    polytope->neighbor_of_agglomerated_neighbor(f);
                  const auto &fe_faces =
                    ah->reinit_interface(polytope, neigh_polytope, f, nofn);

                  const auto &fe_faces0 = fe_faces.first;
                  const auto &fe_faces1 = fe_faces.second;

                  std::vector<types::global_dof_index>
                    local_dof_indices_neighbor(dofs_per_cell);

                  M11 = 0.;
                  M12 = 0.;
                  M21 = 0.;
                  M22 = 0.;

                  const auto &normals = fe_faces0.get_normal_vectors();
                  // M11
                  for (unsigned int q_index :
                       fe_faces0.quadrature_point_indices())
                    {
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

                  // distribute DoFs accordingly

                  // Retrieve DoFs info from the cell iterator.
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
  A_direct.initialize(system_matrix);
  A_direct.vmult(solution, system_rhs);
}



template <int dim>
void
Poisson<dim>::output_results()
{
  {
    const std::string filename = "interpolated_poisson.vtu";
    std::ofstream     output(filename);


    DataOut<dim> data_out;

    Vector<double> interpolated_solution;
    PolyUtils::interpolate_to_fine_grid(*ah, interpolated_solution, solution);

    Vector<float> difference_per_cell(tria.n_active_cells());
    Solution<dim> analytical_solution;
    VectorTools::integrate_difference(mapping,
                                      ah->output_dh,
                                      interpolated_solution,
                                      analytical_solution,
                                      difference_per_cell,
                                      QGauss<dim>(dg_fe.degree),
                                      VectorTools::L2_norm);

    const double L2_error =
      VectorTools::compute_global_error(tria,
                                        difference_per_cell,
                                        VectorTools::L2_norm);

    std::cout << "L2 error:" << L2_error << std::endl;

    /*
        data_out.attach_dof_handler(ah->output_dh);
        data_out.add_data_vector(interpolated_solution,
                                 "u",
                                 DataOut<dim>::type_dof_data);
        data_out.build_patches(mapping);
        data_out.write_vtu(output);
     */
  }
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
main()
{
  Poisson<2> poisson_problem;
  poisson_problem.run();

  return 0;
}