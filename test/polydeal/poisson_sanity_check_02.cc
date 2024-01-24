#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/reference_cell.h>

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <agglomeration_handler.h>
#include <poly_utils.h>

#include <algorithm>

// Agglomerate a 2x2 mesh in the following way:
// |------------|-------------|
// |------------|-------------|
// |---- K0-----|------K1-----|
// |------------|-------------|
// x=1         x=0.5        x=1
// Test the result of v^T A v, where A is the DG matrix (without boundary terms)
// and v the interpolant of:
// - StepFunction (0) in K0, (1) in K1,
// - Vfunction: f(x,y)=|x-.5|


template <int dim>
class Vfunction : public Function<dim>
{
public:
  Vfunction() = default;
  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;
};

template <int dim>
double
Vfunction<dim>::value(const Point<dim> &p, const unsigned int) const
{
  // |x-.5|
  return (p[0] - .5) > 0 ? p[0] - .5 : .5 - p[0];
}

template <int dim>
class StepFunction : public Function<dim>
{
public:
  StepFunction() = default;
  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;
};

template <int dim>
double
StepFunction<dim>::value(const Point<dim> &p, const unsigned int) const
{
  return (p[0] < 0.5) ? 0. : 1.;
}


template <int dim>
class LaplaceOperator
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
  perform_sanity_check();


  Triangulation<dim>                         tria;
  MappingQ<dim>                              mapping;
  FE_DGQ<dim>                                dg_fe;
  std::unique_ptr<AgglomerationHandler<dim>> ah;
  // no hanging node in DG discretization, we define an AffineConstraints object
  // so we can use the distribute_local_to_global() directly.
  AffineConstraints<double>              constraints;
  SparsityPattern                        sparsity;
  SparseMatrix<double>                   system_matrix;
  Vector<double>                         solution;
  Vector<double>                         system_rhs;
  std::unique_ptr<GridTools::Cache<dim>> cached_tria;

public:
  LaplaceOperator(const unsigned int, const unsigned int fe_degree = 1);
  void
  run();

  double       penalty_constant = 10.;
  unsigned int n_subdomains;
};



template <int dim>
LaplaceOperator<dim>::LaplaceOperator(const unsigned int n_subdomains,
                                      const unsigned int fe_degree)
  : mapping(1)
  , dg_fe(fe_degree)
  , n_subdomains(n_subdomains)
{}

template <int dim>
void
LaplaceOperator<dim>::make_grid()
{
  // GridGenerator::hyper_ball(tria);
  // tria.refine_global(5); // 4
  GridGenerator::hyper_cube(tria, 0., 1.);
  tria.refine_global(1);
  cached_tria = std::make_unique<GridTools::Cache<dim>>(tria, mapping);
  constraints.close();
}



template <int dim>
void
LaplaceOperator<dim>::setup_agglomeration()
{
  ah = std::make_unique<AgglomerationHandler<dim>>(*cached_tria);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated = {0, 2};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated,
                                             cells_to_be_agglomerated);
  ah->define_agglomerate(cells_to_be_agglomerated);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated2 = {1, 3};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated2;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated2,
                                             cells_to_be_agglomerated2);
  ah->define_agglomerate(cells_to_be_agglomerated2);

  ah->distribute_agglomerated_dofs(dg_fe);
  ah->create_agglomeration_sparsity_pattern(sparsity);
}



template <int dim>
void
LaplaceOperator<dim>::assemble_system()
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

      const auto &        q_points  = agglo_values.get_quadrature_points();
      const unsigned int  n_qpoints = q_points.size();
      std::vector<double> rhs(n_qpoints);

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
      AssertThrow(n_faces > 0,
                  ExcMessage(
                    "Invalid element: at least 4 faces are required."));

      for (unsigned int f = 0; f < n_faces; ++f)
        {
          const double current_element_diameter =
            std::fabs(polytope->diameter());
          // const double penalty =
          //   penalty_constant * (1. / current_element_diameter);

          if (polytope->at_boundary(f))
            {
              const auto &fe_face = ah->reinit(polytope, f);

              const unsigned int dofs_per_cell = fe_face.dofs_per_cell;
              std::vector<types::global_dof_index> local_dof_indices_bdary_cell(
                dofs_per_cell);

              // Get normal vectors seen from each agglomeration.
              // const auto &normals = fe_face.get_normal_vectors();
              cell_matrix = 0.;
              for ([[maybe_unused]] unsigned int q_index :
                   fe_face.quadrature_point_indices())
                {
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                          cell_matrix(i, j) +=
                            0.; // zero out, for this sanity check we need to
                                // neglect boundary contributions
                        }
                    }
                }

              // distribute DoFs
              polytope->get_dof_indices(local_dof_indices_bdary_cell);
              constraints.distribute_local_to_global(
                cell_matrix, local_dof_indices_bdary_cell, system_matrix);
            }
          else
            {
              const auto &neigh_polytope = polytope->neighbor(f);

              const double neigh_element_diameter =
                std::fabs(neigh_polytope->diameter());
              const double penalty =
                penalty_constant *
                std::max(1. / current_element_diameter,
                         1. / neigh_element_diameter); // Cinv still missing

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
                                 (penalty)*fe_faces0.shape_value(i, q_index) *
                                   fe_faces0.shape_value(j, q_index)) *
                                fe_faces0.JxW(q_index);

                              M12(i, j) +=
                                (0.5 * fe_faces0.shape_grad(i, q_index) *
                                   normals[q_index] *
                                   fe_faces1.shape_value(j, q_index) -
                                 0.5 * fe_faces1.shape_grad(j, q_index) *
                                   normals[q_index] *
                                   fe_faces0.shape_value(i, q_index) -
                                 (penalty)*fe_faces0.shape_value(i, q_index) *
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
                                 (penalty)*fe_faces1.shape_value(i, q_index) *
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
                                 (penalty)*fe_faces1.shape_value(i, q_index) *
                                   fe_faces1.shape_value(j, q_index)) *
                                fe_faces1.JxW(q_index);
                            }
                        }
                    }


                  // Retrieve DoFs info from the polytope iterator.
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
        } // Loop over faces of current polytope
    }     // Loop over cells
}



template <int dim>
void
LaplaceOperator<dim>::perform_sanity_check()
{
  std::vector<std::unique_ptr<Function<dim>>> functions;
  functions.emplace_back(
    new StepFunction<dim>{});                   // v(x,y)= 1 on left, 0 on right
  functions.emplace_back(new Vfunction<dim>{}); // v(x,y)= |x-.5|
  std::array<std::string, 2> names_lookup{{"Step function", "V function"}};

  unsigned int fcts = 0;
  for (const auto &func : functions)
    {
      Vector<double> interp_vector(ah->get_dof_handler().n_dofs());
      VectorTools::interpolate(*(ah->euler_mapping),
                               ah->get_dof_handler(),
                               *func,
                               interp_vector);
      const double value =
        system_matrix.matrix_scalar_product(interp_vector, interp_vector);
      std::cout << "Test with " << names_lookup[fcts] << " = " << value
                << std::endl;
      ++fcts;
    }
}


template <int dim>
void
LaplaceOperator<dim>::run()
{
  make_grid();
  setup_agglomeration();
  assemble_system();
  perform_sanity_check();
}

int
main()
{
  for (const unsigned int n_agglomerates : {50})
    {
      LaplaceOperator<2> sanity_check{n_agglomerates};
      sanity_check.run();
    }

  return 0;
}