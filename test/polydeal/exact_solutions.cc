#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/reference_cell.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/vector_tools_integrate_difference.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <agglomeration_handler.h>
#include <poly_utils.h>

#include <algorithm>
#include <chrono>


// This test checks that the solution u(x,y)= x+y-1 is computed exactly when
// the polygonal mesh is made by polygons whose faces are not axis-aligned. The
// grid used in this test is a 4x4 grid "randomly distorted". Then, the 4 blocks
// naturally induced by the parent-child structure are agglomerated together
// manually.
// Then, the same is done but with a quadratic solution u(x,y)=x^2+y^2-1 and a
// quadratic discontinuous finite element space.

static constexpr double TOL = 1e-14;

enum class SolutionType
{
  LinearSolution,
  QuadraticSolution
};


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
  RightHandSide(const SolutionType &solution_type)
    : Function<dim>()
  {
    sol_type = solution_type;
  }

  virtual void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<double> &          values,
             const unsigned int /*component*/) const override;

  SolutionType sol_type;
};


template <int dim>
void
RightHandSide<dim>::value_list(const std::vector<Point<dim>> &points,
                               std::vector<double> &          values,
                               const unsigned int /*component*/) const
{
  (void)points;
  if (sol_type == SolutionType::LinearSolution)
    {
      for (unsigned int i = 0; i < values.size(); ++i)
        values[i] = 0.; //-4.; // ball, radial solution
    }
  else
    {
      for (unsigned int i = 0; i < values.size(); ++i)
        values[i] = -4.; // quadratic solution
    }
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



template <int dim>
class SolutionQuadratic : public Function<dim>
{
public:
  SolutionQuadratic()
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
SolutionQuadratic<dim>::value(const Point<dim> &p, const unsigned int) const
{
  return p[0] * p[0] + p[1] * p[1] - 1;
}

template <int dim>
Tensor<1, dim>
SolutionQuadratic<dim>::gradient(const Point<dim> &p, const unsigned int) const
{
  Assert(dim == 2, ExcMessage("This test only works in 2D."));
  (void)p;
  Tensor<1, dim> return_value;
  return_value[0] = 2 * p[0];
  return_value[1] = 2 * p[1];
  return return_value;
}



template <int dim>
void
SolutionQuadratic<dim>::value_list(const std::vector<Point<dim>> &points,
                                   std::vector<double> &          values,
                                   const unsigned int /*component*/) const
{
  for (unsigned int i = 0; i < values.size(); ++i)
    values[i] = this->value(points[i]);
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
  FE_DGQ<dim> *                              dg_fe;
  std::unique_ptr<AgglomerationHandler<dim>> ah;
  // no hanging node in DG discretization, we define an AffineConstraints
  // object
  // so we can use the distribute_local_to_global() directly.
  AffineConstraints<double>              constraints;
  SparsityPattern                        sparsity;
  SparseMatrix<double>                   system_matrix;
  Vector<double>                         solution;
  Vector<double>                         system_rhs;
  std::unique_ptr<GridTools::Cache<dim>> cached_tria;
  std::unique_ptr<const Function<dim>>   rhs_function;
  Function<dim> *                        analytical_solution;

public:
  Poisson(const SolutionType &solution_type);
  ~Poisson();

  void
  run();

  double       penalty_constant = 10;
  SolutionType sol_type;
};



template <int dim>
Poisson<dim>::Poisson(const SolutionType &solution_type)
  : mapping(1)
  , sol_type(solution_type)
{
  if (sol_type == SolutionType::LinearSolution)
    {
      dg_fe               = new FE_DGQ<dim>{1};
      analytical_solution = new SolutionLinear<dim>();
    }
  else
    {
      dg_fe               = new FE_DGQ<dim>{2};
      analytical_solution = new SolutionQuadratic<dim>();
    }
}



template <int dim>
Poisson<dim>::~Poisson()
{
  delete dg_fe;
  delete analytical_solution;
}



template <int dim>
void
Poisson<dim>::make_grid()
{
  GridIn<dim> grid_in;
  GridGenerator::hyper_cube(tria, 0, 1);
  tria.refine_global(2);
  GridTools::distort_random(0.25, tria);

  std::cout << "Size of tria: " << tria.n_active_cells() << std::endl;
  cached_tria  = std::make_unique<GridTools::Cache<dim>>(tria, mapping);
  rhs_function = std::make_unique<const RightHandSide<dim>>(sol_type);

  constraints.close();
}

template <int dim>
void
Poisson<dim>::setup_agglomeration()
{
  ah = std::make_unique<AgglomerationHandler<dim>>(*cached_tria);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated = {0, 1, 2, 3};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated,
                                             cells_to_be_agglomerated);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated2 = {4, 5, 6, 7};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated2;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated2,
                                             cells_to_be_agglomerated2);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated3 = {8,
                                                                    9,
                                                                    10,
                                                                    11};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated3;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated3,
                                             cells_to_be_agglomerated3);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated4 = {12,
                                                                    13,
                                                                    14,
                                                                    15};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated4;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated4,
                                             cells_to_be_agglomerated4);

  // Agglomerate the cells just stored
  ah->define_agglomerate(cells_to_be_agglomerated);
  ah->define_agglomerate(cells_to_be_agglomerated2);
  ah->define_agglomerate(cells_to_be_agglomerated3);
  ah->define_agglomerate(cells_to_be_agglomerated4);

  ah->distribute_agglomerated_dofs(*dg_fe);
  ah->create_agglomeration_sparsity_pattern(sparsity);
}



template <int dim>
void
Poisson<dim>::assemble_system()
{
  system_matrix.reinit(sparsity);
  solution.reinit(ah->n_dofs());
  system_rhs.reinit(ah->n_dofs());

  const unsigned int quadrature_degree      = 2 * dg_fe->get_degree() + 1;
  const unsigned int face_quadrature_degree = 2 * dg_fe->get_degree() + 1;
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


  LinearFunction<dim> linear_func{{1, 1}};
  double              test_integral = 0.;
  double              test_bdary    = 0.;
  double              test_volume   = 0.;
  for (const auto &polytope : ah->polytope_iterators())
    {
      // local_volume             = 0.;
      cell_matrix              = 0;
      cell_rhs                 = 0;
      const auto &agglo_values = ah->reinit(polytope);

      const auto &q_points = agglo_values.get_quadrature_points();

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
          test_integral += linear_values[q_index] * agglo_values.JxW(q_index);
          test_volume += agglo_values.JxW(q_index);
        }

      polytope->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);

      // Face terms
      const unsigned int n_faces = polytope->n_faces();
      AssertThrow(n_faces > 0, ExcMessage("Invalid element!"));



      // auto   polygon_boundary_vertices = ah->polytope_boundary(cell);
      for (unsigned int f = 0; f < n_faces; ++f)
        {
          if (polytope->at_boundary(f))
            {
              const auto &fe_face = ah->reinit(polytope, f);

              const unsigned int dofs_per_cell = fe_face.dofs_per_cell;
              std::vector<types::global_dof_index> local_dof_indices_bdary_cell(
                dofs_per_cell);

              const auto &face_q_points = fe_face.get_quadrature_points();
              std::vector<double> analytical_solution_values(
                face_q_points.size());
              analytical_solution->value_list(face_q_points,
                                              analytical_solution_values,
                                              1);

              // Get normal vectors seen from each agglomeration.
              const auto &normals = fe_face.get_normal_vectors();

              const double penalty =
                penalty_constant / std::fabs(polytope->diameter());

              cell_matrix = 0.;
              cell_rhs    = 0.;
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
                             (penalty)*fe_face.shape_value(i, q_index) *
                               fe_face.shape_value(j, q_index)) *
                            fe_face.JxW(q_index);
                        }
                      cell_rhs(i) +=
                        (penalty * analytical_solution_values[q_index] *
                           fe_face.shape_value(i, q_index) -
                         fe_face.shape_grad(i, q_index) * normals[q_index] *
                           analytical_solution_values[q_index]) *
                        fe_face.JxW(q_index);
                    }

                  test_bdary += fe_face.JxW(q_index);
                }

              // distribute DoFs
              polytope->get_dof_indices(local_dof_indices_bdary_cell);
              constraints.distribute_local_to_global(cell_matrix,
                                                     cell_rhs,
                                                     local_dof_indices,
                                                     system_matrix,
                                                     system_rhs);
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

#ifdef AGGLO_DEBUG
                  const auto &points0 = fe_faces0.get_quadrature_points();
                  const auto &points1 = fe_faces1.get_quadrature_points();
                  for (size_t i = 0;
                       i < fe_faces1.get_quadrature_points().size();
                       ++i)
                    {
                      double d = (points0[i] - points1[i]).norm();
                      Assert(d < 1e-15,
                             ExcMessage(
                               "Face qpoints at the interface do not match!"));
                    }

#endif

                  std::vector<types::global_dof_index>
                    local_dof_indices_neighbor(dofs_per_cell);

                  M11 = 0.;
                  M12 = 0.;
                  M21 = 0.;
                  M22 = 0.;

                  const auto & normals = fe_faces0.get_normal_vectors();
                  const double penalty =
                    penalty_constant / std::fabs(polytope->diameter());

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

  AssertThrow(
    std::fabs(test_integral - 1.) < TOL,
    ExcMessage(
      "Value for integral of linear function on this domain is not correct."));
  AssertThrow(std::fabs(test_volume - 1.) < TOL,
              ExcMessage("Value of measure of domain is not correct."));
  AssertThrow(std::fabs(test_bdary - 4.) < TOL,
              ExcMessage(
                "Value for the measure of the boundary is not correct."));
}

void
output_double_number(double input, const std::string &text)
{
  std::cout << text << input << std::endl;
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
  // Compute errors.

  // Prepare interpolation matrix onto the finer grid.
  ah->setup_output_interpolation_matrix();
  Vector<double> interpolated_solution(ah->output_dh.n_dofs());
  ah->output_interpolation_matrix.vmult(interpolated_solution, solution);

  // L2 error
  Vector<float> difference_per_cell(tria.n_active_cells());

  VectorTools::integrate_difference(mapping,
                                    ah->output_dh,
                                    interpolated_solution,
                                    *analytical_solution,
                                    difference_per_cell,
                                    QGauss<dim>(dg_fe->degree + 1),
                                    VectorTools::L2_norm);

  const double L2_error =
    VectorTools::compute_global_error(tria,
                                      difference_per_cell,
                                      VectorTools::L2_norm);

  // std::cout << "L2 error:" << L2_error << std::endl;
  AssertThrow(L2_error < TOL, ExcMessage("L2 error too large."));



  // H1 seminorm
  Vector<float> difference_per_cell_H1_semi(tria.n_active_cells());

  VectorTools::integrate_difference(mapping,
                                    ah->output_dh,
                                    interpolated_solution,
                                    *analytical_solution,
                                    difference_per_cell_H1_semi,
                                    QGauss<dim>(dg_fe->degree + 1),
                                    VectorTools::H1_seminorm);

  const double H1_seminorm =
    VectorTools::compute_global_error(tria,
                                      difference_per_cell_H1_semi,
                                      VectorTools::H1_seminorm);
  AssertThrow(H1_seminorm < TOL, ExcMessage("H1 seminorm too large."));

  // std::cout << "H1 seminorm:" << H1_seminorm << std::endl;
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
  deallog.depth_console(1);

  try
    {
      Poisson<2> poisson_problem_linear_sol{SolutionType::LinearSolution};
      poisson_problem_linear_sol.run();
      std::cout << "Linear: OK" << std::endl;

      Poisson<2> poisson_problem_quadratic_sol{SolutionType::QuadraticSolution};
      poisson_problem_quadratic_sol.run();
      std::cout << "Quadratic: OK" << std::endl;
    }
  catch (const std::exception &exc)
    {
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
    }

  return 0;
}
