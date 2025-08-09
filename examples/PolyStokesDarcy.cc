#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_accessors.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/component_mask.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/derivative_approximation.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <agglomeration_handler.h>
#include <fe_agglodgp.h>
#include <poly_utils.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

/*============================================================================
   Stokes–Darcy Problem (with Beavers–Joseph–Saffman (BJS) interface coupling)

   Domain: Ω = Ω_S ∪ Ω_D ⊂ ℝ^d, where:
       - Ω_S: fluid region (Stokes)
       - Ω_D: porous medium region (Darcy)
       - Γ = ∂Ω_S ∩ ∂Ω_D: interface between the two regions

   Unknowns: (u, p_S, p_D)
       - u  : velocity in Ω_S
       - p_S: pressure in Ω_S
       - p_D: pressure in Ω_D

   Governing equations:

   In Ω_S (Stokes region):
       -ν Δu + ∇p_S = f_S          (momentum balance)
              ∇ · u = 0          (mass conservation)

   In Ω_D (Darcy region):
       -∇ · (K ∇p_D) = f_D           (Darcy’s law)

   Interface conditions on Γ:
       1. u · n = −(K ∇p_D) · n                     (normal flux continuity)
       2. (−p_S I + ν ∇u) · n = −p_D n              (normal stress balance)
       3. (ν / G) (u · τ) = τ · (−p_S I + ν ∇u) · n   (BJS condition)

   Boundary conditions:
       - On ∂Ω_S \ Γ:    u = g_S                    (Dirichlet)
       - On ∂Ω_D \ Γ:    −(K ∇p_D) · n = g_D          (Neumann)

   Integral constraint:
       ∫_Ω (p_S + p_D) dx = 0

   Parameters:
       - ν: fluid viscosity (constant)
       - K: permeability tensor (symmetric, positive-definite)
       - n: unit normal vector
       - τ: unit tangential vector on Γ
       - α_BJ: Beavers–Joseph coefficient (from experiment)
       - G = √[ν (τ · K · τ)] / α_BJ   (BJS slip coefficient)
============================================================================*/

namespace StokesDarcyNamespace
{
  using namespace dealii;

  /*============================================================================
     ExactSolution

     Represents the manufactured solution (u, p_S, p_D) to the coupled
     Stokes–Darcy system with BJS interface condition. Used to verify
     convergence.

     Domain:
      Ω = Ω_S ∪ Ω_D ⊂ ℝ², where:
        Ω_S = [0,1] × [1/2, 1]     (Stokes region)
        Ω_D = [0,1] × [0, 1/2]     (Darcy region)

     Interface: y = 1/2

     Exact solution:

      u = [
               (2−x)(1.5−y)(y−ξ)
               -y³/3 + y²/2 (ξ + 1.5) − 1.5ξy − 0.5 + sin(ωx)
            ]

      p_S = −(sin(ωx) + χ)/(2K) + ν(0.5−ξ) + cos(πy)

      p_D = −χ (y + 0.5)² / (2K) − sin(ωx)y / K

     Parameters:
      ν  = 0.1           (fluid viscosity)
      K  = 1             (scalar permeability, tensor = K * I)
      α₀ = 0.5           (BJ coefficient)
      G  = sqrt(νK)/α₀   (slip coefficient)
      ξ  = (1−G) / [2(1+G)]
      χ  = (−30ξ − 17) / 48
      ω  = 6

  --- Reference ---
    K. Lipnikov, D. Vassilev, and I. Yotov,
    "Discontinuous Galerkin and Mimetic Finite Difference Methods
     for Coupled Stokes–Darcy Flows on Polygonal and Polyhedral Grids",
    Numerische Mathematik, 126, pp. 321–360, 2014.
  ============================================================================*/
  template <int dim>
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution()
      : Function<dim>(dim + 2)
    {}

    virtual void
    vector_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>>   &value_list) const override;

    virtual void
    vector_gradient_list(
      const std::vector<Point<dim>>            &points,
      std::vector<std::vector<Tensor<1, dim>>> &gradient_list) const override;

  private:
    static constexpr double PI     = numbers::PI;
    const double            w      = 6.0;
    const double            mu     = 0.1;
    const double            K      = 1.0;
    const double            alpha0 = 0.5;
    const double            G      = std::sqrt(mu * K) / alpha0;
    const double            xi     = (1.0 - G) / (2.0 * (1.0 + G));
    const double            chi    = (-30.0 * xi - 17.0) / 48.0;
    const double            mean_pressure =
      (cos(6.) - 1.) / 24. - 1. / PI - chi / 4. + mu * (0.5 - xi) / 2. +
      chi * (1. / 48. - 1. / 6.) + cos(6.) / 48. - 1. / 48.;
  };

  template <int dim>
  void
  ExactSolution<dim>::vector_value_list(
    const std::vector<Point<dim>> &points,
    std::vector<Vector<double>>   &value_list) const
  {
    using std::cos;
    using std::pow;
    using std::sin;

    AssertDimension(points.size(), value_list.size());

    for (unsigned int i = 0; i < points.size(); ++i)
      {
        const double x = points[i][0];
        const double y = points[i][1];

        if (y >= 0.5)
          {
            value_list[i][0] = (2. - x) * (1.5 - y) * (y - xi);
            value_list[i][1] = -pow(y, 3) / 3. + y * y / 2. * (xi + 1.5) -
                               1.5 * xi * y - 0.5 + sin(w * x);
            value_list[i][2] = -(sin(w * x) + chi) / (2. * K) +
                               mu * (0.5 - xi) + cos(PI * y) - mean_pressure;
            value_list[i][3] = 0.0;
          }
        else
          {
            value_list[i][0] = 0.0;
            value_list[i][1] = 0.0;
            value_list[i][2] = 0.0;
            value_list[i][3] = -(chi * pow(y + 0.5, 2)) / (K * 2.) -
                               sin(w * x) * y / K - mean_pressure;
          }
      }
  }

  template <int dim>
  void
  ExactSolution<dim>::vector_gradient_list(
    const std::vector<Point<dim>>            &points,
    std::vector<std::vector<Tensor<1, dim>>> &gradient_list) const
  {
    using std::cos;
    using std::sin;

    AssertDimension(points.size(), gradient_list.size());

    for (unsigned int i = 0; i < points.size(); ++i)
      {
        const double x = points[i][0];
        const double y = points[i][1];

        if (y >= 0.5)
          {
            gradient_list[i][0][0] = -(1.5 - y) * (y - xi);
            gradient_list[i][0][1] = (2 - x) * (-2 * y + xi + 1.5);

            gradient_list[i][1][0] = w * cos(w * x);
            gradient_list[i][1][1] = -y * y + y * (xi + 1.5) - 1.5 * xi;

            gradient_list[i][2][0] = -w / 2. * cos(w * x);
            gradient_list[i][2][1] = -PI * sin(PI * y);

            gradient_list[i][3][0] = 0.0;
            gradient_list[i][3][1] = 0.0;
          }
        else
          {
            gradient_list[i][0][0] = 0.0;
            gradient_list[i][0][1] = 0.0;

            gradient_list[i][1][0] = 0.0;
            gradient_list[i][1][1] = 0.0;

            gradient_list[i][2][0] = 0.0;
            gradient_list[i][2][1] = 0.0;

            gradient_list[i][3][0] = -w * cos(w * x) * y;
            gradient_list[i][3][1] = -chi * (y + 0.5) - sin(w * x);
          }
      }
  }

  /*============================================================================
     RightHandSide_S and RightHandSide_D

     The right-hand side functions for the Stokes and Darcy problems.
  ============================================================================*/
  template <int dim>
  class RightHandSide_S : public TensorFunction<1, dim, double>
  {
  public:
    RightHandSide_S()
      : TensorFunction<1, dim, double>()
    {}

    void
    value_list(const std::vector<Point<dim>> &points,
               std::vector<Tensor<1, dim>>   &value_list) const override;

  private:
    static constexpr double PI     = numbers::PI;
    const double            w      = 6.0;
    const double            mu     = 0.1;
    const double            K      = 1.0;
    const double            alpha0 = 0.5;
    const double            G      = std::sqrt(mu * K) / alpha0;
    const double            xi     = (1.0 - G) / (2.0 * (1.0 + G));
  };

  template <int dim>
  void
  RightHandSide_S<dim>::value_list(
    const std::vector<Point<dim>> &points,
    std::vector<Tensor<1, dim>>   &value_list) const
  {
    using std::cos;
    using std::sin;
    AssertDimension(points.size(), value_list.size());

    for (unsigned int i = 0; i < points.size(); ++i)
      {
        const double x = points[i][0];
        const double y = points[i][1];

        value_list[i][0] = 0.2 * (2. - x) - w / 2. * cos(w * x);
        value_list[i][1] =
          -0.1 * (-w * w * sin(w * x) - 2. * y + xi + 1.5) - PI * sin(PI * y);
      }
  }

  template <int dim>
  class RightHandSide_D : public Function<dim>
  {
  public:
    RightHandSide_D()
      : Function<dim>()
    {}

    virtual void
    value_list(const std::vector<Point<dim>> &points,
               std::vector<double>           &value_list,
               const unsigned int /*component*/) const override;

  private:
    static constexpr double PI     = numbers::PI;
    const double            w      = 6.0;
    const double            mu     = 0.1;
    const double            K      = 1.0;
    const double            alpha0 = 0.5;
    const double            G      = std::sqrt(mu * K) / alpha0;
    const double            xi     = (1.0 - G) / (2.0 * (1.0 + G));
    const double            chi    = (-30.0 * xi - 17.0) / 48.0;
  };

  template <int dim>
  void
  RightHandSide_D<dim>::value_list(const std::vector<Point<dim>> &points,
                                   std::vector<double>           &value_list,
                                   const unsigned int /*component*/) const
  {
    using std::sin;
    AssertDimension(points.size(), value_list.size());

    for (unsigned int i = 0; i < points.size(); ++i)
      {
        const double x = points[i][0];
        const double y = points[i][1];

        value_list[i] = -w * w * sin(w * x) * y + chi;
      }
  }
  /*============================================================================
     BoundaryDirichlet_S

     Represents the Dirichlet boundary conditions for the Stokes problem.
 ============================================================================*/
  template <int dim>
  class BoundaryDirichlet_S : public TensorFunction<1, dim, double>
  {
  public:
    BoundaryDirichlet_S()
      : TensorFunction<1, dim, double>()
    {}

    void
    value_list(const std::vector<Point<dim>> &points,
               std::vector<Tensor<1, dim>>   &value_list) const override;

  private:
    static constexpr double PI     = numbers::PI;
    const double            w      = 6.0;
    const double            mu     = 0.1;
    const double            K      = 1.0;
    const double            alpha0 = 0.5;
    const double            G      = std::sqrt(mu * K) / alpha0;
    const double            xi     = (1.0 - G) / (2.0 * (1.0 + G));
  };

  template <int dim>
  void
  BoundaryDirichlet_S<dim>::value_list(
    const std::vector<Point<dim>> &points,
    std::vector<Tensor<1, dim>>   &value_list) const
  {
    using std::pow;
    using std::sin;

    AssertDimension(points.size(), value_list.size());

    for (unsigned int i = 0; i < points.size(); ++i)
      {
        const double x = points[i][0];
        const double y = points[i][1];

        value_list[i][0] = (2. - x) * (1.5 - y) * (y - xi);
        value_list[i][1] = -pow(y, 3) / 3. + y * y / 2. * (xi + 1.5) -
                           1.5 * xi * y - 0.5 + sin(w * x);
      }
  }

  /*============================================================================
     BoundaryNeumann_D

     Represents the Neumann boundary conditions for the Darcy problem.
 ============================================================================*/
  template <int dim>
  class BoundaryNeumann_D : public Function<dim>
  {
  public:
    BoundaryNeumann_D()
      : Function<dim>()
    {}

    virtual void
    value_list(const std::vector<Point<dim>> &points,
               std::vector<double>           &value_list,
               const unsigned int /*component*/) const override;

  private:
    static constexpr double PI     = numbers::PI;
    const double            w      = 6.0;
    const double            mu     = 0.1;
    const double            K      = 1.0;
    const double            alpha0 = 0.5;
    const double            G      = std::sqrt(mu * K) / alpha0;
    const double            xi     = (1.0 - G) / (2.0 * (1.0 + G));
    const double            chi    = (-30.0 * xi - 17.0) / 48.0;
  };

  template <int dim>
  void
  BoundaryNeumann_D<dim>::value_list(const std::vector<Point<dim>> &points,
                                     std::vector<double>           &value_list,
                                     const unsigned int /*component*/) const
  {
    using std::pow;
    using std::sin;

    AssertDimension(points.size(), value_list.size());

    for (unsigned int i = 0; i < points.size(); ++i)
      {
        const double x = points[i][0];
        const double y = points[i][1];

        if (abs(x) <= 1e-9)
          value_list[i] =
            -1. * (w * cos(w * x) * y) +
            0. * (chi * (y + 0.5) + sin(w * x)); // normal vector is (-1,0)

        if (abs(y) <= 1e-9)
          value_list[i] =
            0. * (w * cos(w * x) * y) +
            -1. * (chi * (y + 0.5) + sin(w * x)); // normal vector is (0,-1)

        if (abs(x - 1.) <= 1e-9)
          value_list[i] =
            1. * (w * cos(w * x) * y) +
            0. * (chi * (y + 0.5) + sin(w * x)); // normal vector is (1,0)
      }
  }

  /*============================================================================
     The StokesDarcyProblem class below solves a coupled Stokes–Darcy flow
     using the Interior Penalty Discontinuous Galerkin (IPDG) method with a
     Beavers–Joseph–Saffman (BJS) interface condition.

     The computational domain consists of an upper Stokes region (free-flow)
     and a lower Darcy region (porous medium), separated by a horizontal
     interface at y = 1/2. A manufactured solution is used to verify convergence
     and assess the accuracy of the numerical method.

     Initially, the entire domain is partitioned into 16 curved polygonal cells,
     with the interface conforming to curved boundaries. These initial cells
     are then refined and agglomerated into polygonal elements used
     in the final computation.
  ============================================================================*/
  template <int dim>
  class StokesDarcyProblem
  {
  public:
    StokesDarcyProblem(const unsigned int degree_velocities = 2,
                       const unsigned int degree_pressure_S = 1,
                       const unsigned int degree_pressure_D = 1,
                       const unsigned int extraction_level  = 2);

    // Run the simulation
    void
    run();

    // Get error norms
    double
    get_error_velocity_L2() const;
    double
    get_error_velocity_H1() const;
    double
    get_error_pressure_S() const;
    double
    get_error_pressure_D() const;

    // Get number of degrees of freedom
    unsigned int
    get_n_dofs() const;

  private:
    static bool
    polytope_is_in_Stokes_domain(
      const typename AgglomerationHandler<dim>::agglomeration_iterator
        &polytope);
    static bool
    polytope_is_in_Darcy_domain(
      const typename AgglomerationHandler<dim>::agglomeration_iterator
        &polytope);

    // Grid and agglomeration setup
    void
    make_base_grid();
    void
    make_agglo_grid();
    void
    set_active_fe_indices();
    void
    setup_agglomeration();

    // Assemble system matrix and solve linear system
    void
    assemble_system();
    void
    solve();

    // Constraint handling
    void
    zero_pressure_dof_constraint();
    void
    mean_pressure_to_zero();

    // Post-processing
    void
    compute_errors();
    void
    output_results(unsigned int n_subdomains) const;

    // Degrees of finite elements (velocity and pressure)
    const unsigned int degree_velocities;
    const unsigned int degree_pressure_S;
    const unsigned int degree_pressure_D;
    const unsigned int extraction_level;
    unsigned int n_subdomains; // Number of subdomains after agglomeration.

    // Physical parameters and domain info
    const double         viscosity_nu        = 0.1;
    const double         permeability_scalar = 1.0;
    const Tensor<2, dim> permeability_K =
      permeability_scalar * unit_symmetric_tensor<dim>();
    const double alpha_BJ = 0.5; // Beavers–Joseph coefficient
    const double nu_over_G =
      alpha_BJ * std::sqrt(viscosity_nu) /
      std::sqrt(permeability_scalar); // ν / G = α_BJ * √ν / √(τ · K · τ)
    double       domain_area = 1.;
    unsigned int num_domain; // Number of distinct domains.
                             // Agglomeration is restricted within each domain,
                             // allowing curved interfaces between different
                             // domains to be preserved.

    Triangulation<dim>   triangulation;
    const MappingQ1<dim> mapping;

    hp::FECollection<dim>    fe_collection;
    hp::QCollection<dim>     q_collection;
    hp::QCollection<dim - 1> face_q_collection;

    AffineConstraints<double> constraints;

    std::unique_ptr<AgglomerationHandler<dim>> agglo_handler;
    std::unique_ptr<GridTools::Cache<dim>>     cached_tria;

    SparsityPattern      sparsity;
    SparseMatrix<double> system_matrix;
    Vector<double>       solution;
    Vector<double>       system_rhs;

    std::unique_ptr<const Function<dim>>                  exact_solution;
    std::unique_ptr<const TensorFunction<1, dim, double>> rhs_function_S;
    std::unique_ptr<const Function<dim>>                  rhs_function_D;
    std::unique_ptr<const TensorFunction<1, dim, double>> bcDirichlet_S;
    std::unique_ptr<const Function<dim>>                  bcNeumann_D;

    static constexpr double penalty_constant_v   = 40.0;
    static constexpr double penalty_constant_p_S = 1.0;
    static constexpr double penalty_constant_p_D = 10.0;

    Vector<double> interpolated_solution;

    double error_velocity_L2;
    double error_velocity_H1;
    double error_pressure_S;
    double error_pressure_D;
  };

  template <int dim>
  StokesDarcyProblem<dim>::StokesDarcyProblem(
    const unsigned int degree_velocities,
    const unsigned int degree_pressure_S,
    const unsigned int degree_pressure_D,
    const unsigned int extraction_level)
    : degree_velocities(degree_velocities)
    , degree_pressure_S(degree_pressure_S)
    , degree_pressure_D(degree_pressure_D)
    , extraction_level(extraction_level)
    , triangulation(Triangulation<dim>::maximum_smoothing)
    , mapping()
  {
    FESystem<dim> stokes_fe(FE_AggloDGP<dim>(degree_velocities) ^ dim,
                            FE_AggloDGP<dim>(degree_pressure_S),
                            FE_Nothing<dim>());
    FESystem<dim> darcy_fe(FE_Nothing<dim>() ^ dim,
                           FE_Nothing<dim>(),
                           FE_AggloDGP<dim>(degree_pressure_D));
    fe_collection.push_back(stokes_fe);
    fe_collection.push_back(darcy_fe);

    const QGauss<dim>     quadrature_S(degree_velocities);
    const QGauss<dim>     quadrature_D(degree_pressure_D);
    const QGauss<dim - 1> face_quadrature_S(degree_velocities + 1);
    const QGauss<dim - 1> face_quadrature_D(degree_pressure_D + 1);

    q_collection.push_back(quadrature_S);
    q_collection.push_back(quadrature_D);
    face_q_collection.push_back(face_quadrature_S);
    face_q_collection.push_back(face_quadrature_D);

    exact_solution = std::make_unique<const ExactSolution<dim>>();
    rhs_function_S = std::make_unique<const RightHandSide_S<dim>>();
    rhs_function_D = std::make_unique<const RightHandSide_D<dim>>();
    bcDirichlet_S  = std::make_unique<const BoundaryDirichlet_S<dim>>();
    bcNeumann_D    = std::make_unique<const BoundaryNeumann_D<dim>>();
  }

  /*============================================================================
     Create and cache the base computational grid for the Stokes–Darcy problem.

     This function constructs a structured 4×4 grid over the unit square domain
     Ω = [0,1] × [0,1], with total area = 1.0 and 16 cells. If a cached grid
     file exists, it is loaded directly and relevant manifold IDs are reset.
     Otherwise, the function generates a new grid, assigns manifold IDs to
     selected faces to induce curved interfaces via FunctionManifold, applies
     6 global refinements, and saves the result for future reuse.

     Curved interfaces are encoded using manifold patches defined by explicit
     push-forward maps (from parameter space to physical space). These
     interfaces create oscillating geometries along vertical and horizontal
     face bands:
         - vertical strips near x = 0.25, 0.5, 0.75 (IDs 1–3)
         - horizontal strips near y = 0.25, 0.75 (IDs 4–5)

     The manifolds are assigned only to **interior** faces whose center
     lies within specified coordinate bands.

     Output:
       - Refined grid saved as "cached_base_grid_StokesDarcy"
       - Visualization output to "base_grid_StokesDarcy.msh"
  ============================================================================*/
  template <int dim>
  void
  StokesDarcyProblem<dim>::make_base_grid()
  {
    Point<2>                  bottom_left(0., 0.);
    Point<2>                  top_right(1., 1.);
    std::vector<unsigned int> subdivisions = {4, 4};
    GridGenerator::subdivided_hyper_rectangle(triangulation,
                                              subdivisions,
                                              bottom_left,
                                              top_right);
    domain_area = 1.0;
    num_domain  = 16;

    const std::string filename = "cached_base_grid_StokesDarcy";

    const std::string push_forward_expr1 =
      "0.25 + 0.02*sin(8*" + std::to_string(numbers::PI) +
      "*x) + 0.005*sin(32*" + std::to_string(numbers::PI) + "*x); x"; // (x,y)
    const std::string pull_back_expr1 = "y"; // chart u = y
    auto              manifold1 =
      std::make_shared<FunctionManifold<dim, dim, 1>>(push_forward_expr1,
                                                      pull_back_expr1);

    const std::string push_forward_expr2 =
      "0.5 - 0.02*sin(4*" + std::to_string(numbers::PI) +
      "*x) - 0.002*sin(64*" + std::to_string(numbers::PI) + "*x); x"; // (x,y)
    const std::string pull_back_expr2 = "y"; // chart u = y
    auto              manifold2 =
      std::make_shared<FunctionManifold<dim, dim, 1>>(push_forward_expr2,
                                                      pull_back_expr2);

    const std::string push_forward_expr3 =
      "0.75 + 0.02*sin(8*" + std::to_string(numbers::PI) +
      "*x) + 0.005*sin(32*" + std::to_string(numbers::PI) + "*x); x"; // (x,y)
    const std::string pull_back_expr3 = "y"; // chart u = y
    auto              manifold3 =
      std::make_shared<FunctionManifold<dim, dim, 1>>(push_forward_expr3,
                                                      pull_back_expr3);

    const std::string push_forward_expr4 =
      "x; 0.25 - 0.01*sin(8*" + std::to_string(numbers::PI) + "*x)"; // (x,y)
    const std::string pull_back_expr4 = "x"; // chart u = y
    auto              manifold4 =
      std::make_shared<FunctionManifold<dim, dim, 1>>(push_forward_expr4,
                                                      pull_back_expr4);

    const std::string push_forward_expr5 =
      "x; 0.75 - 0.01*sin(8*" + std::to_string(numbers::PI) + "*x)"; // (x,y)
    const std::string pull_back_expr5 = "x"; // chart u = y
    auto              manifold5 =
      std::make_shared<FunctionManifold<dim, dim, 1>>(push_forward_expr5,
                                                      pull_back_expr5);

    if (std::filesystem::exists(filename + "_triangulation.data"))
      {
        std::cout << "     Loading cached base grid from " << filename << " ..."
                  << std::endl;
        triangulation.load(filename);
        triangulation.set_manifold(1, *manifold1);
        triangulation.set_manifold(2, *manifold2);
        triangulation.set_manifold(3, *manifold3);
        triangulation.set_manifold(4, *manifold4);
        triangulation.set_manifold(5, *manifold5);
      }
    else
      {
        std::cout << "     Cached base grid not found. Generating new grid..."
                  << std::endl;

        triangulation.set_manifold(1, *manifold1);
        triangulation.set_manifold(2, *manifold2);
        triangulation.set_manifold(3, *manifold3);
        triangulation.set_manifold(4, *manifold4);
        triangulation.set_manifold(5, *manifold5);

        int material_id = 0;
        for (const auto &cell : triangulation.active_cell_iterators())
          {
            cell->set_material_id(material_id);
            material_id++;

            for (unsigned int f = 0; f < 4; ++f)
              if (!cell->at_boundary(f))
                {
                  if ((cell->face(f)->center()[0] > 0.15) &&
                      (cell->face(f)->center()[0] < 0.35))
                    cell->face(f)->set_all_manifold_ids(1);

                  if ((cell->face(f)->center()[0] > 0.4) &&
                      (cell->face(f)->center()[0] < 0.6))
                    cell->face(f)->set_all_manifold_ids(2);

                  if ((cell->face(f)->center()[0] > 0.65) &&
                      (cell->face(f)->center()[0] < 0.85))
                    cell->face(f)->set_all_manifold_ids(3);

                  if ((cell->face(f)->center()[1] > 0.15) &&
                      (cell->face(f)->center()[1] < 0.35))
                    cell->face(f)->set_all_manifold_ids(4);

                  if ((cell->face(f)->center()[1] > 0.65) &&
                      (cell->face(f)->center()[1] < 0.85))
                    cell->face(f)->set_all_manifold_ids(5);
                }
          }
        triangulation.refine_global(6);

        triangulation.save(filename);
        std::cout << "     Saved grid to " << filename << std::endl;

        GridOut       grid_out;
        std::ofstream out("base_grid_StokesDarcy.msh");
        grid_out.write_msh(triangulation, out);
      }
  }

  /*============================================================================
    Generate an agglomerated mesh using R3MG.

    Builds a fine background grid, partitions each material subdomain
    using separate R-trees, and performs cell agglomeration independently.
  ============================================================================*/
  template <int dim>
  void
  StokesDarcyProblem<dim>::make_agglo_grid()
  {
    make_base_grid();

    std::cout << "     Size of base grid: " << triangulation.n_active_cells()
              << std::endl;
    cached_tria =
      std::make_unique<GridTools::Cache<dim>>(triangulation, mapping);
    agglo_handler = std::make_unique<AgglomerationHandler<dim>>(*cached_tria);

    // Partition with Rtree
    namespace bgi = boost::geometry::index;
    static constexpr unsigned int max_elem_per_node =
      PolyUtils::constexpr_pow(2, dim);

    std::vector<
      std::vector<std::pair<BoundingBox<dim>,
                            typename Triangulation<dim>::active_cell_iterator>>>
      all_boxes(num_domain);
    // To preserve the curved boundaries of the domains, we use separate R-trees
    // for each domain. A "boxes" is a collection of bounding boxes for a single
    // domain. "all_boxes" is a collection of all such "boxes".
    for (const auto &cell : triangulation.active_cell_iterators())
      all_boxes[cell->material_id()].emplace_back(
        mapping.get_bounding_box(cell), cell);

    for (unsigned int i = 0; i < num_domain; ++i)
      {
        auto tree = pack_rtree<bgi::rstar<max_elem_per_node>>(all_boxes[i]);

        std::cout << "     Total number of available levels in domain_" << i
                  << ": " << n_levels(tree) << std::endl;

        const unsigned int extraction_level_sub =
          std::min(extraction_level -
                     (int)(std::log(num_domain) / std::log(max_elem_per_node)),
                   n_levels(tree));

        CellsAgglomerator<dim, decltype(tree)> agglomerator{
          tree, extraction_level_sub};
        const auto vec_agglomerates = agglomerator.extract_agglomerates();
        for (const auto &agglo : vec_agglomerates)
          agglo_handler->define_agglomerate(agglo, fe_collection.size());
      }

    n_subdomains = agglo_handler->n_agglomerates();
    std::cout << "     N subdomains = " << n_subdomains << std::endl;
  }

  template <int dim>
  bool
  StokesDarcyProblem<dim>::polytope_is_in_Stokes_domain(
    const typename AgglomerationHandler<dim>::agglomeration_iterator &polytope)
  {
    if (polytope.master_cell()->center()[1] > 0.5) // top part
      return true;
    else
      return false;
  }

  template <int dim>
  bool
  StokesDarcyProblem<dim>::polytope_is_in_Darcy_domain(
    const typename AgglomerationHandler<dim>::agglomeration_iterator &polytope)
  {
    if (polytope.master_cell()->center()[1] < 0.5) // bottom part
      return true;
    else
      return false;
  }

  template <int dim>
  void
  StokesDarcyProblem<dim>::set_active_fe_indices()
  {
    for (const auto &polytope : agglo_handler->polytope_iterators())
      {
        if (polytope_is_in_Stokes_domain(polytope))     // top part
          polytope->set_active_fe_index(0);             // Stokes
        else if (polytope_is_in_Darcy_domain(polytope)) // bottom part
          polytope->set_active_fe_index(1);             // Darcy
        else
          Assert(false,ExcMessage(
                   "Polytope with index " + std::to_string(polytope->index()) +
                   " should belong to either Stokes or Darcy domain."));
      }
  }

  template <int dim>
  void
  StokesDarcyProblem<dim>::setup_agglomeration()
  {
    set_active_fe_indices();
    agglo_handler->distribute_agglomerated_dofs(fe_collection);
    DynamicSparsityPattern dsp;
    constraints.clear();
    zero_pressure_dof_constraint();
    constraints.close();
    agglo_handler->create_agglomeration_sparsity_pattern(dsp, constraints);
    sparsity.copy_from(dsp);

    std::string polygon_boundaries{"polygon_rtree_" +
                                   std::to_string(n_subdomains)};
    PolyUtils::export_polygon_to_csv_file(*agglo_handler, polygon_boundaries);
  }

  template <int dim>
  void
  StokesDarcyProblem<dim>::assemble_system()
  {
    system_matrix.reinit(sparsity);
    solution.reinit(agglo_handler->n_dofs());
    system_rhs.reinit(agglo_handler->n_dofs());

    agglo_handler->initialize_fe_values(q_collection,
                                        update_gradients | update_JxW_values |
                                          update_quadrature_points |
                                          update_JxW_values | update_values,
                                        face_q_collection);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure_S(dim);
    const FEValuesExtractors::Scalar pressure_D(dim + 1);

    // Loop over all agglomerated polytopes (cells)
    for (const auto &polytope : agglo_handler->polytope_iterators())
      {
        const unsigned int current_dofs_per_cell =
          polytope->get_fe().dofs_per_cell;

        FullMatrix<double> cell_matrix(current_dofs_per_cell,
                                       current_dofs_per_cell);
        Vector<double>     cell_rhs(current_dofs_per_cell);
        cell_matrix = 0;
        cell_rhs    = 0;

        std::vector<types::global_dof_index> local_dof_indices(
          current_dofs_per_cell);
        polytope->get_dof_indices(local_dof_indices);

        const auto &agglo_values = agglo_handler->reinit(polytope);
        const auto &q_points     = agglo_values.get_quadrature_points();

        if (polytope_is_in_Stokes_domain(polytope)) // Stokes domain
          {
            std::vector<Tensor<1, dim>> rhs_S(q_points.size());
            rhs_function_S->value_list(q_points, rhs_S);

            std::vector<Tensor<1, dim>> S_phi_u(current_dofs_per_cell);
            std::vector<Tensor<2, dim>> S_grad_phi_u(current_dofs_per_cell);
            std::vector<double>         S_div_phi_u(current_dofs_per_cell);
            std::vector<double>         S_phi_p(current_dofs_per_cell);

            for (unsigned int q_index : agglo_values.quadrature_point_indices())
              {
                for (unsigned int k = 0; k < current_dofs_per_cell; ++k)
                  {
                    S_phi_u[k] = agglo_values[velocities].value(k, q_index);
                    S_grad_phi_u[k] =
                      agglo_values[velocities].gradient(k, q_index);
                    S_div_phi_u[k] =
                      agglo_values[velocities].divergence(k, q_index);
                    S_phi_p[k] = agglo_values[pressure_S].value(k, q_index);
                  }

                for (unsigned int i = 0; i < current_dofs_per_cell; ++i)
                  {
                    for (unsigned int j = 0; j < current_dofs_per_cell; ++j)
                      {
                        cell_matrix(i, j) +=
                          (viscosity_nu *
                             scalar_product(S_grad_phi_u[i],
                                            S_grad_phi_u[j]) // + ν ∇v:∇u
                           - S_div_phi_u[i] * S_phi_p[j]     // - ∇·v p_S
                           + S_phi_p[i] * S_div_phi_u[j]     // + ∇·u q_S
                           ) *
                          agglo_values.JxW(q_index); // dx
                        // + ν ∫ ∇v : ∇u dx
                        // -   ∫ (∇·v) p_S dx
                        // +   ∫ (∇·u) q_S dx
                      }
                    cell_rhs(i) += S_phi_u[i] * rhs_S[q_index] *
                                   agglo_values.JxW(q_index); // ∫ v·f_S dx
                  }
              }
          }
        else if (polytope_is_in_Darcy_domain(polytope)) // Darcy domain
          {
            std::vector<double> rhs_D(q_points.size());
            rhs_function_D->value_list(q_points, rhs_D);

            std::vector<double>         D_phi_p(current_dofs_per_cell);
            std::vector<Tensor<1, dim>> D_grad_phi_p(current_dofs_per_cell);

            for (unsigned int q_index : agglo_values.quadrature_point_indices())
              {
                for (unsigned int k = 0; k < current_dofs_per_cell; ++k)
                  {
                    D_phi_p[k] = agglo_values[pressure_D].value(k, q_index);
                    D_grad_phi_p[k] =
                      agglo_values[pressure_D].gradient(k, q_index);
                  }

                for (unsigned int i = 0; i < current_dofs_per_cell; ++i)
                  {
                    for (unsigned int j = 0; j < current_dofs_per_cell; ++j)
                      {
                        cell_matrix(i, j) +=
                          (permeability_K * D_grad_phi_p[i]) *
                          D_grad_phi_p[j] *          // + K ∇q_D · ∇p_D
                          agglo_values.JxW(q_index); // dx
                                                     // + ∫ K ∇q_D · ∇p_D dx
                      }
                    cell_rhs(i) += D_phi_p[i] * rhs_D[q_index] *
                                   agglo_values.JxW(q_index); // + q_D f_D dx
                  }
              }
          }
        else
          Assert(false,ExcMessage(
                   "Polytope with index " + std::to_string(polytope->index()) +
                   " should belong to either Stokes or Darcy domain."));



        // Loop over faces of the current polytope
        const unsigned int n_faces = polytope->n_faces();
        for (unsigned int f = 0; f < n_faces; ++f)
          {
            if (polytope->at_boundary(f))
              {
                // Handle boundary faces
                const auto &fe_face       = agglo_handler->reinit(polytope, f);
                const auto &face_q_points = fe_face.get_quadrature_points();

                // Get normal vectors seen from each agglomeration.
                const auto &normals = fe_face.get_normal_vectors();

                if (polytope_is_in_Stokes_domain(polytope)) // Stokes domain
                  {
                    std::vector<Tensor<1, dim>> g_S(face_q_points.size());
                    bcDirichlet_S->value_list(face_q_points, g_S);

                    std::vector<Tensor<2, dim>> S_aver_grad_phi_v(
                      current_dofs_per_cell);
                    std::vector<Tensor<1, dim>> S_jump_phi_v(
                      current_dofs_per_cell);
                    std::vector<double> S_aver_phi_p(current_dofs_per_cell);
                    std::vector<double> S_jump_phi_p(current_dofs_per_cell);

                    unsigned int deg_v_current =
                      polytope->get_fe().get_sub_fe(0, 1).degree;
                    // get_sub_fe(first_component, n_selected_components)
                    double tau_cell = (viscosity_nu) * (deg_v_current + 1) *
                                      (deg_v_current + dim) /
                                      std::fabs(polytope->diameter());
                    double sigma_v = penalty_constant_v * tau_cell;

                    for (unsigned int q_index :
                         fe_face.quadrature_point_indices())
                      {
                        for (unsigned int k = 0; k < current_dofs_per_cell; ++k)
                          {
                            S_aver_grad_phi_v[k] =
                              fe_face[velocities].gradient(k, q_index);
                            S_jump_phi_v[k] =
                              fe_face[velocities].value(k, q_index);
                            S_aver_phi_p[k] =
                              fe_face[pressure_S].value(k, q_index);
                          }

                        for (unsigned int i = 0; i < current_dofs_per_cell; ++i)
                          {
                            for (unsigned int j = 0; j < current_dofs_per_cell;
                                 ++j)
                              {
                                cell_matrix(i, j) +=
                                  (-viscosity_nu * S_jump_phi_v[i] *
                                     (S_aver_grad_phi_v[j] *
                                      normals[q_index]) // - ν [v] · ({∇u} · n)
                                   -
                                   viscosity_nu * S_jump_phi_v[j] *
                                     (S_aver_grad_phi_v[i] *
                                      normals[q_index]) // - ν [u] · ({∇v} · n)
                                   + sigma_v * S_jump_phi_v[i] *
                                       S_jump_phi_v[j] // + σ_v [v] · [u]
                                   + S_aver_phi_p[j] * S_jump_phi_v[i] *
                                       normals[q_index] // + [v] · n · {p_S}
                                   - S_aver_phi_p[i] * S_jump_phi_v[j] *
                                       normals[q_index])  // - [u] · n · {q_S}
                                  * fe_face.JxW(q_index); // ds

                                // - ν ∫    [v] · ({∇u} · n) ds
                                // - ν ∫    [u] · ({∇v} · n) ds
                                // + ∫     σ_v [v] · [u] ds
                                // + ∫     [v] · n · {p_S} ds
                                // - ∫     [u] · n · {q_S} ds
                              }

                            cell_rhs(i) +=
                              (-viscosity_nu * g_S[q_index] *
                                 (S_aver_grad_phi_v[i] *
                                  normals[q_index]) // - ν g_S · ({∇v} · n)
                               + sigma_v * g_S[q_index] *
                                   S_jump_phi_v[i] // + σ_v g_S · [v]
                               - S_aver_phi_p[i] * g_S[q_index] *
                                   normals[q_index])  // - g_S · n · {q_S}
                              * fe_face.JxW(q_index); // ds

                            // - ν ∫    g_S · ({∇v} · n) ds
                            // + ∫     σ_v g_S · [v] ds
                            // - ∫     g_S · n · {q_S} ds
                          }

                        // where:
                        //   [·] = jump across the face; equals
                        //   value on the current cell at the boundary
                        //   {·} = average across the face; equals value on the
                        //   current cell at the boundary
                        //   g_S = Dirichlet data on the interface (used as
                        //   boundary value of u_S)
                        //   σ_v = penalty parameter for velocity
                      }
                  }
                else if (polytope_is_in_Darcy_domain(polytope)) // Darcy domain
                  {
                    std::vector<double> g_D(face_q_points.size());
                    bcNeumann_D->value_list(face_q_points, g_D);

                    std::vector<double> D_jump_phi_p(current_dofs_per_cell);

                    for (unsigned int q_index :
                         fe_face.quadrature_point_indices())
                      {
                        for (unsigned int k = 0; k < current_dofs_per_cell; ++k)
                          D_jump_phi_p[k] =
                            fe_face[pressure_D].value(k, q_index);

                        for (unsigned int i = 0; i < current_dofs_per_cell; ++i)
                          cell_rhs(i) +=
                            -(D_jump_phi_p[i] * g_D[q_index]) *
                            fe_face.JxW(q_index); // - ∫_F q_D g_D ds
                      }
                  }
                else
                  Assert(false,ExcMessage(
                           "Polytope with index " + std::to_string(polytope->index()) +
                           " should belong to either Stokes or Darcy domain."));
              }
            else
              {
                // Handle internal faces/interfaces
                const auto &neigh_polytope = polytope->neighbor(f);

                // This is necessary to loop over internal faces only once.
                if (polytope->index() < neigh_polytope->index())
                  {
                    const unsigned int neigh_dofs_per_cell =
                      neigh_polytope->get_fe().dofs_per_cell;

                    FullMatrix<double> M11(current_dofs_per_cell,
                                           current_dofs_per_cell);
                    FullMatrix<double> M12(current_dofs_per_cell,
                                           neigh_dofs_per_cell);
                    FullMatrix<double> M21(neigh_dofs_per_cell,
                                           current_dofs_per_cell);
                    FullMatrix<double> M22(neigh_dofs_per_cell,
                                           neigh_dofs_per_cell);
                    M11 = 0.;
                    M12 = 0.;
                    M21 = 0.;
                    M22 = 0.;
                    // During interface integrals, dofs from both
                    // adjacent cells are involved.
                    //
                    // M11 corresponds to test and trial functions both on the
                    // current cell. M12 corresponds to test functions on the
                    // current cell and trial functions on the neighbor cell.
                    // M21 and M22 correspond similarly, with test functions on
                    // the neighbor cell.
                    //
                    // When using hp::FECollection, the number of dofs may
                    // differ between cells, so M12 and M21 are generally not
                    // square.
                    //
                    //                 dof_current   dof_neighbor
                    //   dof_current       M11           M12
                    //   dof_neighbor      M21           M22

                    std::vector<types::global_dof_index>
                      local_dof_indices_neighbor(neigh_dofs_per_cell);
                    neigh_polytope->get_dof_indices(local_dof_indices_neighbor);

                    unsigned int nofn =
                      polytope->neighbor_of_agglomerated_neighbor(f);
                    const auto &fe_faces = agglo_handler->reinit_interface(
                      polytope, neigh_polytope, f, nofn);
                    const auto &fe_faces0 = fe_faces.first;
                    const auto &fe_faces1 = fe_faces.second;

                    const auto &normals = fe_faces0.get_normal_vectors();


                    std::vector<Tensor<1, dim>> D_aver_grad_phi_p(
                      current_dofs_per_cell);
                    std::vector<double> D_jump_phi_p(current_dofs_per_cell);


                    // This is an interior face within the Stokes domain
                    if (polytope_is_in_Stokes_domain(polytope) &&
                        polytope_is_in_Stokes_domain(neigh_polytope))
                      {
                        std::vector<Tensor<2, dim>> S_aver_grad_phi_v0(
                          current_dofs_per_cell);
                        std::vector<Tensor<1, dim>> S_jump_phi_v0(
                          current_dofs_per_cell);
                        std::vector<double> S_aver_phi_p0(
                          current_dofs_per_cell);
                        std::vector<double> S_jump_phi_p0(
                          current_dofs_per_cell);
                        std::vector<Tensor<2, dim>> S_aver_grad_phi_v1(
                          neigh_dofs_per_cell);
                        std::vector<Tensor<1, dim>> S_jump_phi_v1(
                          neigh_dofs_per_cell);
                        std::vector<double> S_aver_phi_p1(neigh_dofs_per_cell);
                        std::vector<double> S_jump_phi_p1(neigh_dofs_per_cell);

                        unsigned int deg_v_current =
                          polytope->get_fe().get_sub_fe(0, 1).degree;
                        unsigned int deg_v_neigh =
                          neigh_polytope->get_fe().get_sub_fe(0, 1).degree;
                        double tau_current = (viscosity_nu) *
                                             (deg_v_current + 1) *
                                             (deg_v_current + dim) /
                                             std::fabs(polytope->diameter());
                        double tau_neigh =
                          (viscosity_nu) * (deg_v_neigh + 1) *
                          (deg_v_neigh + dim) /
                          std::fabs(neigh_polytope->diameter());
                        double sigma_v =
                          penalty_constant_v * std::max(tau_current, tau_neigh);
                        double zeta_current =
                          1. / (viscosity_nu / polytope->diameter());
                        double zeta_neigh =
                          1. / (viscosity_nu / neigh_polytope->diameter());
                        double sigma_p_S = penalty_constant_p_S *
                                           std::max(zeta_current, zeta_neigh);

                        for (unsigned int q_index = 0;
                             q_index < fe_faces0.n_quadrature_points;
                             ++q_index)
                          {
                            for (unsigned int k = 0; k < current_dofs_per_cell;
                                 ++k)
                              {
                                S_aver_grad_phi_v0[k] =
                                  0.5 *
                                  fe_faces0[velocities].gradient(k, q_index);
                                S_jump_phi_v0[k] =
                                  fe_faces0[velocities].value(k, q_index);
                                S_aver_phi_p0[k] =
                                  0.5 * fe_faces0[pressure_S].value(k, q_index);
                                S_jump_phi_p0[k] =
                                  fe_faces0[pressure_S].value(k, q_index);
                              }

                            for (unsigned int k = 0; k < neigh_dofs_per_cell;
                                 ++k)
                              {
                                S_aver_grad_phi_v1[k] =
                                  0.5 *
                                  fe_faces1[velocities].gradient(k, q_index);
                                S_jump_phi_v1[k] =
                                  -fe_faces1[velocities].value(k, q_index);
                                S_aver_phi_p1[k] =
                                  0.5 * fe_faces1[pressure_S].value(k, q_index);
                                S_jump_phi_p1[k] =
                                  -fe_faces1[pressure_S].value(k, q_index);
                              }

                            for (unsigned int i = 0; i < current_dofs_per_cell;
                                 ++i)
                              {
                                for (unsigned int j = 0;
                                     j < current_dofs_per_cell;
                                     ++j)
                                  {
                                    M11(i, j) +=
                                      (-viscosity_nu * S_jump_phi_v0[i] *
                                         (S_aver_grad_phi_v0[j] *
                                          normals[q_index]) // - ν [v] · ({∇u} ·
                                                            // n)
                                       - viscosity_nu * S_jump_phi_v0[j] *
                                           (S_aver_grad_phi_v0[i] *
                                            normals[q_index]) // - ν [u] · ({∇v}
                                                              // · n)
                                       + sigma_v * S_jump_phi_v0[i] *
                                           S_jump_phi_v0[j] // + σ_v [v] · [u]
                                       + S_aver_phi_p0[j] * S_jump_phi_v0[i] *
                                           normals[q_index] // + [v] · n · {p_S}
                                       - S_aver_phi_p0[i] * S_jump_phi_v0[j] *
                                           normals[q_index] // - [u] · n · {q_S}
                                       + sigma_p_S * S_jump_phi_p0[i] *
                                           S_jump_phi_p0[j]) // + σ_p_S [p_S] ·
                                                             // [q_S]
                                      * fe_faces0.JxW(q_index); // ds

                                    // - ν ∫    [v] · ({∇u} · n) ds
                                    // - ν ∫    [u] · ({∇v} · n) ds
                                    // + ∫     σ_v [v] · [u] ds
                                    // + ∫     [v] · n · {p_S} ds
                                    // - ∫     [u] · n · {q_S} ds
                                    // + ∫     σ_p_S [p_S] · [q_S] ds
                                    //
                                    // where:
                                    //   [·]       = jump across face
                                    //   {·}       = average across face
                                    //   ∫         = integral over interior face
                                    //   σ_v       = velocity penalty parameter
                                    //   σ_p_S     = pressure penalty parameter
                                    //   for Stokes pressure
                                    //   σ_p_D     = pressure penalty parameter
                                    //   for Darcy pressure (used elsewhere)
                                    //
                                    // Note:
                                    //   S_ prefix denotes Stokes region
                                    //   Suffix '0' indicates basis functions of
                                    //   the current cell only (no neighbor)
                                  }
                              }

                            for (unsigned int i = 0; i < current_dofs_per_cell;
                                 ++i)
                              {
                                for (unsigned int j = 0;
                                     j < neigh_dofs_per_cell;
                                     ++j)
                                  {
                                    M12(i, j) +=
                                      (-viscosity_nu * S_jump_phi_v0[i] *
                                         (S_aver_grad_phi_v1[j] *
                                          normals[q_index]) -
                                       viscosity_nu * S_jump_phi_v1[j] *
                                         (S_aver_grad_phi_v0[i] *
                                          normals[q_index]) +
                                       sigma_v * S_jump_phi_v0[i] *
                                         S_jump_phi_v1[j] +
                                       S_aver_phi_p1[j] * S_jump_phi_v0[i] *
                                         normals[q_index] -
                                       S_aver_phi_p0[i] * S_jump_phi_v1[j] *
                                         normals[q_index] +
                                       sigma_p_S * S_jump_phi_p0[i] *
                                         S_jump_phi_p1[j]) *
                                      fe_faces0.JxW(q_index);
                                    // Same structure as M11; only the basis
                                    // functions differ.
                                    //
                                    // Suffix '1' refers to neighbor cell basis
                                    // functions, while suffix '0' refers to
                                    // current cell basis functions.
                                    //
                                    // Index [j] corresponds to trial functions,
                                    // and [i] to test functions.
                                    //
                                    // In M21, all [i] indices are associated
                                    // with suffix '1', indicating test
                                    // functions from the neighbor cell and
                                    // trial functions from the current cell.
                                  }
                              }

                            for (unsigned int i = 0; i < neigh_dofs_per_cell;
                                 ++i)
                              {
                                for (unsigned int j = 0;
                                     j < current_dofs_per_cell;
                                     ++j)
                                  {
                                    M21(i, j) +=
                                      (-viscosity_nu * S_jump_phi_v1[i] *
                                         (S_aver_grad_phi_v0[j] *
                                          normals[q_index]) -
                                       viscosity_nu * S_jump_phi_v0[j] *
                                         (S_aver_grad_phi_v1[i] *
                                          normals[q_index]) +
                                       sigma_v * S_jump_phi_v1[i] *
                                         S_jump_phi_v0[j] +
                                       S_aver_phi_p0[j] * S_jump_phi_v1[i] *
                                         normals[q_index] -
                                       S_aver_phi_p1[i] * S_jump_phi_v0[j] *
                                         normals[q_index] +
                                       sigma_p_S * S_jump_phi_p1[i] *
                                         S_jump_phi_p0[j]) *
                                      fe_faces0.JxW(q_index);
                                    // In M21, [j] indices use suffix '0',
                                    // indicating trial functions from the
                                    // current cell, while [i] indices use
                                    // suffix '1', indicating test functions
                                    // from the neighbor cell.
                                  }
                              }

                            for (unsigned int i = 0; i < neigh_dofs_per_cell;
                                 ++i)
                              {
                                for (unsigned int j = 0;
                                     j < neigh_dofs_per_cell;
                                     ++j)
                                  {
                                    M22(i, j) +=
                                      (-viscosity_nu * S_jump_phi_v1[i] *
                                         (S_aver_grad_phi_v1[j] *
                                          normals[q_index]) -
                                       viscosity_nu * S_jump_phi_v1[j] *
                                         (S_aver_grad_phi_v1[i] *
                                          normals[q_index]) +
                                       sigma_v * S_jump_phi_v1[i] *
                                         S_jump_phi_v1[j] +
                                       S_aver_phi_p1[j] * S_jump_phi_v1[i] *
                                         normals[q_index] -
                                       S_aver_phi_p1[i] * S_jump_phi_v1[j] *
                                         normals[q_index] +
                                       sigma_p_S * S_jump_phi_p1[i] *
                                         S_jump_phi_p1[j]) *
                                      fe_faces0.JxW(q_index);
                                    // In M22, both test and trial functions use
                                    // suffix '1', meaning they are associated
                                    // with the neighbor cell only.
                                  }
                              }
                          }
                      }

                    // This is an interior face within the Darcy domain
                    if (polytope_is_in_Darcy_domain(polytope) &&
                        polytope_is_in_Darcy_domain(neigh_polytope))
                      {
                        std::vector<Tensor<1, dim>> D_aver_grad_phi_p0(
                          current_dofs_per_cell);
                        std::vector<double> D_jump_phi_p0(
                          current_dofs_per_cell);
                        std::vector<Tensor<1, dim>> D_aver_grad_phi_p1(
                          neigh_dofs_per_cell);
                        std::vector<double> D_jump_phi_p1(neigh_dofs_per_cell);

                        unsigned int deg_p_current =
                          polytope->get_fe().get_sub_fe(3, 1).degree;
                        unsigned int deg_p_neigh =
                          neigh_polytope->get_fe().get_sub_fe(3, 1).degree;
                        double tau_current = (permeability_scalar) *
                                             (deg_p_current + 1) *
                                             (deg_p_current + dim) /
                                             std::fabs(polytope->diameter());
                        double tau_neigh =
                          (permeability_scalar) * (deg_p_neigh + 1) *
                          (deg_p_neigh + dim) /
                          std::fabs(neigh_polytope->diameter());
                        double sigma_p_D = penalty_constant_p_D *
                                           std::max(tau_current, tau_neigh);

                        for (unsigned int q_index = 0;
                             q_index < fe_faces0.n_quadrature_points;
                             ++q_index)
                          {
                            for (unsigned int k = 0; k < current_dofs_per_cell;
                                 ++k)
                              {
                                D_aver_grad_phi_p0[k] =
                                  0.5 *
                                  fe_faces0[pressure_D].gradient(k, q_index);
                                D_jump_phi_p0[k] =
                                  fe_faces0[pressure_D].value(k, q_index);
                              }

                            for (unsigned int k = 0; k < neigh_dofs_per_cell;
                                 ++k)
                              {
                                D_aver_grad_phi_p1[k] =
                                  0.5 *
                                  fe_faces1[pressure_D].gradient(k, q_index);
                                D_jump_phi_p1[k] =
                                  -fe_faces1[pressure_D].value(k, q_index);
                              }

                            for (unsigned int i = 0; i < current_dofs_per_cell;
                                 ++i)
                              {
                                for (unsigned int j = 0;
                                     j < current_dofs_per_cell;
                                     ++j)
                                  {
                                    M11(i, j) +=
                                      (-D_jump_phi_p0[i] *
                                         D_aver_grad_phi_p0[j] *
                                         permeability_K * normals[q_index] -
                                       D_jump_phi_p0[j] *
                                         D_aver_grad_phi_p0[i] *
                                         permeability_K * normals[q_index] +
                                       sigma_p_D * D_jump_phi_p0[i] *
                                         D_jump_phi_p0[j]) *
                                      fe_faces0.JxW(q_index);
                                  }
                              }

                            for (unsigned int i = 0; i < current_dofs_per_cell;
                                 ++i)
                              {
                                for (unsigned int j = 0;
                                     j < neigh_dofs_per_cell;
                                     ++j)
                                  {
                                    M12(i, j) +=
                                      (-D_jump_phi_p0[i] *
                                         D_aver_grad_phi_p1[j] *
                                         permeability_K * normals[q_index] -
                                       D_jump_phi_p1[j] *
                                         D_aver_grad_phi_p0[i] *
                                         permeability_K * normals[q_index] +
                                       sigma_p_D * D_jump_phi_p0[i] *
                                         D_jump_phi_p1[j]) *
                                      fe_faces0.JxW(q_index);
                                  }
                              }

                            for (unsigned int i = 0; i < neigh_dofs_per_cell;
                                 ++i)
                              {
                                for (unsigned int j = 0;
                                     j < current_dofs_per_cell;
                                     ++j)
                                  {
                                    M21(i, j) +=
                                      (-D_jump_phi_p1[i] *
                                         D_aver_grad_phi_p0[j] *
                                         permeability_K * normals[q_index] -
                                       D_jump_phi_p0[j] *
                                         D_aver_grad_phi_p1[i] *
                                         permeability_K * normals[q_index] +
                                       sigma_p_D * D_jump_phi_p1[i] *
                                         D_jump_phi_p0[j]) *
                                      fe_faces0.JxW(q_index);
                                  }
                              }

                            for (unsigned int i = 0; i < neigh_dofs_per_cell;
                                 ++i)
                              {
                                for (unsigned int j = 0;
                                     j < neigh_dofs_per_cell;
                                     ++j)
                                  {
                                    M22(i, j) +=
                                      (-D_jump_phi_p1[i] *
                                         D_aver_grad_phi_p1[j] *
                                         permeability_K * normals[q_index] -
                                       D_jump_phi_p1[j] *
                                         D_aver_grad_phi_p1[i] *
                                         permeability_K * normals[q_index] +
                                       sigma_p_D * D_jump_phi_p1[i] *
                                         D_jump_phi_p1[j]) *
                                      fe_faces0.JxW(q_index);
                                  }
                              }
                          }
                      }

                    // This is an interface between Stokes and Darcy domains
                    if ((polytope_is_in_Stokes_domain(polytope) &&
                         polytope_is_in_Darcy_domain(neigh_polytope)) ||
                        (polytope_is_in_Darcy_domain(polytope) &&
                         polytope_is_in_Stokes_domain(neigh_polytope)))
                      {
                        bool is_stokes_side =
                          polytope_is_in_Stokes_domain(polytope);
                        Tensor<1, dim> normal;
                        Tensor<1, dim> tangential;

                        std::vector<Tensor<1, dim>> S_phi_v0(
                          current_dofs_per_cell);
                        std::vector<double> D_phi_p0(current_dofs_per_cell);
                        std::vector<Tensor<1, dim>> S_phi_v1(
                          neigh_dofs_per_cell);
                        std::vector<double> D_phi_p1(neigh_dofs_per_cell);

                        for (unsigned int q_index = 0;
                             q_index < fe_faces0.n_quadrature_points;
                             ++q_index)
                          {
                            if (is_stokes_side)
                              normal = normals[q_index];
                            else
                              normal = -normals[q_index];
                            tangential[0] = -normal[1];
                            tangential[1] = normal[0];

                            for (unsigned int k = 0; k < current_dofs_per_cell;
                                 ++k)
                              {
                                S_phi_v0[k] =
                                  fe_faces0[velocities].value(k, q_index);
                                D_phi_p0[k] =
                                  fe_faces0[pressure_D].value(k, q_index);
                              }

                            for (unsigned int k = 0; k < neigh_dofs_per_cell;
                                 ++k)
                              {
                                S_phi_v1[k] =
                                  fe_faces1[velocities].value(k, q_index);
                                D_phi_p1[k] =
                                  fe_faces1[pressure_D].value(k, q_index);
                              }

                            for (unsigned int i = 0; i < current_dofs_per_cell;
                                 ++i)
                              {
                                for (unsigned int j = 0;
                                     j < current_dofs_per_cell;
                                     ++j)
                                  {
                                    M11(i, j) +=
                                      (D_phi_p0[j] * S_phi_v0[i] * normal -
                                       D_phi_p0[i] * S_phi_v0[j] * normal +
                                       nu_over_G * (S_phi_v0[i] * tangential) *
                                         (S_phi_v0[j] * tangential)) *
                                      fe_faces0.JxW(q_index);
                                  }
                              }

                            for (unsigned int i = 0; i < current_dofs_per_cell;
                                 ++i)
                              {
                                for (unsigned int j = 0;
                                     j < neigh_dofs_per_cell;
                                     ++j)
                                  {
                                    M12(i, j) +=
                                      (D_phi_p1[j] * S_phi_v0[i] * normal -
                                       D_phi_p0[i] * S_phi_v1[j] * normal +
                                       nu_over_G * (S_phi_v0[i] * tangential) *
                                         (S_phi_v1[j] * tangential)) *
                                      fe_faces0.JxW(q_index);
                                  }
                              }

                            for (unsigned int i = 0; i < neigh_dofs_per_cell;
                                 ++i)
                              {
                                for (unsigned int j = 0;
                                     j < current_dofs_per_cell;
                                     ++j)
                                  {
                                    M21(i, j) +=
                                      (D_phi_p0[j] * S_phi_v1[i] * normal -
                                       D_phi_p1[i] * S_phi_v0[j] * normal +
                                       nu_over_G * (S_phi_v1[i] * tangential) *
                                         (S_phi_v0[j] * tangential)) *
                                      fe_faces0.JxW(q_index);
                                  }
                              }

                            for (unsigned int i = 0; i < neigh_dofs_per_cell;
                                 ++i)
                              {
                                for (unsigned int j = 0;
                                     j < neigh_dofs_per_cell;
                                     ++j)
                                  {
                                    M22(i, j) +=
                                      (D_phi_p1[j] * S_phi_v1[i] * normal -
                                       D_phi_p1[i] * S_phi_v1[j] * normal +
                                       nu_over_G * (S_phi_v1[i] * tangential) *
                                         (S_phi_v1[j] * tangential)) *
                                      fe_faces0.JxW(q_index);
                                  }
                              }
                          }
                      }


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
        constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
      }
  }

  template <int dim>
  void
  StokesDarcyProblem<dim>::solve()
  {
    constraints.condense(system_matrix);
    constraints.condense(system_rhs);

    SparseDirectUMFPACK A_direct;
    A_direct.initialize(system_matrix);
    A_direct.vmult(solution, system_rhs);

    constraints.distribute(solution);

    std::cout << "   Interpolating..." << std::endl;
    PolyUtils::interpolate_to_fine_grid(*agglo_handler,
                                        interpolated_solution,
                                        solution,
                                        true /*on_the_fly*/);
  }

  template <int dim>
  void
  StokesDarcyProblem<dim>::zero_pressure_dof_constraint()
  {
    const FEValuesExtractors::Scalar pressure_S(dim);
    ComponentMask  pressure_S_mask = fe_collection.component_mask(pressure_S);
    const IndexSet pressure_dofs =
      DoFTools::extract_dofs(agglo_handler->agglo_dh, pressure_S_mask);
    const types::global_dof_index first_pressure_dof =
      pressure_dofs.nth_index_in_set(0);
    constraints.constrain_dof_to_zero(first_pressure_dof);
  }

  template <int dim>
  void
  StokesDarcyProblem<dim>::mean_pressure_to_zero()
  {
    const ComponentSelectFunction<dim> pressure_S_mask(dim, dim + 2);
    const ComponentSelectFunction<dim> pressure_D_mask(dim + 1, dim + 2);

    Vector<double> integral_per_cell(triangulation.n_active_cells());

    VectorTools::integrate_difference(agglo_handler->output_dh,
                                      interpolated_solution,
                                      Functions::ZeroFunction<dim>(dim + 2),
                                      integral_per_cell,
                                      q_collection,
                                      VectorTools::mean,
                                      &pressure_S_mask);
    const double global_pressure_S_integral =
      -VectorTools::compute_global_error(triangulation,
                                         integral_per_cell,
                                         VectorTools::mean);
    VectorTools::integrate_difference(agglo_handler->output_dh,
                                      interpolated_solution,
                                      Functions::ZeroFunction<dim>(dim + 2),
                                      integral_per_cell,
                                      q_collection,
                                      VectorTools::mean,
                                      &pressure_D_mask);
    const double global_pressure_D_integral =
      -VectorTools::compute_global_error(triangulation,
                                         integral_per_cell,
                                         VectorTools::mean);

    const double mean_pressure =
      (global_pressure_S_integral + global_pressure_D_integral) / domain_area;

    for (const auto &cell : agglo_handler->output_dh.active_cell_iterators())
      {
        const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        cell->get_dof_indices(local_dof_indices);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            const unsigned int dof_component =
              cell->get_fe().system_to_component_index(i).first;
            if (dof_component == dim)
              interpolated_solution[local_dof_indices[i]] -= mean_pressure;
            else if (dof_component == dim + 1)
              interpolated_solution[local_dof_indices[i]] -= mean_pressure;
          }
      }
  }

  template <int dim>
  void
  StokesDarcyProblem<dim>::compute_errors()
  {
    Vector<float> difference_per_cell(triangulation.n_active_cells());
    const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim),
                                                     dim + 2);
    const ComponentSelectFunction<dim> pressure_S_mask(dim, dim + 2);
    const ComponentSelectFunction<dim> pressure_D_mask(dim + 1, dim + 2);

    VectorTools::integrate_difference(agglo_handler->output_dh,
                                      interpolated_solution,
                                      *exact_solution,
                                      difference_per_cell,
                                      q_collection,
                                      VectorTools::L2_norm,
                                      &velocity_mask);
    error_velocity_L2 = VectorTools::compute_global_error(triangulation,
                                                          difference_per_cell,
                                                          VectorTools::L2_norm);

    VectorTools::integrate_difference(agglo_handler->output_dh,
                                      interpolated_solution,
                                      *exact_solution,
                                      difference_per_cell,
                                      q_collection,
                                      VectorTools::H1_norm,
                                      &velocity_mask);
    error_velocity_H1 = VectorTools::compute_global_error(triangulation,
                                                          difference_per_cell,
                                                          VectorTools::H1_norm);

    VectorTools::integrate_difference(agglo_handler->output_dh,
                                      interpolated_solution,
                                      *exact_solution,
                                      difference_per_cell,
                                      q_collection,
                                      VectorTools::L2_norm,
                                      &pressure_S_mask);
    error_pressure_S = VectorTools::compute_global_error(triangulation,
                                                         difference_per_cell,
                                                         VectorTools::L2_norm);

    VectorTools::integrate_difference(agglo_handler->output_dh,
                                      interpolated_solution,
                                      *exact_solution,
                                      difference_per_cell,
                                      q_collection,
                                      VectorTools::L2_norm,
                                      &pressure_D_mask);
    error_pressure_D = VectorTools::compute_global_error(triangulation,
                                                         difference_per_cell,
                                                         VectorTools::L2_norm);

    std::cout << "     velocity L2 Error: " << error_velocity_L2 << std::endl
              << "     velocity H1 Error: " << error_velocity_H1 << std::endl
              << "     pressure_S L2 Error: " << error_pressure_S << std::endl
              << "     pressure_D L2 Error: " << error_pressure_D << std::endl;
  }

  template <int dim>
  void
  StokesDarcyProblem<dim>::output_results(unsigned int n_subdomains) const
  {
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("pressure_S");
    solution_names.emplace_back("pressure_D");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(agglo_handler->output_dh);

    data_out.add_data_vector(interpolated_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    Vector<float> agglo_idx(triangulation.n_active_cells());
    for (const auto &polytope : agglo_handler->polytope_iterators())
      {
        const types::global_cell_index polytope_index = polytope->index();
        const auto &patch_of_cells = polytope->get_agglomerate(); // fine cells
        // Flag them
        for (const auto &cell : patch_of_cells)
          agglo_idx[cell->active_cell_index()] = polytope_index;
      }
    data_out.add_data_vector(agglo_idx,
                             "agglo_idx",
                             DataOut<dim>::type_cell_data);

    data_out.build_patches();

    std::ofstream output("solution-" + Utilities::int_to_string(n_subdomains) +
                         ".vtk");

    data_out.write_vtk(output);

    std::cout << "     Solution written to file: "
              << "solution-" + Utilities::int_to_string(n_subdomains) + ".vtk"
              << std::endl;
  }

  template <int dim>
  void
  StokesDarcyProblem<dim>::run()
  {
    std::cout << "   Making grid..." << std::endl;
    make_agglo_grid();

    std::cout << "   Setting up agglomeration..." << std::endl;
    setup_agglomeration();

    std::cout << "   Assembling..." << std::endl;
    assemble_system();

    std::cout << "   Solving..." << std::endl;
    solve();

    std::cout << "   Modifying pressure..." << std::endl;
    mean_pressure_to_zero();

    std::cout << "   Error:" << std::endl;
    compute_errors();

    std::cout << "   Writing output..." << std::endl;
    output_results(n_subdomains);

    std::cout << std::endl;
  }

  template <int dim>
  double
  StokesDarcyProblem<dim>::get_error_velocity_L2() const
  {
    return error_velocity_L2;
  }

  template <int dim>
  double
  StokesDarcyProblem<dim>::get_error_velocity_H1() const
  {
    return error_velocity_H1;
  }

  template <int dim>
  double
  StokesDarcyProblem<dim>::get_error_pressure_S() const
  {
    return error_pressure_S;
  }

  template <int dim>
  double
  StokesDarcyProblem<dim>::get_error_pressure_D() const
  {
    return error_pressure_D;
  }

  template <int dim>
  unsigned int
  StokesDarcyProblem<dim>::get_n_dofs() const
  {
    return agglo_handler->n_dofs();
  }
} // namespace StokesDarcyNamespace

int
main()
{
  try
    {
      using namespace StokesDarcyNamespace;

      ConvergenceTable   convergence_table;
      const unsigned int deg_v   = 3;
      const unsigned int deg_p_S = 2;
      const unsigned int deg_p_D = 2;

      for (unsigned int mesh_level = 2; mesh_level < 7; ++mesh_level)
        {
          std::cout << "Mesh level " << mesh_level << std::endl;
          convergence_table.add_value("level", mesh_level);
          convergence_table.add_value("polytopes",
                                      (int)std::pow(4, mesh_level));

          if (mesh_level < 7)
            {
              StokesDarcyProblem<2> SD_problem(deg_v,
                                               deg_p_S,
                                               deg_p_D,
                                               mesh_level);

              SD_problem.run();
              convergence_table.add_value("dofs", SD_problem.get_n_dofs());
              convergence_table.add_value("velocity_L2",
                                          SD_problem.get_error_velocity_L2());
              convergence_table.add_value("velocity_H1",
                                          SD_problem.get_error_velocity_H1());
              convergence_table.add_value("pressure_S",
                                          SD_problem.get_error_pressure_S());
              convergence_table.add_value("pressure_D",
                                          SD_problem.get_error_pressure_D());
            }
          else
            AssertThrow(
              false,
              ExcMessage(
                "You need to refine the base grid to use higher mesh levels."
                "Please modify make_base_grid()."));
        }

      convergence_table.set_precision("velocity_L2", 3);
      convergence_table.set_precision("velocity_H1", 3);
      convergence_table.set_precision("pressure_S", 3);
      convergence_table.set_precision("pressure_D", 3);
      convergence_table.set_scientific("velocity_L2", true);
      convergence_table.set_scientific("velocity_H1", true);
      convergence_table.set_scientific("pressure_S", true);
      convergence_table.set_scientific("pressure_D", true);

      convergence_table.evaluate_convergence_rates(
        "velocity_L2", "polytopes", ConvergenceTable::reduction_rate_log2, 2);
      convergence_table.evaluate_convergence_rates(
        "velocity_H1", "polytopes", ConvergenceTable::reduction_rate_log2, 2);
      convergence_table.evaluate_convergence_rates(
        "pressure_S", "polytopes", ConvergenceTable::reduction_rate_log2, 2);
      convergence_table.evaluate_convergence_rates(
        "pressure_D", "polytopes", ConvergenceTable::reduction_rate_log2, 2);
      std::cout << "(deg_v, deg_p_S, deg_p_D) = (" << deg_v << ", " << deg_p_S
                << ", " << deg_p_D << ")," << std::endl;
      convergence_table.write_text(std::cout);
      std::ofstream output("convergence_table.vtk");
      convergence_table.write_text(output);
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
