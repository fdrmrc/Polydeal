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
   Oseen Problem Formulation (steady, incompressible)

   Given a domain Ω ⊂ ℝ^d, find a velocity field u : Ω → ℝ^d
   and a pressure field p : Ω → ℝ such that:

       -ν Δu + (β · ∇)u + ∇p = f      in Ω       (momentum equation)
                        ∇ · u = 0      in Ω       (mass conservation)
                             u = g     on ∂Ω       (Dirichlet boundary)

   where:
       - u: velocity vector field
       - p: pressure scalar field, with ∫_Ω p dx = 0
       - f: external body force (RightHandSide)
       - g: prescribed boundary velocity (BoundaryDirichlet)
       - ν: kinematic viscosity
       - β: prescribed convection (advection) velocity field (BetaFunction)
       - Δu: Laplacian of u, i.e., component-wise second derivatives
============================================================================*/

namespace OseenNamespace
{
  using namespace dealii;

  /*============================================================================
     ExactSolution (Kovasznay flow)
     Represents the analytical solution (u, p) to the Oseen problem.

     Re: Reynolds number
     λ = Re/2 - sqrt(Re²/4 + 4π²)
     ν = 1 / Re

     The exact solution is:
       u₁(x,y) = 1 - exp(λx) * cos(2πy)
       u₂(x,y) = (λ / 2π) * exp(λx) * sin(2πy)
       p(x,y)  = 0.5 * exp(2λx) + C, where C ensures ∫_Ω p dx = 0

  --- Reference ---
    B. Cockburn, G. Kanschat, and D. Schötzau,
    "The Local Discontinuous Galerkin Method for the Oseen Equations",
    Mathematics of Computation, Vol. 73, No. 246, pp. 569–593, 2003.
  ============================================================================*/
  template <int dim>
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution(const double Re_in = 10.0);

    virtual void
    vector_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>>   &value_list) const override;

    virtual void
    vector_gradient_list(
      const std::vector<Point<dim>>            &points,
      std::vector<std::vector<Tensor<1, dim>>> &gradient_list) const override;

  private:
    static constexpr double PI = numbers::PI;
    const double            Re;
    const double            lambda;
    const double            mean_pressure;
  };

  template <int dim>
  ExactSolution<dim>::ExactSolution(const double Re_in)
    : Function<dim>(dim + 1)
    , Re(Re_in)
    , lambda(Re / 2. - std::sqrt(Re * Re / 4. + 4. * PI * PI))
    , mean_pressure(1. / (8. * lambda) *
                    (std::exp(3. * lambda) - std::exp(-lambda)))
  {}

  template <int dim>
  void
  ExactSolution<dim>::vector_value_list(
    const std::vector<Point<dim>> &points,
    std::vector<Vector<double>>   &value_list) const
  {
    using std::cos;
    using std::exp;
    using std::sin;

    AssertDimension(points.size(), value_list.size());

    for (unsigned int i = 0; i < points.size(); ++i)
      {
        const double x = points[i][0];
        const double y = points[i][1];

        value_list[i][0] = 1. - exp(lambda * x) * cos(2. * PI * y);
        value_list[i][1] =
          lambda / (2. * PI) * exp(lambda * x) * sin(2. * PI * y);
        value_list[i][2] = 0.5 * exp(2. * lambda * x) - mean_pressure;
      }
  }

  template <int dim>
  void
  ExactSolution<dim>::vector_gradient_list(
    const std::vector<Point<dim>>            &points,
    std::vector<std::vector<Tensor<1, dim>>> &gradient_list) const
  {
    using std::cos;
    using std::exp;
    using std::sin;

    AssertDimension(points.size(), gradient_list.size());

    for (unsigned int i = 0; i < points.size(); ++i)
      {
        const double x = points[i][0];
        const double y = points[i][1];

        gradient_list[i][0][0] = -lambda * exp(lambda * x) * cos(2. * PI * y);
        gradient_list[i][0][1] = 2. * PI * exp(lambda * x) * sin(2. * PI * y);

        gradient_list[i][1][0] =
          lambda * lambda / (2. * PI) * exp(lambda * x) * sin(2. * PI * y);
        gradient_list[i][1][1] = lambda * exp(lambda * x) * cos(2. * PI * y);

        gradient_list[i][2][0] = lambda * exp(2. * lambda * x);
        gradient_list[i][2][1] = 0.;
      }
  }

  /*============================================================================
     RightHandSide
     Defines the body force f in the momentum equation, derived from the exact
     Kovasznay solution.
  ============================================================================*/
  template <int dim>
  class RightHandSide : public TensorFunction<1, dim, double>
  {
  public:
    RightHandSide(const double Re = 10.0);

    void
    value_list(const std::vector<Point<dim>> &points,
               std::vector<Tensor<1, dim>>   &value_list) const override;

  private:
    static constexpr double PI = numbers::PI;
    const double            Re;
    const double            nu;
    const double            lambda;
  };

  template <int dim>
  RightHandSide<dim>::RightHandSide(const double Re_in)
    : TensorFunction<1, dim, double>()
    , Re(Re_in)
    , nu(1. / Re)
    , lambda(Re / 2. - sqrt(Re * Re / 4. + 4. * PI * PI))
  {}

  template <int dim>
  void
  RightHandSide<dim>::value_list(const std::vector<Point<dim>> &points,
                                 std::vector<Tensor<1, dim>> &value_list) const
  {
    using std::cos;
    using std::exp;
    using std::pow;
    using std::sin;
    using std::sqrt;
    AssertDimension(points.size(), value_list.size());

    for (unsigned int i = 0; i < points.size(); ++i)
      {
        const auto  &p = points[i];
        const double x = p[0];
        const double y = p[1];

        value_list[i][0] =
          nu * (lambda * lambda - 4. * PI * PI) * exp(lambda * x) *
            cos(2. * PI * y) +
          lambda * exp(lambda * x) * (2. * exp(lambda * x) - cos(2. * PI * y));

        value_list[i][1] =
          nu * (2. * PI * lambda - pow(lambda, 3.) / (2. * PI)) *
            exp(lambda * x) * sin(2. * PI * y) +
          lambda * lambda / (2. * PI) * exp(lambda * x) * sin(2. * PI * y);
      }
  }

  /*============================================================================
     BoundaryDirichlet
     Defines the Dirichlet boundary condition g = u_exact for velocity on ∂Ω.
  ============================================================================*/
  template <int dim>
  class BoundaryDirichlet : public TensorFunction<1, dim, double>
  {
  public:
    BoundaryDirichlet(const double Re = 10.0);

    void
    value_list(const std::vector<Point<dim>> &points,
               std::vector<Tensor<1, dim>>   &value_list) const override;

  private:
    static constexpr double PI = numbers::PI;
    const double            Re;
    const double            lambda;
  };

  template <int dim>
  BoundaryDirichlet<dim>::BoundaryDirichlet(const double Re_in)
    : TensorFunction<1, dim, double>()
    , Re(Re_in)
    , lambda(Re / 2. - sqrt(Re * Re / 4. + 4. * PI * PI))
  {}

  template <int dim>
  void
  BoundaryDirichlet<dim>::value_list(
    const std::vector<Point<dim>> &points,
    std::vector<Tensor<1, dim>>   &value_list) const
  {
    using std::cos;
    using std::exp;
    using std::sin;
    using std::sqrt;

    AssertDimension(points.size(), value_list.size());

    for (unsigned int i = 0; i < points.size(); ++i)
      {
        const auto  &p = points[i];
        const double x = p[0];
        const double y = p[1];

        value_list[i][0] = 1. - exp(lambda * x) * cos(2. * PI * y);
        value_list[i][1] =
          lambda / (2. * PI) * exp(lambda * x) * sin(2. * PI * y);
      }
  }

  /*============================================================================
   BetaFunction
   Defines the advection field β used in the convective term (β · ∇)u.
   In this formulation, β is set to the exact velocity field: β = u_exact.
  ============================================================================*/
  template <int dim>
  class BetaFunction : public TensorFunction<1, dim, double>
  {
  public:
    BetaFunction(const double Re = 10.0);

    void
    value_list(const std::vector<Point<dim>> &points,
               std::vector<Tensor<1, dim>>   &value_list) const override;

  private:
    static constexpr double PI = numbers::PI;
    const double            Re;
    const double            lambda;
  };

  template <int dim>
  BetaFunction<dim>::BetaFunction(const double Re_in)
    : TensorFunction<1, dim, double>()
    , Re(Re_in)
    , lambda(Re / 2. - sqrt(Re * Re / 4. + 4. * PI * PI))
  {}

  template <int dim>
  void
  BetaFunction<dim>::value_list(const std::vector<Point<dim>> &points,
                                std::vector<Tensor<1, dim>>   &value_list) const
  {
    using std::cos;
    using std::exp;
    using std::sin;
    using std::sqrt;

    AssertDimension(points.size(), value_list.size());

    for (unsigned int i = 0; i < points.size(); ++i)
      {
        const auto  &p = points[i];
        const double x = p[0];
        const double y = p[1];

        value_list[i][0] = 1. - exp(lambda * x) * cos(2. * PI * y);
        value_list[i][1] =
          lambda / (2. * PI) * exp(lambda * x) * sin(2. * PI * y);
      }
  }

  /*============================================================================
    The OseenProblem class below solves the Kovasznay flow problem
    using the Interior Penalty Discontinuous Galerkin (IPDG) method.

    The computational domain Ω = (-1/2, 3/2) × (0, 2) is initially divided
    into four subdomains separated by curved interfaces. These subdomains are
    then refined and agglomerated into a polygonal mesh whose cells may have
    curved faces.

    The left two subdomains share one finite element space,
    while the right two share another. Polynomial degrees of the finite element
    spaces can be set independently for each region.
  ============================================================================*/
  template <int dim>
  class OseenProblem
  {
  public:
    OseenProblem(const unsigned int degree_velocities_left  = 2,
                 const unsigned int degree_pressure_left    = 1,
                 const unsigned int degree_velocities_right = 2,
                 const unsigned int degree_pressure_right   = 1,
                 const unsigned int extraction_level        = 0,
                 const double       Re                      = 10.0);

    // Run the simulation
    void
    run();

    // Get error norms
    double
    get_error_velocity_L2() const;
    double
    get_error_velocity_H1() const;
    double
    get_error_pressure() const;

    // Get number of degrees of freedom
    unsigned int
    get_n_dofs() const;

  private:
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
    const unsigned int degree_velocities_left;
    const unsigned int degree_pressure_left;
    const unsigned int degree_velocities_right;
    const unsigned int degree_pressure_right;
    const unsigned int extraction_level;
    unsigned int n_subdomains; // Number of subdomains after agglomeration.

    // Physical parameters and domain info
    const double Re;
    const double viscosity_nu;
    double       domain_area; // 2x2 square
    unsigned int num_domain;  // Number of distinct domains.
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
    std::unique_ptr<const TensorFunction<1, dim, double>> rhs_function;
    std::unique_ptr<const TensorFunction<1, dim, double>> bcDirichlet;
    std::unique_ptr<const TensorFunction<1, dim, double>> beta_function;

    Vector<double> interpolated_solution;

    double error_velocity_L2;
    double error_velocity_H1;
    double error_pressure;
  };

  template <int dim>
  OseenProblem<dim>::OseenProblem(const unsigned int degree_velocities_left,
                                  const unsigned int degree_pressure_left,
                                  const unsigned int degree_velocities_right,
                                  const unsigned int degree_pressure_right,
                                  const unsigned int extraction_level,
                                  const double       Re)
    : degree_velocities_left(degree_velocities_left)
    , degree_pressure_left(degree_pressure_left)
    , degree_velocities_right(degree_velocities_right)
    , degree_pressure_right(degree_pressure_right)
    , extraction_level(extraction_level)
    , Re(Re)
    , viscosity_nu(1. / Re)
    , triangulation(Triangulation<dim>::maximum_smoothing)
    , mapping()
  {
    FESystem<dim> Oseen_fe1(FE_AggloDGP<dim>(degree_velocities_left) ^ dim,
                            FE_AggloDGP<dim>(degree_pressure_left));
    FESystem<dim> Oseen_fe2(FE_AggloDGP<dim>(degree_velocities_right) ^ dim,
                            FE_AggloDGP<dim>(degree_pressure_right));
    fe_collection.push_back(Oseen_fe1);
    fe_collection.push_back(Oseen_fe2);

    const QGauss<dim> quadrature1(degree_velocities_left);
    const QGauss<dim> quadrature2(degree_velocities_right);
    q_collection.push_back(quadrature1);
    q_collection.push_back(quadrature2);
    const QGauss<dim - 1> face_quadrature1(degree_velocities_left + 1);
    const QGauss<dim - 1> face_quadrature2(degree_velocities_right + 1);
    face_q_collection.push_back(face_quadrature1);
    face_q_collection.push_back(face_quadrature2);

    exact_solution = std::make_unique<const ExactSolution<dim>>(Re);
    rhs_function   = std::make_unique<const RightHandSide<dim>>(Re);
    bcDirichlet    = std::make_unique<const BoundaryDirichlet<dim>>(Re);
    beta_function  = std::make_unique<const BetaFunction<dim>>(Re);
  }

  /*============================================================================
  Create a fine background grid for cell agglomeration.

  This function first checks if a cached coarse grid exists on disk.
  If found, it loads the cached grid and sets manifold descriptions on
  selected faces to enable curved interfaces. If not found, it creates
  a new coarse grid subdivided into 2x2 cells on the domain
  Ω = (-0.5, 1.5) × (0, 2), assigns material IDs to partition the domain
  into four subdomains, sets manifold IDs on internal faces to define curved
  interfaces, globally refines the grid 6 times, then saves the grid to disk.
============================================================================*/
  template <int dim>
  void
  OseenProblem<dim>::make_base_grid()
  {
    Point<2>                  bottom_left(-0.5, 0.0);
    Point<2>                  top_right(1.5, 2.0);
    std::vector<unsigned int> subdivisions = {2, 2};
    GridGenerator::subdivided_hyper_rectangle(triangulation,
                                              subdivisions,
                                              bottom_left,
                                              top_right);
    domain_area = 4.0;
    num_domain  = 4;

    const std::string filename = "cached_base_grid_Oseen";

    /* The following code defines the relationship between manifold IDs and
      their corresponding manifolds, the association between edges and manifold
      IDs, and the mapping from subdomains to material IDs.

            y ↑
              |              P3
          2.0 +---------+----●----+
              |          \        |
           P1 ●   M0   E1|   M1   |
              |  ,---、  /        |
          1.0 +-' E4  `-+-、 E3 ,-+
              |        /   `---'  |
              |   M2   |E2   M3   ● P2
              |        \          |
          0.0 +----●----+----+----+---→ x
            -0.5   P4  0.5       1.5

      Each manifold represents a circular arc defined by a specific center
      point. The mapping from manifold IDs to their corresponding centers is:
      Manifold ID → center:
        1 → P1(-0.5, 1.5)    2 → P2(1.5, 0.5)    3 → P3(1., 2.)
          4 → P4(0., 0.)

      The mapping from curved interfaces to the associated manifold IDs is:
      edge → Manifold ID:
        E1 → 1    E2 → 2    E3 → 3    E4 → 4

      The mapping from subdomains to the corresponding material IDs is:
        M0 → 0 (left top)    M1 → 1 (right top)    M2 → 2 (left bottom)
        M3 → 3 (right bottom)
    */

    if (std::filesystem::exists(filename + "_triangulation.data"))
      {
        std::cout << "     Loading cached base grid from " << filename << " ..."
                  << std::endl;
        triangulation.load(filename);
        triangulation.set_manifold(
          1, PolarManifold<2>(Point<2>(-0.5, 1.5))); // left top
        triangulation.set_manifold(
          2, PolarManifold<2>(Point<2>(1.5, 0.5))); // right bottom
        triangulation.set_manifold(
          3, PolarManifold<2>(Point<2>(1., 2.))); // right top
        triangulation.set_manifold(
          4, PolarManifold<2>(Point<2>(0., 0.))); // left bottom
      }
    else
      {
        std::cout << "     Cached base grid not found. Generating new grid..."
                  << std::endl;

        triangulation.set_manifold(1, PolarManifold<2>(Point<2>(-0.5, 1.5)));
        triangulation.set_manifold(2, PolarManifold<2>(Point<2>(1.5, 0.5)));
        triangulation.set_manifold(3, PolarManifold<2>(Point<2>(1., 2.)));
        triangulation.set_manifold(4, PolarManifold<2>(Point<2>(0., 0.)));
        for (const auto &cell : triangulation.active_cell_iterators())
          {
            if (cell->center()[0] < 0.5 && cell->center()[1] > 1.)
              cell->set_material_id(0);
            if (cell->center()[0] > 0.5 && cell->center()[1] > 1.)
              cell->set_material_id(1);
            if (cell->center()[0] < 0.5 && cell->center()[1] < 1.)
              cell->set_material_id(2);
            if (cell->center()[0] > 0.5 && cell->center()[1] < 1.)
              cell->set_material_id(3);

            for (unsigned int f = 0; f < 4; ++f)
              if (!cell->at_boundary(f))
                {
                  if ((cell->face(f)->center()[0] > 0.1) &&
                      (cell->face(f)->center()[0] < 0.9))
                    {
                      if (cell->face(f)->center()[1] > 1.)
                        cell->face(f)->set_all_manifold_ids(1);
                      else
                        cell->face(f)->set_all_manifold_ids(2);
                    }

                  if ((cell->face(f)->center()[1] > 0.6) &&
                      (cell->face(f)->center()[1] < 1.4))
                    {
                      if (cell->face(f)->center()[0] > 0.5)
                        cell->face(f)->set_all_manifold_ids(3);
                      else
                        cell->face(f)->set_all_manifold_ids(4);
                    }
                }
          }
        triangulation.refine_global(6);

        triangulation.save(filename);
        std::cout << "     Saved grid to " << filename << std::endl;

        GridOut       grid_out;
        std::ofstream out("base_grid_Oseen.msh");
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
  OseenProblem<dim>::make_agglo_grid()
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
          std::min(extraction_level - num_domain / max_elem_per_node,
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
  void
  OseenProblem<dim>::set_active_fe_indices()
  {
    for (const auto &polytope : agglo_handler->polytope_iterators())
      {
        if (polytope.master_cell()->material_id() == 0 ||
            polytope.master_cell()->material_id() == 2) // left part
          polytope->set_active_fe_index(0);
        else if (polytope.master_cell()->material_id() == 1 ||
                 polytope.master_cell()->material_id() == 3) // right part
          polytope->set_active_fe_index(1);
        else
          DEAL_II_NOT_IMPLEMENTED();
      }
  }

  template <int dim>
  void
  OseenProblem<dim>::setup_agglomeration()
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
  OseenProblem<dim>::assemble_system()
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
    const FEValuesExtractors::Scalar pressure(dim);

    // Loop over all agglomerated polytopes (cells)
    for (const auto &polytope : agglo_handler->polytope_iterators())
      {
        const unsigned int current_dofs_per_cell =
          polytope->get_fe().dofs_per_cell;
        FullMatrix<double>                   cell_matrix(current_dofs_per_cell,
                                       current_dofs_per_cell);
        Vector<double>                       cell_rhs(current_dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices(
          current_dofs_per_cell);

        cell_matrix              = 0;
        cell_rhs                 = 0;
        const auto &agglo_values = agglo_handler->reinit(polytope);
        polytope->get_dof_indices(local_dof_indices);

        const auto        &q_points  = agglo_values.get_quadrature_points();
        const unsigned int n_qpoints = q_points.size();
        std::vector<Tensor<1, dim>> rhs(n_qpoints);
        std::vector<Tensor<1, dim>> beta(n_qpoints);
        rhs_function->value_list(q_points, rhs);
        beta_function->value_list(q_points, beta);

        std::vector<Tensor<1, dim>> phi_u(current_dofs_per_cell);
        std::vector<Tensor<2, dim>> grad_phi_u(current_dofs_per_cell);
        std::vector<double>         div_phi_u(current_dofs_per_cell);
        std::vector<double>         phi_p(current_dofs_per_cell);

        for (unsigned int q_index : agglo_values.quadrature_point_indices())
          {
            for (unsigned int k = 0; k < current_dofs_per_cell; ++k)
              {
                phi_u[k]      = agglo_values[velocities].value(k, q_index);
                grad_phi_u[k] = agglo_values[velocities].gradient(k, q_index);
                div_phi_u[k]  = agglo_values[velocities].divergence(k, q_index);
                phi_p[k]      = agglo_values[pressure].value(k, q_index);
              }

            for (unsigned int i = 0; i < current_dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < current_dofs_per_cell; ++j)
                  {
                    cell_matrix(i, j) +=
                      (viscosity_nu * scalar_product(grad_phi_u[i],
                                                     grad_phi_u[j]) // + ν ∇v:∇u
                       - div_phi_u[i] * phi_p[j]                    // - ∇·v p
                       + phi_p[i] * div_phi_u[j]                    // + ∇·u q
                       +
                       phi_u[i] * (grad_phi_u[j] * beta[q_index]) // + v· (β·∇)u
                       ) *
                      agglo_values.JxW(q_index); // dx
                    // + ν ∫ ∇v : ∇u dx
                    // -   ∫ (∇·v) p dx
                    // +   ∫ (∇·u) q dx
                    // +   ∫ v · (β·∇u) dx
                  }
                cell_rhs(i) += phi_u[i] * rhs[q_index] *
                               agglo_values.JxW(q_index); // ∫ v·f dx
              }
          }

        // Loop over faces of the current polytope
        const unsigned int n_faces = polytope->n_faces();
        AssertThrow(n_faces > 0,
                    ExcMessage(
                      "Invalid element: at least 4 faces are required."));

        auto polygon_boundary_vertices = polytope->polytope_boundary();
        for (unsigned int f = 0; f < n_faces; ++f)
          {
            if (polytope->at_boundary(f))
              {
                // Handle boundary faces
                const auto &fe_face       = agglo_handler->reinit(polytope, f);
                const auto &face_q_points = fe_face.get_quadrature_points();

                std::vector<Tensor<2, dim>> aver_grad_phi_v(
                  current_dofs_per_cell);
                std::vector<Tensor<1, dim>> jump_phi_v(current_dofs_per_cell);
                std::vector<double>         aver_phi_p(current_dofs_per_cell);
                std::vector<double>         jump_phi_p(current_dofs_per_cell);

                std::vector<Tensor<1, dim>> g(face_q_points.size());
                bcDirichlet->value_list(face_q_points, g);
                std::vector<Tensor<1, dim>> beta(face_q_points.size());
                beta_function->value_list(face_q_points, beta);

                // Get normal vectors seen from each agglomeration.
                const auto &normals = fe_face.get_normal_vectors();

                unsigned int deg_v_current =
                  polytope->get_fe().get_sub_fe(0, 1).degree;
                double tau_cell = (viscosity_nu) * (deg_v_current + 1) *
                                  (deg_v_current + dim) /
                                  std::fabs(polytope->diameter());
                double sigma_v = 40.0 * tau_cell;

                for (unsigned int q_index : fe_face.quadrature_point_indices())
                  {
                    double is_face_inflow_of_cell =
                      (beta[q_index] * normals[q_index]) < 0 ? 1.0 : 0.0;

                    for (unsigned int k = 0; k < current_dofs_per_cell; ++k)
                      {
                        aver_grad_phi_v[k] =
                          fe_face[velocities].gradient(k, q_index);
                        jump_phi_v[k] = fe_face[velocities].value(k, q_index);
                        aver_phi_p[k] = fe_face[pressure].value(k, q_index);
                      }

                    for (unsigned int i = 0; i < current_dofs_per_cell; ++i)
                      {
                        for (unsigned int j = 0; j < current_dofs_per_cell; ++j)
                          {
                            cell_matrix(i, j) +=
                              (-viscosity_nu * jump_phi_v[i] *
                                 (aver_grad_phi_v[j] *
                                  normals[q_index]) // - ν [v]·({∇u}·n)
                               - viscosity_nu * jump_phi_v[j] *
                                   (aver_grad_phi_v[i] *
                                    normals[q_index]) // - ν [u]·({∇v}·n)
                               + sigma_v * jump_phi_v[i] *
                                   jump_phi_v[j] // + σ_v [v]·[u]
                               + aver_phi_p[j] * jump_phi_v[i] *
                                   normals[q_index] // + [v]·n {p}
                               - aver_phi_p[i] * jump_phi_v[j] *
                                   normals[q_index]       // - [u]·n {q}
                               - is_face_inflow_of_cell * // inflow faces only
                                   (beta[q_index] * normals[q_index]) *
                                   jump_phi_v[j] *
                                   jump_phi_v[i] // - (β·n) v_down·[u]
                                                 // v_down = [v] at inflow
                                                 // boundary
                               ) *
                              fe_face.JxW(q_index); // ds
                            // - ν ∫    [v] · ({∇u} · n) ds
                            // - ν ∫    [u] · ({∇v} · n) ds
                            // +   ∫    σ_v [v] · [u] ds
                            // +   ∫    [v] · n · {p} ds
                            // -   ∫    [u] · n · {q} ds
                            // -   ∫_in (β · n) v_down · [u] ds
                            //
                            // where:
                            //   [·]      = jump across face; equals value
                            //              on current cell at boundary
                            //   {·}      = average across face; equals
                            //              value on current cell at boundary
                            //   v_down   = value from downwind side,
                            //              taken as v_current at inflow
                            //              boundary
                            //   ∫_in     = integral over inflow faces
                            //              where (β · n) < 0
                            //
                            // Note:
                            //  Although inflow (upwind) directions usually
                            //  matter more for stability,
                            // here we integrate over faces, not cells.
                            // The downwind-side cell regards the face as an
                            // inflow face. Thus, convection terms on each face
                            // should couple to the downwind-side cell.
                          }
                        cell_rhs(i) +=
                          (-viscosity_nu * g[q_index] *
                             (aver_grad_phi_v[i] *
                              normals[q_index]) // - ν g · ({∇v} · n)
                           +
                           sigma_v * g[q_index] * jump_phi_v[i] // + σ_v g · [v]
                           - aver_phi_p[i] * g[q_index] *
                               normals[q_index] // - {q} (g · n)
                           - is_face_inflow_of_cell *
                               (beta[q_index] * normals[q_index]) * g[q_index] *
                               jump_phi_v[i] // - (β·n) g · [v]
                           ) *
                          fe_face.JxW(q_index); // ds
                        // - ∫     ν g · ({∇v} · n) ds
                        // + ∫     σ_v g · [v] ds
                        // - ∫     {q} (g · n) ds
                        // - ∫_in  (β · n) g · [v] ds
                      }
                  }
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

                    unsigned int nofn =
                      polytope->neighbor_of_agglomerated_neighbor(f);

                    const auto &fe_faces = agglo_handler->reinit_interface(
                      polytope, neigh_polytope, f, nofn);

                    const auto &fe_faces0 = fe_faces.first;
                    const auto &fe_faces1 = fe_faces.second;

                    std::vector<types::global_dof_index>
                      local_dof_indices_neighbor(neigh_dofs_per_cell);

                    // Next, we define the four dofsxdofs matrices needed to
                    // assemble jumps and averages.
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

                    const auto &normals = fe_faces0.get_normal_vectors();

                    std::vector<Tensor<2, dim>> aver_grad_phi_v0(
                      current_dofs_per_cell);
                    std::vector<Tensor<1, dim>> jump_phi_v0(
                      current_dofs_per_cell);
                    std::vector<double> aver_phi_p0(current_dofs_per_cell);
                    std::vector<double> jump_phi_p0(current_dofs_per_cell);
                    std::vector<Tensor<1, dim>> downwind_phi_v0(
                      current_dofs_per_cell);

                    std::vector<Tensor<2, dim>> aver_grad_phi_v1(
                      neigh_dofs_per_cell);
                    std::vector<Tensor<1, dim>> jump_phi_v1(
                      neigh_dofs_per_cell);
                    std::vector<double> aver_phi_p1(neigh_dofs_per_cell);
                    std::vector<double> jump_phi_p1(neigh_dofs_per_cell);
                    std::vector<Tensor<1, dim>> downwind_phi_v1(
                      neigh_dofs_per_cell);

                    std::vector<Tensor<1, dim>> beta(
                      fe_faces0.n_quadrature_points);
                    beta_function->value_list(fe_faces0.get_quadrature_points(),
                                              beta);

                    double beta_max = 0;
                    for (unsigned int q_index = 0;
                         q_index < fe_faces0.n_quadrature_points;
                         ++q_index)
                      {
                        double beta_max0 = beta[q_index].norm();
                        if (beta_max0 > beta_max)
                          beta_max = beta_max0;
                      }

                    unsigned int deg_v_current =
                      polytope->get_fe().get_sub_fe(0, 1).degree;
                    unsigned int deg_v_neigh =
                      neigh_polytope->get_fe().get_sub_fe(0, 1).degree;
                    double tau_current = (viscosity_nu) * (deg_v_current + 1) *
                                         (deg_v_current + dim) /
                                         std::fabs(polytope->diameter());
                    double tau_neigh = (viscosity_nu) * (deg_v_neigh + 1) *
                                       (deg_v_neigh + dim) /
                                       std::fabs(neigh_polytope->diameter());
                    double sigma_v = 40.0 * std::max(tau_current, tau_neigh);

                    double zeta_current =
                      1. / (viscosity_nu / polytope->diameter() + beta_max);
                    double zeta_neigh =
                      1. /
                      (viscosity_nu / neigh_polytope->diameter() + beta_max);
                    double sigma_p = 1.0 * std::max(zeta_current, zeta_neigh);

                    for (unsigned int q_index = 0;
                         q_index < fe_faces0.n_quadrature_points;
                         ++q_index)
                      {
                        bool is_face_inflow_of_cell =
                          ((beta[q_index] * normals[q_index]) < 0);

                        for (unsigned int k = 0; k < current_dofs_per_cell; ++k)
                          {
                            aver_grad_phi_v0[k] =
                              0.5 * fe_faces0[velocities].gradient(k, q_index);
                            jump_phi_v0[k] =
                              fe_faces0[velocities].value(k, q_index);
                            aver_phi_p0[k] =
                              0.5 * fe_faces0[pressure].value(k, q_index);
                            jump_phi_p0[k] =
                              fe_faces0[pressure].value(k, q_index);
                            if (is_face_inflow_of_cell)
                              downwind_phi_v0[k] =
                                fe_faces0[velocities].value(k, q_index);
                            else
                              downwind_phi_v0[k] =
                                -fe_faces0[velocities].value(k, q_index);
                          }

                        for (unsigned int k = 0; k < neigh_dofs_per_cell; ++k)
                          {
                            aver_grad_phi_v1[k] =
                              0.5 * fe_faces1[velocities].gradient(k, q_index);
                            jump_phi_v1[k] =
                              -fe_faces1[velocities].value(k, q_index);
                            aver_phi_p1[k] =
                              0.5 * fe_faces1[pressure].value(k, q_index);
                            jump_phi_p1[k] =
                              -fe_faces1[pressure].value(k, q_index);
                            if (is_face_inflow_of_cell)
                              downwind_phi_v1[k] =
                                -fe_faces1[velocities].value(k, q_index);
                            else
                              downwind_phi_v1[k] =
                                fe_faces1[velocities].value(k, q_index);
                          }

                        for (unsigned int i = 0; i < current_dofs_per_cell; ++i)
                          {
                            for (unsigned int j = 0; j < current_dofs_per_cell;
                                 ++j)
                              {
                                M11(i, j) +=
                                  (-viscosity_nu * jump_phi_v0[i] *
                                     (aver_grad_phi_v0[j] *
                                      normals[q_index]) // - ν [v] · ({∇u} · n)
                                   -
                                   viscosity_nu * jump_phi_v0[j] *
                                     (aver_grad_phi_v0[i] *
                                      normals[q_index]) // - ν [u] · ({∇v} · n)
                                   + sigma_v * jump_phi_v0[i] *
                                       jump_phi_v0[j] // + σ_v [v] · [u]
                                   + aver_phi_p0[j] * jump_phi_v0[i] *
                                       normals[q_index] // + [v] · n · {p}
                                   - aver_phi_p0[i] * jump_phi_v0[j] *
                                       normals[q_index] // - [u] · n · {q}
                                   + sigma_p * jump_phi_p0[i] *
                                       jump_phi_p0[j] // + σ_p [p] · [q]
                                   - (beta[q_index] * normals[q_index]) *
                                       jump_phi_v0[j] *
                                       downwind_phi_v0[i] // - (β·n) v_down·[u]
                                   ) *
                                  fe_faces0.JxW(q_index); // ds
                                // - ν ∫    [v] · ({∇u} · n) ds
                                // - ν ∫    [u] · ({∇v} · n) ds
                                // + ∫     σ_v [v] · [u] ds
                                // + ∫     [v] · n · {p} ds
                                // - ∫     [u] · n · {q} ds
                                // + ∫     σ_p [p] · [q] ds
                                // - ∫_in  (β · n) v_down · [u] ds
                                //
                                // where:
                                //   [·]      = jump across face; equals value
                                //              on current cell at boundary
                                //   {·}      = average across face; equals
                                //              value on current cell at
                                //              boundary
                                //   v_down   = value from downwind side,
                                //              taken as v_current at inflow
                                //              boundary
                                //   ∫_in     = integral over inflow faces
                                //              where (β · n) < 0
                                //   σ_v      = velocity penalty parameter
                                //   σ_p      = pressure penalty parameter
                                //
                                // Note:
                                //   Suffix '0' denotes basis functions of the
                                //   current cell. M11 involves only basis
                                //   functions from the current cell.
                              }
                          }

                        for (unsigned int i = 0; i < current_dofs_per_cell; ++i)
                          {
                            for (unsigned int j = 0; j < neigh_dofs_per_cell;
                                 ++j)
                              {
                                M12(i, j) +=
                                  (-viscosity_nu * jump_phi_v0[i] *
                                     (aver_grad_phi_v1[j] * normals[q_index]) -
                                   viscosity_nu * jump_phi_v1[j] *
                                     (aver_grad_phi_v0[i] * normals[q_index]) +
                                   sigma_v * jump_phi_v0[i] * jump_phi_v1[j] +
                                   aver_phi_p1[j] * jump_phi_v0[i] *
                                     normals[q_index] -
                                   aver_phi_p0[i] * jump_phi_v1[j] *
                                     normals[q_index] +
                                   sigma_p * jump_phi_p0[i] * jump_phi_p1[j] -
                                   (beta[q_index] * normals[q_index]) *
                                     jump_phi_v1[j] * downwind_phi_v0[i]) *
                                  fe_faces0.JxW(q_index);
                                // Same structure as M11; only the basis functions differ.
                                //
                                // Suffix '1' refers to neighbor cell basis
                                // functions, while suffix '0' refers to current
                                // cell basis functions.
                                //
                                // Index [j] corresponds to trial functions,
                                // and [i] to test functions.
                                //
                                // In M21, all [i] indices are associated with
                                // suffix '1', indicating test functions from
                                // the neighbor cell and trial functions from
                                // the current cell.
                              }
                          }

                        for (unsigned int i = 0; i < neigh_dofs_per_cell; ++i)
                          {
                            for (unsigned int j = 0; j < current_dofs_per_cell;
                                 ++j)
                              {
                                M21(i, j) +=
                                  (-viscosity_nu * jump_phi_v1[i] *
                                     (aver_grad_phi_v0[j] * normals[q_index]) -
                                   viscosity_nu * jump_phi_v0[j] *
                                     (aver_grad_phi_v1[i] * normals[q_index]) +
                                   sigma_v * jump_phi_v1[i] * jump_phi_v0[j] +
                                   aver_phi_p0[j] * jump_phi_v1[i] *
                                     normals[q_index] -
                                   aver_phi_p1[i] * jump_phi_v0[j] *
                                     normals[q_index] +
                                   sigma_p * jump_phi_p1[i] * jump_phi_p0[j] -
                                   (beta[q_index] * normals[q_index]) *
                                     jump_phi_v0[j] * downwind_phi_v1[i]) *
                                  fe_faces0.JxW(q_index);
                                // Same structure as M11; only the basis functions differ.
                                //
                                // Suffix '1' refers to neighbor cell basis
                                // functions, while suffix '0' refers to current
                                // cell basis functions.
                                //
                                // Index [j] corresponds to trial functions,
                                // and [i] to test functions.
                                //
                                // In M21, [j] indices use suffix '0',
                                // indicating trial functions from the current
                                // cell, while [i] indices use suffix '1',
                                // indicating test functions from the neighbor
                                // cell.
                              }
                          }

                        for (unsigned int i = 0; i < neigh_dofs_per_cell; ++i)
                          {
                            for (unsigned int j = 0; j < neigh_dofs_per_cell;
                                 ++j)
                              {
                                M22(i, j) +=
                                  (-viscosity_nu * jump_phi_v1[i] *
                                     (aver_grad_phi_v1[j] * normals[q_index]) -
                                   viscosity_nu * jump_phi_v1[j] *
                                     (aver_grad_phi_v1[i] * normals[q_index]) +
                                   sigma_v * jump_phi_v1[i] * jump_phi_v1[j] +
                                   aver_phi_p1[j] * jump_phi_v1[i] *
                                     normals[q_index] -
                                   aver_phi_p1[i] * jump_phi_v1[j] *
                                     normals[q_index] +
                                   sigma_p * jump_phi_p1[i] * jump_phi_p1[j] -
                                   (beta[q_index] * normals[q_index]) *
                                     jump_phi_v1[j] * downwind_phi_v1[i]) *
                                  fe_faces0.JxW(q_index);
                                // Same structure as M11; only the basis functions differ.
                                //
                                // Suffix '1' refers to neighbor cell basis
                                // functions, while suffix '0' refers to current
                                // cell basis functions.
                                //
                                // Index [j] corresponds to trial functions,
                                // and [i] to test functions.
                                //
                                // In M22, both test and trial functions use
                                // suffix '1', meaning they are associated with
                                // the neighbor cell only.
                              }
                          }
                      }

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
        constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
      }
  }

  template <int dim>
  void
  OseenProblem<dim>::solve()
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
  OseenProblem<dim>::zero_pressure_dof_constraint()
  {
    const FEValuesExtractors::Scalar pressure(dim);
    ComponentMask  pressure_mask = fe_collection.component_mask(pressure);
    const IndexSet pressure_dofs =
      DoFTools::extract_dofs(agglo_handler->agglo_dh, pressure_mask);
    const types::global_dof_index first_pressure_dof =
      pressure_dofs.nth_index_in_set(0);
    constraints.constrain_dof_to_zero(first_pressure_dof);
  }

  template <int dim>
  void
  OseenProblem<dim>::mean_pressure_to_zero()
  {
    const ComponentSelectFunction<dim> pressure_mask(dim, dim + 1);
    Vector<double> integral_per_cell(triangulation.n_active_cells());

    VectorTools::integrate_difference(agglo_handler->output_dh,
                                      interpolated_solution,
                                      Functions::ZeroFunction<dim>(dim + 1),
                                      integral_per_cell,
                                      q_collection,
                                      VectorTools::mean,
                                      &pressure_mask);
    const double global_pressure_integral =
      -VectorTools::compute_global_error(triangulation,
                                         integral_per_cell,
                                         VectorTools::mean);
    const double mean_pressure = global_pressure_integral / domain_area;

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
          }
      }
  }

  template <int dim>
  void
  OseenProblem<dim>::compute_errors()
  {
    Vector<float> difference_per_cell(triangulation.n_active_cells());
    const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim),
                                                     dim + 1);
    const ComponentSelectFunction<dim> pressure_mask(dim, dim + 1);

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
                                      &pressure_mask);
    error_pressure = VectorTools::compute_global_error(triangulation,
                                                       difference_per_cell,
                                                       VectorTools::L2_norm);

    std::cout << "     Velocity L2 Error: " << error_velocity_L2 << std::endl
              << "     Velocity H1 Error: " << error_velocity_H1 << std::endl
              << "     pressure L2 Error: " << error_pressure << std::endl;
  }

  template <int dim>
  void
  OseenProblem<dim>::output_results(unsigned int n_subdomains) const
  {
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("pressure");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
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
  OseenProblem<dim>::run()
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
  OseenProblem<dim>::get_error_velocity_L2() const
  {
    return error_velocity_L2;
  }

  template <int dim>
  double
  OseenProblem<dim>::get_error_velocity_H1() const
  {
    return error_velocity_H1;
  }

  template <int dim>
  double
  OseenProblem<dim>::get_error_pressure() const
  {
    return error_pressure;
  }

  template <int dim>
  unsigned int
  OseenProblem<dim>::get_n_dofs() const
  {
    return agglo_handler->n_dofs();
  }
} // namespace OseenNamespace

int
main()
{
  try
    {
      using namespace OseenNamespace;

      ConvergenceTable convergence_table;
      int              deg_v_left  = 3;
      int              deg_p_left  = 2;
      int              deg_v_right = 2;
      int              deg_p_right = 1;
      double           Re          = 1.0;

      for (unsigned int mesh_level = 2; mesh_level < 7; ++mesh_level)
        {
          std::cout << "Mesh level " << mesh_level << std::endl;
          convergence_table.add_value("level", mesh_level);
          convergence_table.add_value("polytopes",
                                      (int)std::pow(4, mesh_level));

          if (mesh_level < 7)
            {
              OseenProblem<2> Oseen_problem(deg_v_left,
                                            deg_p_left,
                                            deg_v_right,
                                            deg_p_right,
                                            mesh_level,
                                            Re);

              Oseen_problem.run();
              convergence_table.add_value("dofs", Oseen_problem.get_n_dofs());
              convergence_table.add_value(
                "velocity_L2", Oseen_problem.get_error_velocity_L2());
              convergence_table.add_value(
                "velocity_H1", Oseen_problem.get_error_velocity_H1());
              convergence_table.add_value("pressure_L2",
                                          Oseen_problem.get_error_pressure());
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
      convergence_table.set_precision("pressure_L2", 3);
      convergence_table.set_scientific("velocity_L2", true);
      convergence_table.set_scientific("velocity_H1", true);
      convergence_table.set_scientific("pressure_L2", true);
      convergence_table.evaluate_convergence_rates(
        "velocity_L2", "polytopes", ConvergenceTable::reduction_rate_log2, 2);
      convergence_table.evaluate_convergence_rates(
        "velocity_H1", "polytopes", ConvergenceTable::reduction_rate_log2, 2);
      convergence_table.evaluate_convergence_rates(
        "pressure_L2", "polytopes", ConvergenceTable::reduction_rate_log2, 2);
      std::cout << "(deg_v_left, deg_p_left) = (" << deg_v_left << ", "
                << deg_p_left << ")," << std::endl
                << "(deg_v_right, deg_p_right) = (" << deg_v_right << ", "
                << deg_p_right << ")," << std::endl
                << std::fixed << std::setprecision(1)
                << "Reynolds number = " << Re << "," << std::endl;
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
