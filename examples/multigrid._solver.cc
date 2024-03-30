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

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <agglomeration_handler.h>
#include <multigrid_utils.h>
#include <poly_utils.h>

#include <algorithm>
#include <chrono>


template <int dim>
class SolutionProductSine : public Function<dim>
{
public:
  SolutionProductSine()
    : Function<dim>()
  {
    static_assert(dim == 2);
  }

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
SolutionProductSine<dim>::value(const Point<dim> &p, const unsigned int) const
{
  return std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]);
}

template <int dim>
Tensor<1, dim>
SolutionProductSine<dim>::gradient(const Point<dim> &p,
                                   const unsigned int) const
{
  Tensor<1, dim> return_value;
  return_value[0] =
    numbers::PI * std::cos(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]);
  return_value[1] =
    numbers::PI * std::cos(numbers::PI * p[1]) * std::sin(numbers::PI * p[0]);
  return return_value;
}


template <int dim>
void
SolutionProductSine<dim>::value_list(const std::vector<Point<dim>> &points,
                                     std::vector<double> &          values,
                                     const unsigned int /*component*/) const
{
  for (unsigned int i = 0; i < values.size(); ++i)
    values[i] = this->value(points[i]);
}



template <int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide()
    : Function<dim>()
  {
    static_assert(dim == 2);
  }

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
  // 2pi^2*sin(pi*x)*sin(pi*y)
  for (unsigned int i = 0; i < values.size(); ++i)
    values[i] = 2. * numbers::PI * numbers::PI *
                std::sin(numbers::PI * points[i][0]) *
                std::sin(numbers::PI * points[i][1]);
}



/**
 * MG parameters
 */
struct GMGParameters
{
  struct CoarseSolverParameters
  {
    std::string  type            = "cg"; // "cg";
    unsigned int maxiter         = 10000;
    double       abstol          = 1e-20;
    double       reltol          = 1e-12;
    unsigned int smoother_sweeps = 1;
    unsigned int n_cycles        = 1;
    std::string  smoother_type   = "ILU";
  };

  struct SmootherParameters
  {
    std::string  type                = "jacobi";
    double       smoothing_range     = 20;
    unsigned int degree              = 5;
    unsigned int eig_cg_n_iterations = 20;
  };

  SmootherParameters     smoother;
  CoarseSolverParameters coarse_solver;

  unsigned int maxiter = 10000;
  double       abstol  = 1e-12;
  double       reltol  = 1e-9;
};

template <typename VectorType,
          int dim,
          typename SystemMatrixType,
          typename LevelMatrixType,
          typename MGTransferType>
static void
mg_solve(SolverControl &                       solver_control,
         VectorType &                          dst,
         const VectorType &                    src,
         const GMGParameters &                 mg_data,
         const DoFHandler<dim> &               dof,
         const SystemMatrixType &              fine_matrix,
         const MGLevelObject<LevelMatrixType> &mg_matrices,
         const MGTransferType &                mg_transfer)
{
  AssertThrow(mg_data.smoother.type == "jacobi", ExcNotImplemented());

  const unsigned int min_level = mg_matrices.min_level();
  const unsigned int max_level = mg_matrices.max_level();

  using Number                     = typename VectorType::value_type;
  using SmootherPreconditionerType = DiagonalMatrix<VectorType>;
  using SmootherType               = PreconditionJacobi<LevelMatrixType>;
  // using SmootherType               = PreconditionChebyshev<LevelMatrixType,
  //                                            VectorType,
  //                                            SmootherPreconditionerType>;
  using PreconditionerType = PreconditionMG<dim, VectorType, MGTransferType>;

  // Initialize level operators.
  mg::Matrix<VectorType> mg_matrix(mg_matrices);

  // // Initialize smoothers.
  // MGLevelObject<typename SmootherType::AdditionalData>
  // smoother_data(min_level,
  //                                                                    max_level);

  // for (unsigned int level = min_level; level <= max_level; ++level)
  //   {
  //     smoother_data[level].preconditioner =
  //       std::make_shared<SmootherPreconditionerType>();
  //     mg_matrices[level].compute_inverse_diagonal(
  //       smoother_data[level].preconditioner->get_vector());
  //     smoother_data[level].smoothing_range =
  //     mg_data.smoother.smoothing_range; smoother_data[level].degree =
  //     mg_data.smoother.degree; smoother_data[level].eig_cg_n_iterations =
  //       mg_data.smoother.eig_cg_n_iterations;
  //   }

  // MGSmootherPrecondition<LevelMatrixType, SmootherType, VectorType>
  // mg_smoother; mg_smoother.initialize(mg_matrices, smoother_data);


  typename SmootherType::AdditionalData data;
  data.relaxation      = .5;
  data.smoothing_range = 20;


  MGSmootherPrecondition<LevelMatrixType, SmootherType, VectorType> mg_smoother;
  mg_smoother.initialize(mg_matrices, data);
  mg_smoother.set_steps(2);

  // Initialize coarse-grid solver.
  ReductionControl     coarse_grid_solver_control(mg_data.coarse_solver.maxiter,
                                              mg_data.coarse_solver.abstol,
                                              mg_data.coarse_solver.reltol,
                                              false,
                                              false);
  SolverCG<VectorType> coarse_grid_solver(coarse_grid_solver_control);

  PreconditionIdentity precondition_identity;
  PreconditionChebyshev<LevelMatrixType, VectorType, DiagonalMatrix<VectorType>>
    precondition_chebyshev;

  std::unique_ptr<MGCoarseGridBase<VectorType>> mg_coarse;

  if (mg_data.coarse_solver.type == "cg")
    {
      // CG with identity matrix as preconditioner

      mg_coarse =
        std::make_unique<MGCoarseGridIterativeSolver<VectorType,
                                                     SolverCG<VectorType>,
                                                     LevelMatrixType,
                                                     PreconditionIdentity>>(
          coarse_grid_solver, mg_matrices[min_level], precondition_identity);
    }
  else
    {
      AssertThrow(false, ExcNotImplemented());
    }

  // Create multigrid object.
  Multigrid<VectorType> mg(
    mg_matrix, *mg_coarse, mg_transfer, mg_smoother, mg_smoother);

  // Convert it to a preconditioner.
  PreconditionerType preconditioner(dof, mg, mg_transfer);

  // Finally, solve.
  SolverCG<VectorType>(solver_control)
    .solve(fine_matrix, dst, src, preconditioner);
}



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
class Test
{
private:
  void
  make_grid();
  void
  check_transfer();

  Triangulation<dim>                         tria;
  MappingQ<dim>                              mapping;
  FE_DGQ<dim>                                dg_fe;
  std::unique_ptr<AgglomerationHandler<dim>> ah_coarse;
  std::unique_ptr<AgglomerationHandler<dim>> ah_fine;
  std::unique_ptr<GridTools::Cache<dim>>     cached_tria;


public:
  Test(const unsigned int = 0, const unsigned int fe_degree = 1);
  void
  run();

  unsigned int extraction_level;
};



template <int dim>
Test<dim>::Test(const unsigned int extraction_level,
                const unsigned int fe_degree)
  : mapping(1)
  , dg_fe(fe_degree)
  , extraction_level(extraction_level)

{}

template <int dim>
void
Test<dim>::make_grid()
{
  // // Cube
  GridGenerator::hyper_cube(tria, 0., 1.);
  tria.refine_global(5);

  // // Ball
  // GridGenerator::hyper_ball(tria);
  // tria.refine_global(4);

  // External mesh
  // GridIn<dim> grid_in;
  // grid_in.attach_triangulation(tria);
  // std::ifstream gmsh_file(
  //   "../../meshes/t3.msh"); // unstructured square [0, 1] ^ 2
  // grid_in.read_msh(gmsh_file);
  // tria.refine_global(3);

  std::cout << "N cells original tria: " << tria.n_active_cells() << std::endl;
  cached_tria = std::make_unique<GridTools::Cache<dim>>(tria, mapping);
}



template <int dim>
void
Test<dim>::check_transfer()
{
  std::unique_ptr<AgglomerationHandler<dim>> ah_test;
  ah_test = std::make_unique<AgglomerationHandler<dim>>(*cached_tria);
  {
    // build the tree
    namespace bgi = boost::geometry::index;
    static constexpr unsigned int max_elem_per_node =
      PolyUtils::constexpr_pow(2, dim); // 2^dim
    std::vector<std::pair<BoundingBox<dim>,
                          typename Triangulation<dim>::active_cell_iterator>>
                 boxes(tria.n_active_cells());
    unsigned int i = 0;
    for (const auto &cell : tria.active_cell_iterators())
      boxes[i++] = std::make_pair(mapping.get_bounding_box(cell), cell);

    const auto tree = pack_rtree<bgi::rstar<max_elem_per_node>>(boxes);

    std::cout << "Number of available levels " << n_levels(tree) << std::endl;

    const auto &csr_and_agglomerates =
      PolyUtils::extract_children_of_level(tree, extraction_level - 1);
    const auto &crss         = csr_and_agglomerates.first;  // vec<vec<>>
    const auto &agglomerates = csr_and_agglomerates.second; // vec<vec<>>
    {
      std::size_t agglo_index = 0;
      for (std::size_t i = 0; i < agglomerates.size(); ++i)
        {
#ifdef AGGLO_DEBUG
          std::cout << "AGGLO " + std::to_string(i) << std::endl;
#endif
          const auto &agglo = agglomerates[i];
          for (const auto &el : agglo)
            {
              el->set_subdomain_id(agglo_index);
#ifdef AGGLO_DEBUG
              std::cout << el->active_cell_index() << std::endl;
#endif
            }
          ++agglo_index; // one agglomerate has been processed, increment
                         // counter
        }

      const unsigned int n_subdomains = agglo_index;

      std::cout << "N elements (test level) = " << n_subdomains << std::endl;

      std::vector<
        std::vector<typename Triangulation<dim>::active_cell_iterator>>
        cells_per_subdomain(n_subdomains);
      for (const auto &cell : tria.active_cell_iterators())
        cells_per_subdomain[cell->subdomain_id()].push_back(cell);

      // For every subdomain, agglomerate elements together
      for (std::size_t i = 0; i < cells_per_subdomain.size(); ++i)
        ah_test->define_agglomerate(cells_per_subdomain[i]);
    }



    // begin debug
    {
      for (std::size_t i = 0; i < agglomerates.size(); ++i)
        {
          std::cout << "AGGLO " + std::to_string(i) << std::endl;
          const auto &agglo = agglomerates[i];
          // std::vector<types::global_cell_index> cell_indices;
          // for (const auto &cell : agglo)
          //   cell_indices.push_back(cell->active_cell_index());

          // const auto dof_cell =
          //   agglo[std::distance(std::begin(cell_indices),
          //                       std::min_element(std::begin(cell_indices),
          //                                        std::end(cell_indices)))]
          //     ->as_dof_handler_iterator(agglo_dh); // min index is master
          //     cell

          // std::vector<types::global_dof_index>
          // dof_indices(dg_fe.dofs_per_cell);
          // dof_cell->get_dof_indices(dof_indices);
          // std::cout << "DoF indices:" << std::endl;
          // for (const unsigned int idx : dof_indices)
          //   std::cout << idx << std::endl;
          // std::cout << std::endl;
          const auto &crs = crss[i];
          std::cout << "crss[ " << i << "] has size " << crs.size()
                    << std::endl;
          std::cout << "i = " << i << std::endl;
          for (unsigned int k = 0; k < crs.size() - 1; ++k)
            {
              std::cout << "crs[" << k << "]" << std::endl;
              for (unsigned int j = crs[k]; j < crs[k + 1]; ++j)
                {
                  std::cout << j << " and " << agglo[j] << std::endl;
                }
              std::cout << std::endl;
            }
        }
    }

    // end debug
  }
  ah_test->distribute_agglomerated_dofs(dg_fe);
  DynamicSparsityPattern dsp_test;
  SparsityPattern        sparsity_test;
  ah_test->create_agglomeration_sparsity_pattern(dsp_test);
  sparsity_test.copy_from(dsp_test);


  // Check construction of transfer operator



  // ****************do the same ****************
  ah_coarse = std::make_unique<AgglomerationHandler<dim>>(*cached_tria);

  // build the tree
  namespace bgi = boost::geometry::index;
  static constexpr unsigned int max_elem_per_node =
    PolyUtils::constexpr_pow(2, dim); // 2^dim
  std::vector<std::pair<BoundingBox<dim>,
                        typename Triangulation<dim>::active_cell_iterator>>
               boxes(tria.n_active_cells());
  unsigned int i = 0;
  for (const auto &cell : tria.active_cell_iterators())
    boxes[i++] = std::make_pair(mapping.get_bounding_box(cell), cell);

  const auto tree = pack_rtree<bgi::rstar<max_elem_per_node>>(boxes);

  std::cout << "Number of available levels " << n_levels(tree) << std::endl;

  const auto &csr_and_agglomerates =
    PolyUtils::extract_children_of_level(tree, extraction_level);
  const auto &agglomerates = csr_and_agglomerates.second; // vec<vec<>>
  {
    std::size_t agglo_index = 0;
    for (std::size_t i = 0; i < agglomerates.size(); ++i)
      {
#ifdef FALSE
        std::cout << "AGGLO " + std::to_string(i) << std::endl;
#endif
        const auto &agglo = agglomerates[i];
        for (const auto &el : agglo)
          {
            el->set_subdomain_id(agglo_index);
#ifdef FALSE
            std::cout << el->active_cell_index() << std::endl;
#endif
          }
        ++agglo_index; // one agglomerate has been processed, increment
                       // counter
      }

    const unsigned int n_subdomains = agglo_index;

    std::cout << "N elements (coarse) = " << n_subdomains << std::endl;

    std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>>
      cells_per_subdomain(n_subdomains);
    for (const auto &cell : tria.active_cell_iterators())
      cells_per_subdomain[cell->subdomain_id()].push_back(cell);

    // For every subdomain, agglomerate elements together
    for (std::size_t i = 0; i < cells_per_subdomain.size(); ++i)
      ah_coarse->define_agglomerate(cells_per_subdomain[i]);
  }


  ah_coarse->distribute_agglomerated_dofs(dg_fe);
  DynamicSparsityPattern dsp_coarse;
  SparsityPattern        sparsity_coarse;
  ah_coarse->create_agglomeration_sparsity_pattern(dsp_coarse);
  sparsity_coarse.copy_from(dsp_coarse);

  std::cout << "Master cells coarse level: " << std::endl;
  for (const auto &cell : ah_coarse->master_cells_container)
    std::cout << cell->active_cell_index() << std::endl;


  std::cout << "Construct transfer operator test->coarse" << std::endl;
  const auto &csr_and_agglomerates_test =
    PolyUtils::extract_children_of_level(tree, extraction_level - 1);
  RtreeInfo<2> rtree_info_test{csr_and_agglomerates_test.first,
                               csr_and_agglomerates_test.second};

  const auto &agglomerates_test = csr_and_agglomerates_test.second;

  std::vector<std::vector<unsigned int>> crs_new(agglomerates_test.size());

  for (std::size_t i = 0; i < agglomerates_test.size(); ++i)
    {
      std::cout << "Agglo " << i << std::endl;
      const auto &                     agglo = agglomerates_test[i];
      const std::vector<unsigned int> &crs = csr_and_agglomerates_test.first[i];
      unsigned int                     kk  = 0;
      crs_new[i].push_back(kk);
      for (unsigned int k = 0; k < crs.size() - 1; k += 5)
        {
          // crs[k]
          for (unsigned int j = crs[k]; j < crs[k + 4]; ++j)
            {
              if (k != 4 || k != 9 || k != 14 || k != 19)
                {
                  std::cout << j << " AND " << agglo[j] << std::endl;
                  ++kk;
                }
            }
          crs_new[i].push_back(kk);
        }
      std::cout << std::endl;
    }

  // std::cout << "DEBUG crs:" << std::endl;
  // unsigned int agglo_ctr = 0;
  // for (const std::vector<unsigned int> &v : crs_new)
  //   {
  //     std::cout << "Agglo " << agglo_ctr++ << std::endl;
  //     for (const auto i : v)
  //       std::cout << i << std::endl;
  //     std::cout << std::endl;
  //   }


  std::vector<std::vector<unsigned int>> crs_test(agglomerates_test.size());
  for (unsigned int i = 0; i < crs_test.size() - 1; ++i)
    crs_test[i] = {0, 64, 128, 192, 256};
  crs_test[crs_test.size() - 1] = {0, 64, 128, 192};

  // crs_test[0] = {0, 64, 128, 192, 255};
  // crs_test[1] = {0, 64, 128, 192, 255};
  // crs_test[2] = {0, 64, 128, 192, 255};
  // crs_test[3] = {0, 64, 128, 192, 255};

  RtreeInfo<2> rtree_info_test_level{crs_test, agglomerates_test};
  MGTwoLevelTransferAgglomeration<dim, Vector<double>>
    agglomeration_transfer_coarse_test(rtree_info_test_level);
  agglomeration_transfer_coarse_test.reinit(
    *ah_coarse, *ah_test); // ah_test is coarser handler



  // ****************do the same for the fine grid one****************
  ah_fine = std::make_unique<AgglomerationHandler<dim>>(*cached_tria);


  {
    const auto &csr_and_agglomerates_fine =
      PolyUtils::extract_children_of_level(tree,
                                           extraction_level + 1); //! level+1
    const auto &agglomerates = csr_and_agglomerates_fine.second;  // vec<vec<>>

    std::size_t agglo_index_fine  = 0;
    std::size_t n_subdomains_fine = 0;
    for (std::size_t i = 0; i < agglomerates.size(); ++i)
      {
#ifdef FALSE
        std::cout << "AGGLO FINE" + std::to_string(i) << std::endl;
#endif
        const auto &agglo = agglomerates[i];
        for (const auto &el : agglo)
          {
            el->set_material_id(agglo_index_fine);
#ifdef FALSE
            std::cout << el->active_cell_index() << std::endl;
#endif
          }
        ++agglo_index_fine; // one agglomerate has been processed, increment
                            // counter
      }

    n_subdomains_fine = agglo_index_fine;

    std::cout << "N elements (fine) = " << n_subdomains_fine << std::endl;


    std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>>
      cells_per_subdomain(n_subdomains_fine);
    for (const auto &cell : tria.active_cell_iterators())
      cells_per_subdomain[cell->material_id()].push_back(cell);

    // For every subdomain, agglomerate elements together
    for (std::size_t i = 0; i < cells_per_subdomain.size(); ++i)
      ah_fine->define_agglomerate(cells_per_subdomain[i]);
  }
  ah_fine->distribute_agglomerated_dofs(dg_fe);
  DynamicSparsityPattern dsp_fine;
  SparsityPattern        sparsity_fine;
  ah_fine->create_agglomeration_sparsity_pattern(dsp_fine);
  sparsity_fine.copy_from(dsp_fine);

#ifdef FALSE
  std::cout << "Master cells fine: " << std::endl;
  for (const auto &cell : ah_fine->master_cells_container)
    std::cout << cell->active_cell_index() << std::endl;
#endif


  // Check construction of transfer operator

  std::cout << "Construct transfer operator" << std::endl;
  RtreeInfo<2> rtree_info{csr_and_agglomerates.first,
                          csr_and_agglomerates.second};
  MGTwoLevelTransferAgglomeration<dim, Vector<double>> agglomeration_transfer(
    rtree_info);
  agglomeration_transfer.reinit(*ah_fine, *ah_coarse);


#ifdef AGGLO_DEBUG
  const auto &do_test = [&](const Function<dim> &func) {
    // Test with linear function
    Vector<double>   interp_coarse(ah_coarse->agglo_dh.n_dofs());
    std::vector<int> coeffs{1, 1};
    VectorTools::interpolate(*(ah_coarse->euler_mapping),
                             ah_coarse->agglo_dh,
                             func,
                             interp_coarse);

    Vector<double> dst(ah_fine->agglo_dh.n_dofs());
    agglomeration_transfer.prolongate(dst, interp_coarse);


#  ifdef FALSE
    DataOut<2> data_out;
    data_out.attach_dof_handler(ah_coarse->agglo_dh);
    data_out.add_data_vector(interp_coarse, "solution");
    data_out.build_patches(*(ah_coarse->euler_mapping));
    std::ofstream output_coarse("coarse_sol_linear.vtk");
    data_out.write_vtk(output_coarse);
    data_out.clear();
    std::ofstream output_fine("prolonged_solution_linear.vtk");
    data_out.attach_dof_handler(ah_fine->agglo_dh);
    data_out.add_data_vector(dst, "prolonged_solution_linear");
    data_out.build_patches(*(ah_fine->euler_mapping));
    data_out.write_vtk(output_fine);
#  endif

    // Compute error:
    Vector<double> interp_fine(ah_fine->agglo_dh.n_dofs());
    VectorTools::interpolate(*(ah_fine->euler_mapping),
                             ah_fine->agglo_dh,
                             func,
                             interp_fine);

    Vector<double> err;
    dst -= interp_fine;
    std::cout << "Norm of error(L2): " << dst.l2_norm() << std::endl;
  };

  std::cout << "f= 1" << std::endl;
  do_test(Functions::ConstantFunction<dim>(1.));
  std::cout << "f= x+y" << std::endl;
  std::vector<int> coeffs_linear{1, 1};
  do_test(LinearFunction<dim>{coeffs_linear});
  std::cout << "f= x" << std::endl;
  std::vector<int> coeffs_x{1, 0};
  do_test(LinearFunction<dim>{coeffs_x});
  std::cout << "f= y" << std::endl;
  std::vector<int> coeffs_y{0, 1};
  do_test(LinearFunction<dim>{coeffs_y});
#endif


  /**************** SOLVE WITH MG ****************/


  SolutionProductSine<dim> analytical_solution;
  RightHandSide<dim>       rhs_function;


  // Assemble matrices and rhs

  const auto &assemble_matrix = [&](AgglomerationHandler<dim> &ah,
                                    const SparsityPattern &    sparsity,
                                    SparseMatrix<double> &     system_matrix,
                                    Vector<double> &           system_rhs) {
    AffineConstraints<double> constraints;
    constraints.close();

    system_matrix.reinit(sparsity);
    system_rhs.reinit(ah.n_dofs());

    const unsigned int quadrature_degree      = 2 * dg_fe.get_degree() + 1;
    const unsigned int face_quadrature_degree = 2 * dg_fe.get_degree() + 1;
    ah.initialize_fe_values(QGauss<dim>(quadrature_degree),
                            update_gradients | update_JxW_values |
                              update_quadrature_points | update_JxW_values |
                              update_values,
                            QGauss<dim - 1>(face_quadrature_degree));

    const unsigned int dofs_per_cell = ah.n_dofs_per_cell();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    // Next, we define the four dofsxdofs matrices needed to assemble jumps and
    // averages.
    FullMatrix<double> M11(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> M12(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> M21(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> M22(dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &polytope : ah.polytope_iterators())
      {
        cell_matrix              = 0;
        cell_rhs                 = 0;
        const auto &agglo_values = ah.reinit(polytope);
        polytope->get_dof_indices(local_dof_indices);

        const auto &        q_points  = agglo_values.get_quadrature_points();
        const unsigned int  n_qpoints = q_points.size();
        std::vector<double> rhs(n_qpoints);
        rhs_function.value_list(q_points, rhs, 1);

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


        // Face terms
        const unsigned int n_faces = polytope->n_faces();
        AssertThrow(n_faces > 0,
                    ExcMessage(
                      "Invalid element: at least 4 faces are required."));


        auto polygon_boundary_vertices = polytope->polytope_boundary();
        for (unsigned int f = 0; f < n_faces; ++f)
          {
            if (polytope->at_boundary(f))
              {
                // std::cout << "at boundary!" << std::endl;
                const auto &fe_face = ah.reinit(polytope, f);

                const unsigned int dofs_per_cell = fe_face.dofs_per_cell;
                // std::cout << "With dofs_per_cell =" << fe_face.dofs_per_cell
                //           << std::endl;

                const auto &face_q_points = fe_face.get_quadrature_points();
                std::vector<double> analytical_solution_values(
                  face_q_points.size());
                analytical_solution.value_list(face_q_points,
                                               analytical_solution_values,
                                               1);

                // Get normal vectors seen from each agglomeration.
                const auto &normals = fe_face.get_normal_vectors();

                // const double penalty =
                //   20. / PolyUtils::compute_h_orthogonal(
                //                        f, polygon_boundary_vertices,
                //                        normals[0]);

                const double penalty = 20. / std::fabs(polytope->diameter());

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
                               fe_face.shape_grad(i, q_index) *
                                 normals[q_index] *
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
                  }
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
                      ah.reinit_interface(polytope, neigh_polytope, f, nofn);

                    const auto &fe_faces0 = fe_faces.first;
                    const auto &fe_faces1 = fe_faces.second;


                    std::vector<types::global_dof_index>
                      local_dof_indices_neighbor(dofs_per_cell);

                    M11 = 0.;
                    M12 = 0.;
                    M21 = 0.;
                    M22 = 0.;

                    const auto &normals = fe_faces0.get_normal_vectors();

                    // const double penalty =
                    //   20. /
                    //   PolyUtils::compute_h_orthogonal(f,
                    //                                   polygon_boundary_vertices,
                    //                                   normals[0]);
                    const double penalty =
                      20. / std::fabs(polytope->diameter());

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

        // distribute DoFs
        constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
      } // Loop over cells
  };



  Vector<double> system_rhs;


  const unsigned int min_level = 0;
  const unsigned int max_level = 2;
  GMGParameters      mg_data; // TODO

  MGLevelObject<SparseMatrix<double>> mg_matrices(min_level, max_level);
  MGLevelObject<SparsityPattern *>    sparsities(min_level, max_level);
  sparsities[min_level] = &sparsity_test;
  sparsities[1]         = &sparsity_coarse;
  sparsities[max_level] = &sparsity_fine;
  MGLevelObject<AgglomerationHandler<dim> *> handlers(min_level, max_level);
  handlers[min_level] = ah_test.get();
  handlers[1]         = ah_coarse.get();
  handlers[max_level] = ah_fine.get();


  Vector<double> dummy_vec; // not needed on coarser levels


  for (unsigned int l = min_level; l < max_level; ++l)
    {
      assemble_matrix(*handlers[l], *sparsities[l], mg_matrices[l], dummy_vec);
    }

  // Create finer level
  assemble_matrix(*ah_fine,
                  sparsity_fine,
                  mg_matrices[max_level],
                  system_rhs); // fine level

  std::cout << "Assembled level matrices" << std::endl;


  // assemble fine and coarse matrices
  MGLevelObject<
    std::shared_ptr<MGTwoLevelTransferAgglomeration<dim, Vector<double>>>>
    transfers(min_level, max_level);

  // for (auto l = min_level; l < max_level; ++l)
  //   {
  //     transfers[l + 1] =
  //       std::make_shared<MGTwoLevelTransferAgglomeration<dim,
  //       Vector<double>>>(
  //         rtree_info);
  //     transfers[l + 1]->reinit(*ah_fine, *ah_coarse);
  //   }

  transfers[1] =
    std::make_shared<MGTwoLevelTransferAgglomeration<dim, Vector<double>>>(
      rtree_info_test_level);
  transfers[1]->reinit(*ah_coarse, *ah_test);

  transfers[2] =
    std::make_shared<MGTwoLevelTransferAgglomeration<dim, Vector<double>>>(
      rtree_info);
  transfers[2]->reinit(*ah_fine, *ah_coarse);

  // Construct the actual transfer object to be used inside MG
  std::vector<const DoFHandler<dim> *> dof_handlers_vector;
  dof_handlers_vector.push_back(&ah_test->agglo_dh);
  dof_handlers_vector.push_back(&ah_coarse->agglo_dh);
  dof_handlers_vector.push_back(&ah_fine->agglo_dh);

  MGTransferAgglomeration<dim, Vector<double>> transfer(transfers,
                                                        dof_handlers_vector);

  Vector<double> dst;
  dst.reinit(ah_fine->agglo_dh.n_dofs());

  ReductionControl solver_control(
    mg_data.maxiter, mg_data.abstol, mg_data.reltol, false, false);

  std::cout << "Start solver" << std::endl;
  mg_solve(solver_control,
           dst,
           system_rhs,
           mg_data,
           ah_fine->agglo_dh,
           mg_matrices[max_level],
           mg_matrices,
           transfer);

  std::cout << "CG converged in " << solver_control.last_step()
            << " iterations." << std::endl;

  // Post processing: interpolante onto finer grid and compute L2 error.
  {
    DataOut<dim>   data_out;
    Vector<double> interpolated_solution;
    PolyUtils::interpolate_to_fine_grid(*ah_fine, interpolated_solution, dst);
    data_out.attach_dof_handler(ah_fine->output_dh);
    data_out.add_data_vector(interpolated_solution,
                             "interpolated_solution",
                             DataOut<dim>::type_dof_data);

    Vector<double> check(ah_fine->output_dh.n_dofs());
    VectorTools::interpolate(mapping,
                             ah_fine->output_dh,
                             analytical_solution,
                             check);

    data_out.add_data_vector(check, "check", DataOut<dim>::type_dof_data);
    const std::string filename = "check_multigrid.vtu";
    std::ofstream     output(filename);
    data_out.build_patches(mapping);
    data_out.write_vtu(output);


    // L2 error
    Vector<float> difference_per_cell(tria.n_active_cells());
    VectorTools::integrate_difference(mapping,
                                      ah_fine->output_dh,
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
  }
}



template <int dim>
void
Test<dim>::run()
{
  make_grid();
  check_transfer();
}


int
main()
{
  // {
  //   // Square
  //   const unsigned int fe_degree = 1;
  //   Test<2>            prolongation_test{2 /*extaction_level*/, fe_degree};
  //   prolongation_test.run();
  // }

  {
    // Square unstructured
    const unsigned int fe_degree = 1;
    Test<2>            prolongation_test{4 /*extaction_level*/, fe_degree};
    prolongation_test.run();
  }

  // {
  //   // Ball
  //   const unsigned int fe_degree = 2;
  //   Test<2>            prolongation_test{3 /*extaction_level*/, fe_degree};
  //   prolongation_test.run();
  // }


  return 0;
}
