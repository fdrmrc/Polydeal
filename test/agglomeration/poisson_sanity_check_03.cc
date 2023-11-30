#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
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

#include <algorithm>

#include "../tests.h"

#include "../include/agglomeration_handler.h"



enum class GridTypes
{
  grid_generator,
  unstructured
};



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
class Solution : public Function<dim>
{
public:
  Solution()
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
void
Solution<dim>::value_list(const std::vector<Point<dim>> &points,
                          std::vector<double> &          values,
                          const unsigned int /*component*/) const
{
  for (unsigned int i = 0; i < values.size(); ++i)
    values[i] = this->value(points[i]);
}



template <int dim>
class LinearFunction : public Function<dim>
{
public:
  LinearFunction() = default;
  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;
};



template <int dim>
double
LinearFunction<dim>::value(const Point<dim> &p, const unsigned int) const
{
  // x
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
  std::unique_ptr<const Function<dim>>   rhs_function;
  GridTypes                              grid_type;
  unsigned int                           extraction_level;
  unsigned int                           n_subdomains;

public:
  Poisson(const GridTypes &grid_type   = GridTypes::grid_generator,
          const unsigned int           = 0,
          const unsigned int           = 0,
          const unsigned int fe_degree = 1);
  void
  run();

  double penalty_constant = 10.; // =1 => bad; = 10 => 4e-2; // 100 => 4e-1
};



template <int dim>
Poisson<dim>::Poisson(const GridTypes &  grid_type,
                      const unsigned int extraction_level,
                      const unsigned int n_subdomains,
                      const unsigned int fe_degree)
  : mapping(1)
  , dg_fe(fe_degree)
  , grid_type(grid_type)
  , extraction_level(extraction_level)
  , n_subdomains(n_subdomains)
{}

template <int dim>
void
Poisson<dim>::make_grid()
{
  GridIn<dim> grid_in;
  if (grid_type == GridTypes::unstructured)
    {
      grid_in.attach_triangulation(tria);
      std::ifstream gmsh_file(SOURCE_DIR "/../../meshes/t2.msh");
      grid_in.read_msh(gmsh_file);
      tria.refine_global(2);
    }
  else
    {
      GridGenerator::hyper_ball(tria);
      tria.refine_global(6);
    }
  std::cout << "Size of tria: " << tria.n_active_cells() << std::endl;
  cached_tria  = std::make_unique<GridTools::Cache<dim>>(tria, mapping);
  rhs_function = std::make_unique<const RightHandSide<dim>>();

  // Partition with Rtree
  namespace bgi = boost::geometry::index;
  // const unsigned int            extraction_level  = 4; // 3 okay too
  static constexpr unsigned int max_elem_per_node = 4;
  std::vector<std::pair<BoundingBox<dim>,
                        typename Triangulation<dim>::active_cell_iterator>>
               boxes(tria.n_active_cells());
  unsigned int i = 0;
  for (const auto &cell : tria.active_cell_iterators())
    {
      boxes[i++] = std::make_pair(mapping.get_bounding_box(cell), cell);
    }

  // const auto tree = pack_rtree<bgi::rstar<max_elem_per_node>>(boxes);
  const auto tree = pack_rtree<bgi::rstar<max_elem_per_node>>(boxes);
  // const auto tree = pack_rtree<bgi::rstar<max_elem_per_node>>(tria_cells);

  Assert(n_levels(tree) >= 2, ExcMessage("At least two levels are needed."));
  std::cout << "Total number of available levels: " << n_levels(tree)
            << std::endl;
  // Rough description of the tria with a tree of BBoxes
  const auto vec_boxes = extract_rtree_level(tree, extraction_level);
  std::vector<BoundingBox<dim>> bboxes;
  for (const auto &box : vec_boxes)
    bboxes.push_back(box);

  unsigned int k = 0;
  std::vector<std::pair<BoundingBox<dim>,
                        typename Triangulation<dim, dim>::active_cell_iterator>>
    cells;
  std::vector<typename Triangulation<dim, dim>::active_cell_iterator>
                                        cells_to_agglomerate;
  std::vector<types::global_cell_index> idxs_to_agglomerate;
  const auto                            csr_and_agglomerates =
    extract_children_of_level(tree, extraction_level);

  // boost::geometry::index::detail::rtree::utilities::print(std::cout, tree);

  const auto &agglomerates   = csr_and_agglomerates.second; // vec<vec<>>
  [[maybe_unused]] auto csrs = csr_and_agglomerates.first;

  std::size_t agglo_index = 0;
  for (std::size_t i = 0; i < agglomerates.size(); ++i)
    {
      const auto &agglo = agglomerates[i];
      for (const auto &el : agglo)
        {
          el->set_subdomain_id(agglo_index);
          // std::cout << el->active_cell_index() << std::endl;
        }
      ++agglo_index; // one agglomerate has been processed, increment counter
    }
  n_subdomains = agglo_index;
  std::cout << "Number of agglomerates = " << n_subdomains << std::endl;

  constraints.close();


  // Check number of agglomerates
  {
    for (unsigned int j = 0; j < n_subdomains; ++j)
      std::cout << GridTools::count_cells_with_subdomain_association(tria, j)
                << " cells have subdomain " + std::to_string(j) << std::endl;
    GridOut           grid_out_svg;
    GridOutFlags::Svg svg_flags;
    svg_flags.label_subdomain_id = true;
    svg_flags.coloring           = GridOutFlags::Svg::subdomain_id;
    grid_out_svg.set_flags(svg_flags);
    std::string   grid_type = "agglomerated_grid";
    std::ofstream out(grid_type + ".svg");
    grid_out_svg.write_svg(tria, out);
  }
}



template <int dim>
void
Poisson<dim>::setup_agglomeration()

{
  ah = std::make_unique<AgglomerationHandler<dim>>(*cached_tria);

  std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>>
    cells_per_subdomain(n_subdomains);
  for (const auto &cell : tria.active_cell_iterators())
    {
      cells_per_subdomain[cell->subdomain_id()].push_back(cell);
    }

  for (std::size_t i = 0; i < cells_per_subdomain.size(); ++i)
    {
      std::vector<types::global_cell_index> idxs_to_be_agglomerated;
      std::vector<typename Triangulation<dim>::active_cell_iterator>
        cells_to_be_agglomerated;
      // Get all the elements associated with the present subdomain_id
      for (const auto element : cells_per_subdomain[i])
        {
          idxs_to_be_agglomerated.push_back(element->active_cell_index());
        }
      Tests::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated,
                                             cells_to_be_agglomerated);
      // Agglomerate the cells just stored
      ah->agglomerate_cells(cells_to_be_agglomerated);
    }

  ah->distribute_agglomerated_dofs(dg_fe);
  ah->create_agglomeration_sparsity_pattern(sparsity);
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
  const auto &                         bboxes = ah->get_bboxes();

  Solution<dim> analytical_solution;

  for (const auto &cell : ah->agglomeration_cell_iterators())
    {
      cell_matrix              = 0;
      cell_rhs                 = 0;
      const auto &agglo_values = ah->reinit(cell);

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
                  cell_matrix(i, j) += 0. *
                                       agglo_values.shape_grad(i, q_index) *
                                       agglo_values.shape_grad(j, q_index) *
                                       agglo_values.JxW(q_index);
                }
              cell_rhs(i) += agglo_values.shape_value(i, q_index) *
                             rhs[q_index] * agglo_values.JxW(q_index);
            }
        }

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);

      // Face terms
      const unsigned int n_faces = ah->n_faces(cell);
      AssertThrow(n_faces >= 4,
                  ExcMessage(
                    "Invalid element: at least 4 faces are required."));

      // const double agglo_measure =
      // bboxes[cell->active_cell_index()].volume();
      for (unsigned int f = 0; f < n_faces; ++f)
        {
          // double       hf                   = cell->face(0)->measure();
          // const double current_element_size = std::fabs(ah->volume(cell));
          const double current_element_diameter = std::fabs(ah->diameter(cell));
          const double penalty =
            penalty_constant * (1. / current_element_diameter);

          if (ah->at_boundary(cell, f))
            {
              const auto &fe_face = ah->reinit(cell, f);

              const unsigned int dofs_per_cell = fe_face.dofs_per_cell;
              std::vector<types::global_dof_index> local_dof_indices_bdary_cell(
                dofs_per_cell);

              const auto &face_q_points = fe_face.get_quadrature_points();
              std::vector<double> analytical_solution_values(
                face_q_points.size());
              analytical_solution.value_list(face_q_points,
                                             analytical_solution_values,
                                             1);

              // Get normal vectors seen from each agglomeration.
              const auto &normals = fe_face.get_normal_vectors();
              cell_matrix         = 0.;
              cell_rhs            = 0.;
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
                             (penalty)*fe_face.shape_value(i, q_index) *
                               fe_face.shape_value(j, q_index)) *
                            fe_face.JxW(q_index);
                        }
                      cell_rhs(i) +=
                        0. *
                        (penalty * analytical_solution_values[q_index] *
                           fe_face.shape_value(i, q_index) -
                         fe_face.shape_grad(i, q_index) * normals[q_index] *
                           analytical_solution_values[q_index]) *
                        fe_face.JxW(q_index);
                    }
                }

              // distribute DoFs
              cell->get_dof_indices(local_dof_indices_bdary_cell);
              constraints.distribute_local_to_global(cell_matrix,
                                                     cell_rhs,
                                                     local_dof_indices,
                                                     system_matrix,
                                                     system_rhs);
              // constraints.distribute_local_to_global(
              //   cell_matrix, local_dof_indices_bdary_cell, system_matrix);
            }
          else
            {
              const auto &neigh_cell = ah->agglomerated_neighbor(cell, f);

              // const double neigh_element_size =
              //   std::fabs(ah->volume(neigh_cell));
              const double neigh_element_diameter =
                std::fabs(ah->diameter(neigh_cell));

              // const double penalty =
              //   penalty_constant *
              //   std::max(hf / current_element_size,
              //            hf / neigh_element_size); // Cinv still missing
              const double penalty =
                penalty_constant *
                std::max(1. / current_element_diameter,
                         1. / neigh_element_diameter); // Cinv still missing

              // This is necessary to loop over internal faces only once.
              if (cell->active_cell_index() < neigh_cell->active_cell_index())
                {
                  unsigned int nofn =
                    ah->neighbor_of_agglomerated_neighbor(cell, f);

                  const auto &fe_faces =
                    ah->reinit_interface(cell, neigh_cell, f, nofn);

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

                  // distribute DoFs accordingly
                  // Retrieve DoFs info from the cell iterator.
                  typename DoFHandler<dim>::cell_iterator neigh_dh(
                    *neigh_cell, &(ah->agglo_dh));
                  neigh_dh->get_dof_indices(local_dof_indices_neighbor);

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
Poisson<dim>::perform_sanity_check()
{
  LinearFunction<dim> func{}; // v(x,y)= x
  Vector<double>      interp_vector(ah->get_dof_handler().n_dofs());
  VectorTools::interpolate(*(ah->euler_mapping),
                           ah->get_dof_handler(),
                           func,
                           interp_vector);
  const double value =
    system_matrix.matrix_scalar_product(interp_vector, interp_vector);
  Assert(
    std::fabs(value) < 1e-15,
    ExcMessage(
      "Jumps terms for a linear function should be zero (up to machine precision)."));
  std::cout << "Test with linear function: OK" << std::endl;
}

template <int dim>
void
Poisson<dim>::run()
{
  make_grid();
  setup_agglomeration();
  assemble_system();
  perform_sanity_check();
}

int
main()
{
  // Convergence test for Rtree (unstructured gmsh grid)

  Poisson<2> poisson_problem{GridTypes::unstructured, 2 /*extraction_level*/};
  poisson_problem.run();



  return 0;
}
