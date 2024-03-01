// /* ---------------------------------------------------------------------
//  *
//  * Copyright (C) 2022 by the deal.II authors
//  *
//  * This file is part of the deal.II library.
//  *
//  * The deal.II library is free software; you can use it, redistribute
//  * it, and/or modify it under the terms of the GNU Lesser General
//  * Public License as published by the Free Software Foundation; either
//  * version 2.1 of the License, or (at your option) any later version.
//  * The full text of the license can be found in the file LICENSE.md at
//  * the top level directory of deal.II.
//  *
//  * ---------------------------------------------------------------------
//  */


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

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <agglomeration_handler.h>
#include <poly_utils.h>

#include <algorithm>
#include <chrono>


template <typename T>
constexpr T
constexpr_pow(T num, unsigned int pow)
{
  return (pow >= sizeof(unsigned int) * 8) ? 0 :
         pow == 0                          ? 1 :
                                             num * constexpr_pow(num, pow - 1);
}



enum class GridType
{
  grid_generator, // hyper_cube or hyper_ball
  unstructured    // square generated with gmsh, unstructured
};



enum class PartitionerType
{
  metis,
  rtree,
  no_partition
};



enum SolutionType
{
  linear,      // x+y-1
  quadratic,   // x^2+y^2-1
  product,     // xy(x-1)(y-1)
  product_sine // sin(pi*x)*sin(pi*y)
};



template <int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide(const SolutionType &sol_type = SolutionType::linear)
    : Function<dim>()
  {
    solution_type = sol_type;
  }

  virtual void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<double> &          values,
             const unsigned int /*component*/) const override;

private:
  SolutionType solution_type;
};



template <int dim>
void
RightHandSide<dim>::value_list(const std::vector<Point<dim>> &points,
                               std::vector<double> &          values,
                               const unsigned int /*component*/) const
{
  if (solution_type == SolutionType::linear)
    {
      for (unsigned int i = 0; i < values.size(); ++i)
        values[i] = 0.; // Laplacian of linear function
    }
  else if (solution_type == SolutionType::quadratic)
    {
      for (unsigned int i = 0; i < values.size(); ++i)
        values[i] = -4.; // quadratic (radial) solution
    }
  else if (solution_type == SolutionType::product)
    {
      for (unsigned int i = 0; i < values.size(); ++i)
        values[i] = -2. * points[i][0] * (points[i][0] - 1.) -
                    2. * points[i][1] * (points[i][1] - 1.);
    }
  else if (solution_type == SolutionType::product_sine)
    {
      // 2pi^2*sin(pi*x)*sin(pi*y)
      for (unsigned int i = 0; i < values.size(); ++i)
        values[i] = 2. * numbers::PI * numbers::PI *
                    std::sin(numbers::PI * points[i][0]) *
                    std::sin(numbers::PI * points[i][1]);
    }
  else
    {
      Assert(false, ExcNotImplemented());
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
  double sum = 0;
  for (unsigned int d = 0; d < dim; ++d)
    sum += p[d];

  return sum - 1; // p[0]+p[1]+p[2]-1
}

template <int dim>
Tensor<1, dim>
SolutionLinear<dim>::gradient(const Point<dim> &p, const unsigned int) const
{
  (void)p;
  Tensor<1, dim> return_value;
  for (unsigned int d = 0; d < dim; ++d)
    return_value[d] = 0.;
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
  {
    Assert(dim == 2, ExcNotImplemented());
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
SolutionQuadratic<dim>::value(const Point<dim> &p, const unsigned int) const
{
  return p[0] * p[0] + p[1] * p[1] - 1; // ball, radial solution
}

template <int dim>
Tensor<1, dim>
SolutionQuadratic<dim>::gradient(const Point<dim> &p, const unsigned int) const
{
  Tensor<1, dim> return_value;
  return_value[0] = 2. * p[0];
  return_value[1] = 2. * p[1];
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
class SolutionProduct : public Function<dim>
{
public:
  SolutionProduct()
    : Function<dim>()
  {
    Assert(dim == 2, ExcNotImplemented());
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
SolutionProduct<dim>::value(const Point<dim> &p, const unsigned int) const
{
  return p[0] * (p[0] - 1.) * p[1] * (p[1] - 1.); // square
}

template <int dim>
Tensor<1, dim>
SolutionProduct<dim>::gradient(const Point<dim> &p, const unsigned int) const
{
  Tensor<1, dim> return_value;
  return_value[0] = (-1 + 2 * p[0]) * (-1 + p[1]) * p[1];
  return_value[1] = (-1 + 2 * p[1]) * (-1 + p[0]) * p[0];
  return return_value;
}


template <int dim>
void
SolutionProduct<dim>::value_list(const std::vector<Point<dim>> &points,
                                 std::vector<double> &          values,
                                 const unsigned int /*component*/) const
{
  for (unsigned int i = 0; i < values.size(); ++i)
    values[i] = this->value(points[i]);
}



template <int dim>
class SolutionProductSine : public Function<dim>
{
public:
  SolutionProductSine()
    : Function<dim>()
  {
    Assert(dim == 2, ExcNotImplemented());
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
  AffineConstraints<double>                  constraints;
  SparsityPattern                            sparsity;
  SparseMatrix<double>                       system_matrix;
  Vector<double>                             solution;
  Vector<double>                             system_rhs;
  std::unique_ptr<GridTools::Cache<dim>>     cached_tria;
  std::unique_ptr<const Function<dim>>       rhs_function;
  std::unique_ptr<const Function<dim>>       analytical_solution;

public:
  Poisson(const GridType &       grid_type        = GridType::grid_generator,
          const PartitionerType &partitioner_type = PartitionerType::rtree,
          const SolutionType &   solution_type    = SolutionType::linear,
          const unsigned int                      = 0,
          const unsigned int                      = 0,
          const unsigned int fe_degree            = 1);
  void
  run();

  GridType        grid_type;
  PartitionerType partitioner_type;
  SolutionType    solution_type;
  unsigned int    extraction_level;
  unsigned int    n_subdomains;
  double penalty_constant = 60.; // 10*(p+1)(p+d) for p = 1 and d = 2 => 60
};



template <int dim>
Poisson<dim>::Poisson(const GridType &       grid_type,
                      const PartitionerType &partitioner_type,
                      const SolutionType &   solution_type,
                      const unsigned int     extraction_level,
                      const unsigned int     n_subdomains,
                      const unsigned int     fe_degree)
  : mapping(1)
  , dg_fe(fe_degree)
  , grid_type(grid_type)
  , partitioner_type(partitioner_type)
  , solution_type(solution_type)
  , extraction_level(extraction_level)
  , n_subdomains(n_subdomains)
  , penalty_constant(10. * (fe_degree + 1) * (fe_degree + dim))
{
  // Initialize manufactured solution
  if (solution_type == SolutionType::linear)
    analytical_solution = std::make_unique<SolutionLinear<dim>>();
  else if (solution_type == SolutionType::quadratic)
    analytical_solution = std::make_unique<SolutionQuadratic<dim>>();
  else if (solution_type == SolutionType::product)
    analytical_solution = std::make_unique<SolutionProduct<dim>>();
  else if (solution_type == SolutionType::product_sine)
    analytical_solution = std::make_unique<SolutionProductSine<dim>>();

  rhs_function = std::make_unique<const RightHandSide<dim>>(solution_type);
}

template <int dim>
void
Poisson<dim>::make_grid()
{
  GridIn<dim> grid_in;
  if (grid_type == GridType::unstructured)
    {
      if constexpr (dim == 2)
        {
          grid_in.attach_triangulation(tria);
          std::ifstream gmsh_file(
            "../../meshes/t3.msh"); // unstructured square [0,1]^2
          grid_in.read_msh(gmsh_file);
          tria.refine_global(5); // 4
        }
      else if constexpr (dim == 3)
        {
          // {
          //   grid_in.attach_triangulation(tria);
          //   std::ifstream gmsh_file("../../meshes/piston_3.inp"); //
          // piston
          //   mesh grid_in.read_abaqus(gmsh_file); tria.refine_global(1);
          // }
          grid_in.attach_triangulation(tria);
          std::ifstream mesh_file(
            "../../meshes/csf_brain_filled_centered_UCD.inp"); // piston mesh
          grid_in.read_ucd(mesh_file);
          tria.refine_global(1);
        }
    }
  else
    {
      GridGenerator::hyper_cube(tria, 0., 1.);
      tria.refine_global(9);
    }
  std::cout << "Size of tria: " << tria.n_active_cells() << std::endl;
  cached_tria = std::make_unique<GridTools::Cache<dim>>(tria, mapping);

  if (partitioner_type == PartitionerType::metis)
    {
      // Partition the triangulation with graph partitioner.
      auto start = std::chrono::system_clock::now();
      GridTools::partition_triangulation(n_subdomains,
                                         tria,
                                         SparsityTools::Partitioner::metis);
      std::chrono::duration<double> wctduration =
        (std::chrono::system_clock::now() - start);
      std::cout << "METIS built in " << wctduration.count()
                << " seconds [Wall Clock]" << std::endl;
      std::cout << "N subdomains: " << n_subdomains << std::endl;
    }
  else if (partitioner_type == PartitionerType::rtree)
    {
      // Partition with Rtree

      namespace bgi = boost::geometry::index;
      static constexpr unsigned int max_elem_per_node =
        constexpr_pow(2, dim); // 2^dim
      std::vector<std::pair<BoundingBox<dim>,
                            typename Triangulation<dim>::active_cell_iterator>>
                   boxes(tria.n_active_cells());
      unsigned int i = 0;
      for (const auto &cell : tria.active_cell_iterators())
        boxes[i++] = std::make_pair(mapping.get_bounding_box(cell), cell);

      auto       start = std::chrono::system_clock::now();
      const auto tree  = pack_rtree<bgi::rstar<max_elem_per_node>>(boxes);
      std::cout << "Total number of available levels: " << n_levels(tree)
                << std::endl;

#ifdef AGGLO_DEBUG
      // boost::geometry::index::detail::rtree::utilities::print(std::cout,
      // tree);
      Assert(n_levels(tree) >= 2,
             ExcMessage("At least two levels are needed."));
#endif

      const auto &csr_and_agglomerates =
        PolyUtils::extract_children_of_level(tree, extraction_level);
      const auto &agglomerates = csr_and_agglomerates.second; // vec<vec<>>

      std::size_t agglo_index = 0;
      for (std::size_t i = 0; i < agglomerates.size(); ++i)
        {
          // std::cout << "AGGLO " + std::to_string(i) << std::endl;
          const auto &agglo = agglomerates[i];
          for (const auto &el : agglo)
            {
              el->set_subdomain_id(agglo_index);
              // std::cout << el->active_cell_index() << std::endl;
            }
          ++agglo_index; // one agglomerate has been processed, increment
                         // counter
        }
      std::chrono::duration<double> wctduration =
        (std::chrono::system_clock::now() - start);
      std::cout << "R-tree agglomerates built in " << wctduration.count()
                << " seconds [Wall Clock]" << std::endl;
      n_subdomains = agglo_index;

      std::cout << "N subdomains = " << n_subdomains << std::endl;



      // Check number of agglomerates
      if constexpr (dim == 2)
        {
#ifdef AGGLO_DEBUG
          for (unsigned int j = 0; j < n_subdomains; ++j)
            std::cout << GridTools::count_cells_with_subdomain_association(tria,
                                                                           j)
                      << " cells have subdomain " + std::to_string(j)
                      << std::endl;
#endif
          GridOut           grid_out_svg;
          GridOutFlags::Svg svg_flags;
          svg_flags.background     = GridOutFlags::Svg::Background::transparent;
          svg_flags.line_thickness = 1;
          svg_flags.boundary_line_thickness = 1;
          svg_flags.label_subdomain_id      = true;
          svg_flags.coloring =
            GridOutFlags::Svg::subdomain_id; // GridOutFlags::Svg::none
          grid_out_svg.set_flags(svg_flags);
          std::string   grid_type = "agglomerated_grid";
          std::ofstream out(grid_type + ".svg");
          grid_out_svg.write_svg(tria, out);
        }
    }
  else if (partitioner_type == PartitionerType::no_partition)
    {}
  else
    {
      Assert(false, ExcMessage("Wrong partitioning."));
    }


  constraints.close();
}

template <int dim>
void
Poisson<dim>::setup_agglomeration()
{
  ah = std::make_unique<AgglomerationHandler<dim>>(*cached_tria);

  if (partitioner_type != PartitionerType::no_partition)
    {
      std::vector<
        std::vector<typename Triangulation<dim>::active_cell_iterator>>
        cells_per_subdomain(n_subdomains);
      for (const auto &cell : tria.active_cell_iterators())
        cells_per_subdomain[cell->subdomain_id()].push_back(cell);

      // For every subdomain, agglomerate elements together
      for (std::size_t i = 0; i < cells_per_subdomain.size(); ++i)
        ah->define_agglomerate(cells_per_subdomain[i]);
    }
  else
    {
      // No partitioning means that each cell is a master cell
      for (const auto &cell : tria.active_cell_iterators())
        {
          ah->define_agglomerate({cell});
        }
    }

  ah->distribute_agglomerated_dofs(dg_fe);
  ah->create_agglomeration_sparsity_pattern(sparsity);

  {
    std::string partitioner;
    if (partitioner_type == PartitionerType::metis)
      partitioner = "metis";
    else if (partitioner_type == PartitionerType::rtree)
      partitioner = "rtree";
    else
      partitioner = "no_partitioning";

    const std::string filename =
      "grid" + partitioner + "_" + std::to_string(n_subdomains) + ".vtu";
    std::ofstream output(filename);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(ah->agglo_dh);

    Vector<float> agglomerated(tria.n_active_cells());
    Vector<float> agglo_idx(tria.n_active_cells());
    for (const auto &cell : tria.active_cell_iterators())
      {
        agglomerated[cell->active_cell_index()] =
          ah->get_relationships()[cell->active_cell_index()];
        agglo_idx[cell->active_cell_index()] = cell->subdomain_id();
      }
    data_out.add_data_vector(agglomerated,
                             "agglo_relationships",
                             DataOut<dim>::type_cell_data);
    data_out.add_data_vector(agglo_idx,
                             "agglomerated_idx",
                             DataOut<dim>::type_cell_data);
    data_out.build_patches(mapping);
    data_out.write_vtu(output);
  }



  /*
  #ifdef AGGLO_DEBUG
  {
     for (const auto &cell : ah->agglomeration_cell_iterators())
       {
         std::cout << "Cell with idx: " << cell->active_cell_index()
                   << std::endl;
         unsigned int n_agglomerated_faces_per_cell = ah->n_faces(cell);
         std::cout << "Number of faces for the agglomeration: "
                   << n_agglomerated_faces_per_cell << std::endl;
         for (unsigned int f = 0; f < n_agglomerated_faces_per_cell; ++f)
           {
             std::cout << "Agglomerated face with idx: " << f << std::endl;
             const auto &[local_face_idx, neigh, local_face_idx_out, dummy] =
               ah->get_agglomerated_connectivity()[{cell, f}];
             {
               std::cout << "Face idx: " << local_face_idx << std::endl;
               if (neigh.state() == IteratorState::valid)
                 {
                   std::cout << "Neighbor idx: " <<
                   neigh->active_cell_index()
                             << std::endl;
                 }

               std::cout << "Face idx from outside: " << local_face_idx_out
                         << std::endl;
             }
             std::cout << std::endl;
           }
       }
   }
   #endif
   */
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
#ifdef AGGLO_DEBUG
      std::cout << "Polytope with idx: " << polytope->index() << std::endl;
#endif
      cell_matrix              = 0;
      cell_rhs                 = 0;
      const auto &agglo_values = ah->reinit(polytope);
      polytope->get_dof_indices(local_dof_indices);

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


      // Face terms
      const unsigned int n_faces = polytope->n_faces();
      AssertThrow(n_faces > 0,
                  ExcMessage(
                    "Invalid element: at least 4 faces are required."));


#ifdef AGGLO_DEBUG
      std::cout << "Face loop for " << polytope->index() << std::endl;
      std::cout << "n faces = " << n_faces << std::endl;
#endif

      auto polygon_boundary_vertices = polytope->polytope_boundary();
      for (unsigned int f = 0; f < n_faces; ++f)
        {
          if (polytope->at_boundary(f))
            {
              // std::cout << "at boundary!" << std::endl;
              const auto &fe_face = ah->reinit(polytope, f);

              const unsigned int dofs_per_cell = fe_face.dofs_per_cell;
              // std::cout << "With dofs_per_cell =" << fe_face.dofs_per_cell
              //           << std::endl;

              const auto &face_q_points = fe_face.get_quadrature_points();
              std::vector<double> analytical_solution_values(
                face_q_points.size());
              analytical_solution->value_list(face_q_points,
                                              analytical_solution_values,
                                              1);

              // Get normal vectors seen from each agglomeration.
              const auto &normals = fe_face.get_normal_vectors();

              // const double penalty =
              //   penalty_constant / PolyUtils::compute_h_orthogonal(
              //                        f, polygon_boundary_vertices,
              //                        normals[0]);

              const double penalty =
                penalty_constant / std::fabs(polytope->diameter());

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
                }
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
                    std::cout << "Quadrature points from first polytope: "
                              << std::endl;
                    for (const auto &q : fe_faces0.get_quadrature_points())
                      std::cout << q << std::endl;
                    std::cout << "Quadrature points from second polytope: "
                              << std::endl;
                    for (const auto &q : fe_faces1.get_quadrature_points())
                      std::cout << q << std::endl;


                    std::cout << "Check: " << std::endl;
                    const auto &points0 = fe_faces0.get_quadrature_points();
                    const auto &points1 = fe_faces1.get_quadrature_points();
                    for (size_t i = 0;
                         i < fe_faces1.get_quadrature_points().size();
                         ++i)
                      {
                        double d = (points0[i] - points1[i]).norm();
                        Assert(
                          d < 1e-15,
                          ExcMessage(
                            "Face qpoints at the interface do not match!"));
                        std::cout << d << std::endl;
                      }
                  }
#endif

                  std::vector<types::global_dof_index>
                    local_dof_indices_neighbor(dofs_per_cell);

                  M11 = 0.;
                  M12 = 0.;
                  M21 = 0.;
                  M22 = 0.;

                  const auto &normals = fe_faces0.get_normal_vectors();

                  // const double penalty =
                  //   penalty_constant /
                  //   PolyUtils::compute_h_orthogonal(f,
                  //                                   polygon_boundary_vertices,
                  //                                   normals[0]);
                  const double penalty =
                    penalty_constant / std::fabs(polytope->diameter());

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
  {
    const std::string filename = "agglomerated_Poisson.vtu";
    std::ofstream     output(filename);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(ah->agglo_dh);
    data_out.add_data_vector(solution, "u", DataOut<dim>::type_dof_data);
    data_out.build_patches(/**(ah->euler_mapping)*/ mapping);
    data_out.write_vtu(output);
  }
  {
    const std::string filename = "agglomerated_Poisson_test.vtu";
    std::ofstream     output(filename);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(ah->agglo_dh);
    data_out.add_data_vector(solution, "u", DataOut<dim>::type_dof_data);
    data_out.build_patches(*(ah->euler_mapping), 3);
    data_out.write_vtu(output);
  }

  {
    std::string partitioner;
    if (partitioner_type == PartitionerType::metis)
      partitioner = "metis";
    else if (partitioner_type == PartitionerType::rtree)
      partitioner = "rtree";
    else
      partitioner = "no_partitioning";

    const std::string filename = "interpolated_solution_" + partitioner + "_" +
                                 std::to_string(n_subdomains) + ".vtu";
    std::ofstream output(filename);

    DataOut<dim>   data_out;
    Vector<double> interpolated_solution;
    PolyUtils::interpolate_to_fine_grid(*ah, interpolated_solution, solution);
    data_out.attach_dof_handler(ah->output_dh);
    data_out.add_data_vector(interpolated_solution,
                             "u",
                             DataOut<dim>::type_dof_data);

    Vector<float> agglomerated(tria.n_active_cells());
    Vector<float> agglo_idx(tria.n_active_cells());
    for (const auto &cell : tria.active_cell_iterators())
      {
        agglomerated[cell->active_cell_index()] =
          ah->get_relationships()[cell->active_cell_index()];
        agglo_idx[cell->active_cell_index()] = cell->subdomain_id();
      }
    data_out.add_data_vector(agglomerated,
                             "agglo_relationships",
                             DataOut<dim>::type_cell_data);
    data_out.add_data_vector(agglo_idx,
                             "agglo_idx",
                             DataOut<dim>::type_cell_data);


    {
      // L2 error
      Vector<float> difference_per_cell(tria.n_active_cells());
      VectorTools::integrate_difference(mapping,
                                        ah->output_dh,
                                        interpolated_solution,
                                        *analytical_solution,
                                        difference_per_cell,
                                        QGauss<dim>(dg_fe.degree),
                                        VectorTools::L2_norm);

      // Plot the error per cell

      data_out.add_data_vector(difference_per_cell,
                               "error_per_cell",
                               DataOut<dim>::type_cell_data);

      const double L2_error =
        VectorTools::compute_global_error(tria,
                                          difference_per_cell,
                                          VectorTools::L2_norm);

      std::cout << "L2 error:" << L2_error << std::endl;
    }

    {
      // H1 seminorm
      Vector<float> difference_per_cell(tria.n_active_cells());
      VectorTools::integrate_difference(mapping,
                                        ah->output_dh,
                                        interpolated_solution,
                                        *analytical_solution,
                                        difference_per_cell,
                                        QGauss<dim>(dg_fe.degree + 1),
                                        VectorTools::H1_seminorm);

      const double H1_seminorm =
        VectorTools::compute_global_error(tria,
                                          difference_per_cell,
                                          VectorTools::H1_seminorm);

      std::cout << "H1 seminorm:" << H1_seminorm << std::endl;
    }

    data_out.build_patches(mapping);
    data_out.write_vtu(output);
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
  std::cout << "Benchmarking with Rtree:" << std::endl;

  const unsigned int fe_degree = 1;
  // for (const unsigned int extraction_level : {2, 3})
  // for (const unsigned int extraction_level : {2, 3, 4, 5, 6, 7})
  //   {
  //     std::cout << "Level " << extraction_level << std::endl;
  //     Poisson<2> poisson_problem{GridType::unstructured,
  //                                PartitionerType::rtree,
  //                                SolutionType::product,
  //                                extraction_level,
  //                                0,
  //                                fe_degree};
  //     poisson_problem.run();
  //   }

  // Testing p-convergence
  std::cout << "Testing p-convergence" << std::endl;
  {
    // for (unsigned int fe_degree : {1, 2, 3, 4, 5})
    for (unsigned int fe_degree : {1})
      {
        std::cout << "Fe degree: " << fe_degree << std::endl;
        Poisson<2> poisson_problem{GridType::grid_generator,
                                   PartitionerType::rtree,
                                   SolutionType::product_sine,
                                   5 /*extaction_level*/,
                                   0,
                                   fe_degree};
        poisson_problem.run();
      }
  }


  std::cout << std::endl;
  return 0;

  // std::cout << "Benchmarking with METIS:" << std::endl;
  // for (const unsigned int target_partitions :
  //      {16, 64, 256, 1024, 4096}) // ball
  //                                 //  {6, 23, 91, 364, 1456, 5824}) //
  //                                 //  unstructured {16, 64, 256, 1024,
  //                                 //  4096})
  //                                 //  // structured square
  //   {
  //     Poisson<2> poisson_problem{GridType::grid_generator,
  //                                PartitionerType::metis,
  //                                SolutionType::product,
  //                                0,
  //                                target_partitions,
  //                                fe_degree};
  //     poisson_problem.run();
  //   }

  // Testing p-convergence
  // std::cout << "Testing p-convergence" << std::endl;
  // {
  //   for (unsigned int fe_degree : {1, 2, 3, 4, 5})
  //     {
  //       std::cout << "Fe degree: " << fe_degree << std::endl;
  //       Poisson<2> poisson_problem{GridType::grid_generator,
  //                                  PartitionerType::metis,
  //                                  SolutionType::product_sine,
  //                                  0 /*extaction_level*/,
  //                                  1024,
  //                                  fe_degree};
  //       poisson_problem.run();
  //     }
  // }
}
