// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright ( ) XXXX - YYYY by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

using namespace dealii;

#include "continuous_agglo_utils.h"
#include "multigrid_amg.h"
#include "utils.h"

namespace Utils
{

  template <typename MatrixType, typename VectorType, typename DirectSolverType>
  class MGCoarseDirectMUMPS : public MGCoarseGridBase<VectorType>
  {
  public:
    MGCoarseDirectMUMPS()
    {}

    void
    initialize(const MatrixType &matrix)
    {
#ifdef DEAL_II_WITH_MUMPS
      coarse_matrix = &matrix;
      direct_solver = std::make_unique<SparseDirectMUMPS>(
        typename SparseDirectMUMPS::AdditionalData(),
        matrix.get_mpi_communicator());
      direct_solver->initialize(*coarse_matrix);

#else
      DEAL_II_NOT_IMPLEMENTED();
#endif
    }

    virtual void
    operator()([[maybe_unused]] const unsigned int level,
               VectorType                         &dst,
               const VectorType                   &src) const override
    {
#ifdef DEAL_II_WITH_MUMPS
      direct_solver->vmult(dst, src);
#else
      DEAL_II_NOT_IMPLEMENTED();
#endif
    }

    const MatrixType                 *coarse_matrix;
    std::unique_ptr<DirectSolverType> direct_solver;
  };

} // namespace Utils

template <int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide(const std::string &sol_type = "linear")
    : Function<dim>()
  {
    solution_type = sol_type;
  }

  virtual void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<double>           &values,
             const unsigned int /*component*/) const override;

private:
  std::string solution_type;
};



template <int dim>
void
RightHandSide<dim>::value_list(const std::vector<Point<dim>> &points,
                               std::vector<double>           &values,
                               const unsigned int /*component*/) const
{
  if (solution_type == "linear")
    {
      for (unsigned int i = 0; i < values.size(); ++i)
        values[i] = 0.; // Laplacian of linear function
    }
  else if (solution_type == "quadratic")
    {
      for (unsigned int i = 0; i < values.size(); ++i)
        values[i] = -2.0 * dim; // -Δ(Σ x_d^2 - 1) = -2*dim
    }
  else if (solution_type == "product")
    {
      for (unsigned int i = 0; i < values.size(); ++i)
        values[i] = -2. * points[i][0] * (points[i][0] - 1.) -
                    2. * points[i][1] * (points[i][1] - 1.);
    }
  else if (solution_type == "product_sine")
    {
      // 2pi^2*sin(pi*x)*sin(pi*y)
      if constexpr (dim == 2)
        for (unsigned int i = 0; i < values.size(); ++i)
          values[i] = 2. * numbers::PI * numbers::PI *
                      std::sin(numbers::PI * points[i][0]) *
                      std::sin(numbers::PI * points[i][1]);
      else if constexpr (dim == 3)
        for (unsigned int i = 0; i < values.size(); ++i)
          values[i] = 3. * numbers::PI * numbers::PI *
                      std::sin(numbers::PI * points[i][0]) *
                      std::sin(numbers::PI * points[i][1]) *
                      std::sin(numbers::PI * points[i][2]);
      else
        DEAL_II_NOT_IMPLEMENTED();
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
             std::vector<double>           &values,
             const unsigned int /*component*/) const override;

  virtual Tensor<1, dim>
  gradient(const Point<dim>  &p,
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
                                std::vector<double>           &values,
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
             std::vector<double>           &values,
             const unsigned int /*component*/) const override;

  virtual Tensor<1, dim>
  gradient(const Point<dim>  &p,
           const unsigned int component = 0) const override;
};

template <int dim>
double
SolutionQuadratic<dim>::value(const Point<dim> &p, const unsigned int) const
{
  double s = 0.;
  for (unsigned int d = 0; d < dim; ++d)
    s += p[d] * p[d];
  return s - 1.;
}

template <int dim>
Tensor<1, dim>
SolutionQuadratic<dim>::gradient(const Point<dim> &p, const unsigned int) const
{
  Tensor<1, dim> return_value;
  for (unsigned int d = 0; d < dim; ++d)
    return_value[d] = 2. * p[d];
  return return_value;
}


template <int dim>
void
SolutionQuadratic<dim>::value_list(const std::vector<Point<dim>> &points,
                                   std::vector<double>           &values,
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
             std::vector<double>           &values,
             const unsigned int /*component*/) const override;

  virtual Tensor<1, dim>
  gradient(const Point<dim>  &p,
           const unsigned int component = 0) const override;

  virtual void
  gradient_list(const std::vector<Point<dim>> &points,
                std::vector<Tensor<1, dim>>   &gradients,
                const unsigned int /*component*/) const override;
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
                                 std::vector<double>           &values,
                                 const unsigned int /*component*/) const
{
  for (unsigned int i = 0; i < values.size(); ++i)
    values[i] = this->value(points[i]);
}



template <int dim>
void
SolutionProduct<dim>::gradient_list(const std::vector<Point<dim>> &points,
                                    std::vector<Tensor<1, dim>>   &gradients,
                                    const unsigned int /*component*/) const
{
  for (unsigned int i = 0; i < gradients.size(); ++i)
    gradients[i] = this->gradient(points[i]);
}



template <int dim>
class SolutionProductSine : public Function<dim>
{
public:
  SolutionProductSine()
    : Function<dim>()
  {
    static_assert(dim > 1, "Dimension must be greater than 1");
  }

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

  virtual void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<double>           &values,
             const unsigned int /*component*/) const override;

  virtual Tensor<1, dim>
  gradient(const Point<dim>  &p,
           const unsigned int component = 0) const override;
};

template <int dim>
double
SolutionProductSine<dim>::value(const Point<dim> &p, const unsigned int) const
{
  return dim == 2 ?
           std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]) :
           std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]) *
             std::sin(numbers::PI * p[2]);
}

template <int dim>
Tensor<1, dim>
SolutionProductSine<dim>::gradient(const Point<dim> &p,
                                   const unsigned int) const
{
  Tensor<1, dim> return_value;
  if constexpr (dim == 2)
    {
      return_value[0] = numbers::PI * std::cos(numbers::PI * p[0]) *
                        std::sin(numbers::PI * p[1]);
      return_value[1] = numbers::PI * std::cos(numbers::PI * p[1]) *
                        std::sin(numbers::PI * p[0]);
    }
  else if constexpr (dim == 3)
    {
      return_value[0] = numbers::PI * std::cos(numbers::PI * p[0]) *
                        std::sin(numbers::PI * p[1]) *
                        std::sin(numbers::PI * p[2]);
      return_value[1] = numbers::PI * std::cos(numbers::PI * p[1]) *
                        std::sin(numbers::PI * p[0]) *
                        std::sin(numbers::PI * p[2]);
      return_value[2] = numbers::PI * std::cos(numbers::PI * p[2]) *
                        std::sin(numbers::PI * p[0]) *
                        std::sin(numbers::PI * p[1]);
    }
  else
    DEAL_II_NOT_IMPLEMENTED();

  return return_value;
}


template <int dim>
void
SolutionProductSine<dim>::value_list(const std::vector<Point<dim>> &points,
                                     std::vector<double>           &values,
                                     const unsigned int /*component*/) const
{
  for (unsigned int i = 0; i < values.size(); ++i)
    values[i] = this->value(points[i]);
}

template <int dim>
class ProblemParameters : public ParameterAcceptor
{
public:
  ProblemParameters();

  std::string  output_directory    = ".";
  unsigned int mg_levels           = 2;
  unsigned int smoother_steps      = 1;
  bool         use_piston          = false;
  bool         use_real_lv_mesh    = false;
  unsigned int fe_degree           = 1;
  unsigned int coarse_fe_degree    = 1;
  std::string  grid_type           = "unstructured";
  std::string  partitioner_type    = "rtree";
  std::string  solution_type       = "linear";
  unsigned int n_refinements       = 1;
  unsigned int n_ref_cycles        = 1;
  bool         keep_ratio_constant = false; // try to keep H/h fixed
  bool         do_cells_agglo      = true; // test also cell-based agglomeration
  bool         do_points_agglo   = true; // test also point-based agglomeration
  bool         do_trilinos_amg   = true;
  bool         skip_leaves_level = true;

  mutable ParameterAcceptorProxy<ReductionControl> outer_solver_control;
};

template <int dim>
ProblemParameters<dim>::ProblemParameters()
  : ParameterAcceptor("R-tree based MG/")
  , outer_solver_control("Reduction control")

{
  add_parameter("Finite element degree", fe_degree);
  add_parameter("Output directory", output_directory);
  add_parameter("Solution type", solution_type);
  add_parameter("Coarse Finite element degree", coarse_fe_degree);



  enter_subsection("Grid generation");
  {
    add_parameter(
      "Grid type",
      grid_type,
      "Type of grid to use. Options are 'grid_generator' and 'unstructured'.");
    add_parameter(
      "Number of initial refinements",
      n_refinements,
      "Number of global refinements to perform on the initial mesh.");
    add_parameter("Number of refinements cycles",
                  n_ref_cycles,
                  "Number of cycles to perform.");
    add_parameter("Use piston mesh", use_piston);
    add_parameter("Use real lv mesh", use_real_lv_mesh);
  }
  leave_subsection();

  enter_subsection("R3MG");
  {
    add_parameter("Partitioner type", partitioner_type);
    add_parameter("MG Levels", mg_levels);
    add_parameter("Smoother steps", smoother_steps);
    add_parameter("Keep ratio constant", keep_ratio_constant);
    add_parameter("Do point agglomeration", do_points_agglo);
    add_parameter("Do cell agglomeration", do_cells_agglo);
    add_parameter("Do Trilinos AMG", do_trilinos_amg);
    add_parameter("Skip leaves level", skip_leaves_level);
  }
  leave_subsection();

  outer_solver_control.declare_parameters_call_back.connect([]() -> void {
    ParameterAcceptor::prm.set("Max steps", "100");
    ParameterAcceptor::prm.set("Tolerance", "1.e-9");
    ParameterAcceptor::prm.set("Reduction", "1.e-6");
    ParameterAcceptor::prm.set("Log history", "true");
    ParameterAcceptor::prm.set("Log result", "true");
  });
}

template <int dim, unsigned int rtree_m_points, unsigned int rtree_m_cells>
class Poisson
{
private:
  void
  make_grid();
  void
  assemble_system();
  void
  solve_with_point_agglo_amg();
  void
  solve_with_cell_agglo_amg();
  void
  solve_with_trilinos_amg();

  const ProblemParameters<dim>                       &parameters;
  MappingQ1<dim>                                      mapping;
  MPI_Comm                                            comm;
  unsigned int                                        my_rank;
  unsigned int                                        n_mpi_processes;
  parallel::fullydistributed::Triangulation<dim, dim> distributed_tria;
  FE_Q<dim>                                           fe_q;
  AffineConstraints<double>                           constraints;
  TrilinosWrappers::SparsityPattern                   sparsity;
  TrilinosWrappers::SparseMatrix                      system_matrix;
  LinearAlgebra::distributed::Vector<double>          locally_relevant_solution;
  LinearAlgebra::distributed::Vector<double>          system_rhs;
  std::unique_ptr<const Function<dim>>                rhs_function;
  std::unique_ptr<const Function<dim>>                analytical_solution;


  static constexpr unsigned int rtree_M_points = 2 * rtree_m_points;
  static constexpr unsigned int rtree_M_cells  = 2 * rtree_m_cells;

public:
  Poisson(const ProblemParameters<dim> &);
  void
  run();

  std::string grid_type;
  std::string partitioner_type;
  std::string solution_type;
  // std::string  output_info_filename;
  unsigned int mg_levels;

  DoFHandler<dim> original_dof_handler;
  // std::vector<TrilinosWrappers::SparseMatrix>    injection_matrices;
  // std::vector<TrilinosWrappers::SparsityPattern> injection_sparsity_patterns;
  ReductionControl solver_control;
};

template <int dim, unsigned int rtree_m_points, unsigned int rtree_m_cells>
Poisson<dim, rtree_m_points, rtree_m_cells>::Poisson(
  const ProblemParameters<dim> &problem_parameters)
  : parameters(problem_parameters)
  , mapping()
  , comm(MPI_COMM_WORLD)
  , my_rank(Utilities::MPI::this_mpi_process(comm))
  , n_mpi_processes(Utilities::MPI::n_mpi_processes(comm))
  , distributed_tria(comm)
  , fe_q(parameters.fe_degree)
  , grid_type(parameters.grid_type)
  , partitioner_type(parameters.partitioner_type)
  , solution_type(parameters.solution_type)
  , mg_levels(parameters.mg_levels)
  // , output_info_filename(
  //     parameters.output_directory + "/output_info_" +
  //     sanitize_for_filename(parameters.grid_type) + "_p" +
  //     std::to_string(parameters.fe_degree) + "_cp" +
  //     std::to_string(parameters.coarse_fe_degree) + "_ref" +
  //     (parameters.keep_ratio_constant ? "_fixed_ratio" : "_variable_ratio") +
  //     (parameters.do_points_agglo ? "_point_agglo" : "") +
  //     (parameters.do_cells_agglo ? "_cell_agglo" : "") +
  //     std::to_string(parameters.n_refinements) + ".txt")
  , original_dof_handler(distributed_tria)
  , solver_control(parameters.outer_solver_control)
{
  bool is_valid_m = false;
  if constexpr (dim == 3)
    is_valid_m = rtree_m_points >= 4 ? true : false;
  else if constexpr (dim == 2)
    is_valid_m = rtree_m_points >= 2 ? true : false;
  else
    DEAL_II_NOT_IMPLEMENTED();

  AssertThrow(
    is_valid_m,
    ExcMessage(
      "Invalid m for R-tree partitioning. Adjust parameter m accordingly to the dimension."));

  if (solution_type == "linear")
    analytical_solution = std::make_unique<SolutionLinear<dim>>();
  else if (solution_type == "quadratic")
    analytical_solution = std::make_unique<SolutionQuadratic<dim>>();
  else if (solution_type == "product")
    analytical_solution = std::make_unique<SolutionProduct<dim>>();
  else if (solution_type == "product_sine")
    analytical_solution = std::make_unique<SolutionProductSine<dim>>();

  rhs_function = std::make_unique<const RightHandSide<dim>>(solution_type);
  constraints.close();

  solver_control.log_history(parameters.outer_solver_control.log_history() &&
                             (my_rank == 0));
  solver_control.log_result(parameters.outer_solver_control.log_result() &&
                            (my_rank == 0));
}

template <int dim, unsigned int rtree_m_points, unsigned int rtree_m_cells>
void
Poisson<dim, rtree_m_points, rtree_m_cells>::make_grid()
{
  GridIn<dim>        grid_in;
  Triangulation<dim> tria;
  if (grid_type == "unstructured")
    {
      if constexpr (dim == 2)
        {
          grid_in.attach_triangulation(tria);
          std::ifstream gmsh_file("../../meshes/t3.msh"); // unstructured square
          grid_in.read_msh(gmsh_file);
          tria.refine_global(parameters.n_refinements);
          GridTools::partition_triangulation(n_mpi_processes, tria);
          auto description = TriangulationDescription::Utilities::
            create_description_from_triangulation(tria, comm);
          distributed_tria.create_triangulation(description);
        }
      else if constexpr (dim == 3)
        {
          grid_in.attach_triangulation(tria);
          if (parameters.use_piston)
            {
              std::ifstream filename(
                "../../meshes/piston_3.inp"); // piston mesh
              grid_in.read_abaqus(filename);
              tria.refine_global(parameters.n_refinements);
              GridTools::partition_triangulation(n_mpi_processes, tria);
              auto description = TriangulationDescription::Utilities::
                create_description_from_triangulation(tria, comm);
              distributed_tria.create_triangulation(description);
            }
          else if (parameters.use_real_lv_mesh)
            {
              std::ifstream filename("../../meshes/realistic_lv.msh");
              grid_in.read_msh(filename);
              tria.refine_global(parameters.n_refinements);
              GridTools::scale(1e-2, tria);
              GridTools::partition_triangulation(n_mpi_processes, tria);
              auto description = TriangulationDescription::Utilities::
                create_description_from_triangulation(tria, comm);
              distributed_tria.create_triangulation(description);
            }
          else
            {
              std::ifstream filename("../../meshes/idealized_lv.msh");
              grid_in.read_msh(filename);
              tria.refine_global(parameters.n_refinements);
              GridTools::scale(1e-2, tria);
              GridTools::partition_triangulation(n_mpi_processes, tria);
              auto description = TriangulationDescription::Utilities::
                create_description_from_triangulation(tria, comm);
              distributed_tria.create_triangulation(description);
            }

          AssertThrow(tria.all_reference_cells_are_hyper_cube(),
                      ExcMessage("Mixed mesh. Bailing out"));
        }
    }
  else
    {
      // Grids generated through using GridGenerator
      if constexpr (dim == 2)
        GridGenerator::hyper_cube(tria, 0., 1.);
      // GridGenerator::hyper_ball(tria, Point<dim>(), 1.);
      else if constexpr (dim == 3)
        GridGenerator::hyper_cube(tria, 0., 1.);
      // GridGenerator::eccentric_hyper_shell(tria,
      //                                      Point<dim>(1., 1., 1.),
      //                                      Point<dim>(0.7, 0.7, 0.7),
      //                                      0.2,
      //                                      1.,
      //                                      12 /*cells along circumference*/);
      tria.refine_global(parameters.n_refinements);
      GridTools::partition_triangulation(n_mpi_processes, tria);
      auto description = TriangulationDescription::Utilities::
        create_description_from_triangulation(tria, comm);
      distributed_tria.create_triangulation(description);
    }

  if (partitioner_type != "rtree")
    {
      // TODO: maybe add more partitioning?
      Assert(
        false,
        ExcMessage(
          "This test is meant to be run with R-tree partitioning only for now."));
    }
}

template <int dim, unsigned int rtree_m_points, unsigned int rtree_m_cells>
void
Poisson<dim, rtree_m_points, rtree_m_cells>::assemble_system()
{
  original_dof_handler.distribute_dofs(fe_q);

  if (my_rank == 0)
    {
      std::cout << "Assembling system..." << std::endl;
      std::cout << "Number of active cells: "
                << distributed_tria.n_global_active_cells() << std::endl;
      std::cout << "Number of degrees of freedom: "
                << original_dof_handler.n_dofs() << std::endl;
    }

  const IndexSet locally_owned_dofs = original_dof_handler.locally_owned_dofs();
  const IndexSet locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(original_dof_handler);

  constraints.clear();
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

  system_rhs.reinit(locally_owned_dofs, locally_relevant_dofs, comm);
  locally_relevant_solution.reinit(locally_owned_dofs,
                                   locally_relevant_dofs,
                                   comm);

  if (grid_type == "unstructured" && dim == 3)
    {
      if (parameters.use_piston)
        {
          VectorTools::interpolate_boundary_values(original_dof_handler,
                                                   types::boundary_id(0),
                                                   *analytical_solution,
                                                   constraints);
          VectorTools::interpolate_boundary_values(original_dof_handler,
                                                   types::boundary_id(1),
                                                   *analytical_solution,
                                                   constraints);
          VectorTools::interpolate_boundary_values(original_dof_handler,
                                                   types::boundary_id(2),
                                                   *analytical_solution,
                                                   constraints);
        }
      else
        {
          VectorTools::interpolate_boundary_values(original_dof_handler,
                                                   types::boundary_id(10),
                                                   *analytical_solution,
                                                   constraints);
          VectorTools::interpolate_boundary_values(original_dof_handler,
                                                   types::boundary_id(20),
                                                   *analytical_solution,
                                                   constraints);
          VectorTools::interpolate_boundary_values(original_dof_handler,
                                                   types::boundary_id(50),
                                                   *analytical_solution,
                                                   constraints);
        }
    }
  else
    {
      VectorTools::interpolate_boundary_values(original_dof_handler,
                                               types::boundary_id(0),
                                               *analytical_solution,
                                               constraints);
    }

  constraints.close();

  // DynamicSparsityPattern dsp(locally_relevant_dofs);

  // DoFTools::make_sparsity_pattern(original_dof_handler,
  //                                 dsp,
  //                                 constraints,
  //                                 false);
  // SparsityTools::distribute_sparsity_pattern(dsp,
  //                                            locally_owned_dofs,
  //                                            comm,
  //                                            locally_relevant_dofs);

  // system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, comm);

  TrilinosWrappers::SparsityPattern sparsity_pattern(locally_owned_dofs, comm);
  DoFTools::make_sparsity_pattern(original_dof_handler,
                                  sparsity_pattern,
                                  constraints,
                                  false);
  sparsity_pattern.compress();
  system_matrix.reinit(sparsity_pattern);

  const QGauss<dim> quadrature_formula(fe_q.get_degree() + 1);
  FEValues<dim>     fe_values(mapping,
                          fe_q,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe_q.n_dofs_per_cell();
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : original_dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);

        cell_matrix = 0.;
        cell_rhs    = 0.;
        std::vector<double> rhs_values(n_q_points);
        rhs_function->value_list(fe_values.get_quadrature_points(), rhs_values);

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  cell_matrix(i, j) += fe_values.shape_grad(i, q_point) *
                                       fe_values.shape_grad(j, q_point) *
                                       fe_values.JxW(q_point);

                cell_rhs(i) += rhs_values[q_point] *
                               fe_values.shape_value(i, q_point) *
                               fe_values.JxW(q_point);
              }
          }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
      }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}

template <int dim, unsigned int rtree_m_points, unsigned int rtree_m_cells>
void
Poisson<dim, rtree_m_points, rtree_m_cells>::solve_with_point_agglo_amg()
{
  if (my_rank == 0)
    {
      std::cout << "Running R-tree based AMG with point agglomeration"
                << std::endl;
      std::cout << "R-tree m (points) = " << rtree_m_points << std::endl;
      std::cout << "R-tree M (points) = " << rtree_M_points << std::endl;
    }

  const IndexSet locally_owned_dofs = original_dof_handler.locally_owned_dofs();
  const IndexSet locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(original_dof_handler);

  std::vector<TrilinosWrappers::SparseMatrix>    injection_matrices;
  std::vector<TrilinosWrappers::SparsityPattern> injection_sparsity_patterns;
  std::vector<unsigned int> coarse_space_degrees(mg_levels - 1,
                                                 parameters.coarse_fe_degree);

  std::vector<
    std::unique_ptr<parallel::fullydistributed::Triangulation<dim, dim>>>
                                                triangulations(mg_levels - 1);
  std::vector<std::unique_ptr<DoFHandler<dim>>> support_dof_handlers(mg_levels -
                                                                     1);

  ContinuousAggloUtils::PointsAgglo::
    parallel_agglomerate_and_compute_injection_matrices<dim,
                                                        rtree_m_points,
                                                        rtree_M_points>(
      original_dof_handler,
      mapping,
      parameters.skip_leaves_level,
      injection_matrices,
      injection_sparsity_patterns,
      mg_levels,
      coarse_space_degrees,
      triangulations,
      support_dof_handlers);

  AmgProjector<dim, TrilinosWrappers::SparseMatrix, double> amg_projector(
    injection_matrices);

  MGLevelObject<std::unique_ptr<TrilinosWrappers::SparseMatrix>>
    multigrid_matrices(0, mg_levels - 1);

  multigrid_matrices[multigrid_matrices.max_level()] =
    std::make_unique<TrilinosWrappers::SparseMatrix>();
  multigrid_matrices[multigrid_matrices.max_level()]->reinit(system_matrix);
  multigrid_matrices[multigrid_matrices.max_level()]->copy_from(system_matrix);

  amg_projector.compute_level_matrices(multigrid_matrices);

  // Set up multigrid solver
  using LevelMatrixType = TrilinosWrappers::SparseMatrix;
  using VectorType      = LinearAlgebra::distributed::Vector<double>;
  mg::Matrix<VectorType> mg_matrix(multigrid_matrices);

  using SmootherType = PreconditionChebyshev<LevelMatrixType, VectorType>;
  mg::SmootherRelaxation<SmootherType, VectorType>     mg_smoother;
  MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
  smoother_data.resize(0, mg_levels - 1);

  VectorType diag_inverse(system_matrix.locally_owned_range_indices(), comm);
  for (unsigned int row = system_matrix.local_range().first;
       row < system_matrix.local_range().second;
       ++row)
    diag_inverse[row] = 1. / system_matrix.diag_element(row);
  diag_inverse.compress(VectorOperation::insert);

  std::vector<VectorType> diag_inverses(mg_levels);
  diag_inverses[mg_levels - 1] = diag_inverse;

  smoother_data[mg_levels - 1].preconditioner =
    std::make_shared<DiagonalMatrix<VectorType>>(diag_inverses[mg_levels - 1]);

  for (unsigned int level = 0; level < mg_levels - 1; ++level)
    {
      smoother_data[level].smoothing_range = 8;
      diag_inverses[level].reinit(
        multigrid_matrices[level]->locally_owned_range_indices(), comm);
      for (unsigned int row = multigrid_matrices[level]->local_range().first;
           row < multigrid_matrices[level]->local_range().second;
           ++row)
        diag_inverses[level][row] =
          1. / multigrid_matrices[level]->diag_element(row);
      diag_inverses[level].compress(VectorOperation::insert);

      smoother_data[level].preconditioner =
        std::make_shared<DiagonalMatrix<VectorType>>(diag_inverses[level]);
    }

  for (unsigned int level = 0; level < mg_levels; ++level)
    {
      if (level > 0)
        {
          smoother_data[level].smoothing_range     = 20.;
          smoother_data[level].degree              = 3;
          smoother_data[level].eig_cg_n_iterations = 20;
        }
      else
        {
          smoother_data[0].smoothing_range     = 1e-3;
          smoother_data[0].degree              = 3;
          smoother_data[0].eig_cg_n_iterations = 20;
        }
    }

  mg_smoother.set_steps(parameters.smoother_steps);
  mg_smoother.initialize(multigrid_matrices, smoother_data);

  const unsigned int min_level = 0;

  // Utils::MGCoarseDirect<VectorType,
  //                       TrilinosWrappers::SparseMatrix,
  //                       TrilinosWrappers::SolverDirect>
  //   mg_coarse(*multigrid_matrices[0]);

  Utils::MGCoarseDirectMUMPS<TrilinosWrappers::SparseMatrix,
                             VectorType,
                             SparseDirectMUMPS>
    mg_coarse;
  mg_coarse.initialize(*multigrid_matrices[min_level]);

  MGLevelObject<TrilinosWrappers::SparseMatrix *> mg_level_transfers(0,
                                                                     mg_levels -
                                                                       1);
  for (unsigned int l = 0; l < mg_levels - 1; ++l)
    mg_level_transfers[l] = &injection_matrices[l];

  std::vector<DoFHandler<dim> *> dof_handlers(support_dof_handlers.size() + 1);
  for (unsigned int i = 0; i < support_dof_handlers.size(); ++i)
    dof_handlers[i] = support_dof_handlers[i].get();
  dof_handlers[support_dof_handlers.size()] = &original_dof_handler;

  MGTransferAgglomeration<dim, VectorType> mg_transfer(mg_level_transfers,
                                                       dof_handlers);

  Multigrid<VectorType> mg(mg_matrix,
                           mg_coarse,
                           mg_transfer,
                           mg_smoother,
                           mg_smoother,
                           0,
                           numbers::invalid_unsigned_int,
                           Multigrid<VectorType>::v_cycle);

  PreconditionMG<dim, VectorType, MGTransferAgglomeration<dim, VectorType>>
    preconditioner(original_dof_handler, mg, mg_transfer);

  VectorType dist_solution(locally_owned_dofs, comm);
  dist_solution = 0.0;
  SolverCG<VectorType> cg(solver_control);

  cg.connect_condition_number_slot([this](double input) {
    if (my_rank == 0)
      std::cout << "Condition number estimate: " << std::setprecision(6)
                << input << std::endl;
  });

  if (my_rank == 0)
    std::cout
      << "-----------PCG with point agglomeration AMG preconditioner start-----------"
      << std::endl;

  cg.solve(system_matrix, dist_solution, system_rhs, preconditioner);

  if (my_rank == 0)
    std::cout << "Point agglo CG converged in " << solver_control.last_step()
              << " steps, final residual = " << solver_control.last_value()
              << std::endl;

  constraints.distribute(dist_solution);
  locally_relevant_solution = dist_solution;
  locally_relevant_solution.update_ghost_values();

  Vector<double> error_per_cell(distributed_tria.n_active_cells());
  VectorTools::integrate_difference(mapping,
                                    original_dof_handler,
                                    locally_relevant_solution,
                                    *analytical_solution,
                                    error_per_cell,
                                    QGauss<dim>(fe_q.get_degree() + 2),
                                    VectorTools::L2_norm);

  double l2_error =
    std::sqrt(Utilities::MPI::sum(error_per_cell.norm_sqr(), comm));
  if (my_rank == 0)
    std::cout << "Solution L2 error against analytical solution: " << l2_error
              << std::endl;
}

template <int dim, unsigned int rtree_m_points, unsigned int rtree_m_cells>
void
Poisson<dim, rtree_m_points, rtree_m_cells>::solve_with_cell_agglo_amg()
{
  if (my_rank == 0)
    {
      std::cout << "Running R-tree based AMG with cell agglomeration"
                << std::endl;
      std::cout << "R-tree m (cells) = " << rtree_m_cells << std::endl;
      std::cout << "R-tree M (cells) = " << rtree_M_cells << std::endl;
    }

  const IndexSet locally_owned_dofs = original_dof_handler.locally_owned_dofs();
  const IndexSet locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(original_dof_handler);

  std::vector<TrilinosWrappers::SparseMatrix>    injection_matrices;
  std::vector<TrilinosWrappers::SparsityPattern> injection_sparsity_patterns;
  std::vector<unsigned int> coarse_space_degrees(mg_levels - 1,
                                                 parameters.coarse_fe_degree);

  std::vector<
    std::unique_ptr<parallel::fullydistributed::Triangulation<dim, dim>>>
                                                triangulations(mg_levels - 1);
  std::vector<std::unique_ptr<DoFHandler<dim>>> support_dof_handlers(mg_levels -
                                                                     1);

  ContinuousAggloUtils::CellsAgglo::
    parallel_agglomerate_and_compute_injection_matrices<dim,
                                                        rtree_m_cells,
                                                        rtree_M_cells>(
      original_dof_handler,
      mapping,
      parameters.skip_leaves_level,
      injection_matrices,
      injection_sparsity_patterns,
      mg_levels,
      coarse_space_degrees,
      triangulations,
      support_dof_handlers);

  AmgProjector<dim, TrilinosWrappers::SparseMatrix, double> amg_projector(
    injection_matrices);

  MGLevelObject<std::unique_ptr<TrilinosWrappers::SparseMatrix>>
    multigrid_matrices(0, mg_levels - 1);

  multigrid_matrices[multigrid_matrices.max_level()] =
    std::make_unique<TrilinosWrappers::SparseMatrix>();
  multigrid_matrices[multigrid_matrices.max_level()]->reinit(system_matrix);
  multigrid_matrices[multigrid_matrices.max_level()]->copy_from(system_matrix);

  amg_projector.compute_level_matrices(multigrid_matrices);

  // Set up multigrid solver
  using LevelMatrixType = TrilinosWrappers::SparseMatrix;
  using VectorType      = LinearAlgebra::distributed::Vector<double>;
  mg::Matrix<VectorType> mg_matrix(multigrid_matrices);

  using SmootherType = PreconditionChebyshev<LevelMatrixType, VectorType>;
  mg::SmootherRelaxation<SmootherType, VectorType>     mg_smoother;
  MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
  smoother_data.resize(0, mg_levels - 1);

  VectorType diag_inverse(system_matrix.locally_owned_range_indices(), comm);
  for (unsigned int row = system_matrix.local_range().first;
       row < system_matrix.local_range().second;
       ++row)
    diag_inverse[row] = 1. / system_matrix.diag_element(row);
  diag_inverse.compress(VectorOperation::insert);

  std::vector<VectorType> diag_inverses(mg_levels);
  diag_inverses[mg_levels - 1] = diag_inverse;

  smoother_data[mg_levels - 1].preconditioner =
    std::make_shared<DiagonalMatrix<VectorType>>(diag_inverses[mg_levels - 1]);

  for (unsigned int level = 0; level < mg_levels - 1; ++level)
    {
      smoother_data[level].smoothing_range = 8;
      diag_inverses[level].reinit(
        multigrid_matrices[level]->locally_owned_range_indices(), comm);
      for (unsigned int row = multigrid_matrices[level]->local_range().first;
           row < multigrid_matrices[level]->local_range().second;
           ++row)
        diag_inverses[level][row] =
          1. / multigrid_matrices[level]->diag_element(row);
      diag_inverses[level].compress(VectorOperation::insert);

      smoother_data[level].preconditioner =
        std::make_shared<DiagonalMatrix<VectorType>>(diag_inverses[level]);
    }

  for (unsigned int level = 0; level < mg_levels; ++level)
    {
      if (level > 0)
        {
          smoother_data[level].smoothing_range     = 20.;
          smoother_data[level].degree              = 3;
          smoother_data[level].eig_cg_n_iterations = 20;
        }
      else
        {
          smoother_data[0].smoothing_range     = 1e-3;
          smoother_data[0].degree              = 3;
          smoother_data[0].eig_cg_n_iterations = 20;
        }
    }

  mg_smoother.set_steps(parameters.smoother_steps);
  mg_smoother.initialize(multigrid_matrices, smoother_data);

  const unsigned int min_level = 0;

  // Utils::MGCoarseDirect<VectorType,
  //                       TrilinosWrappers::SparseMatrix,
  //                       TrilinosWrappers::SolverDirect>
  //   mg_coarse(*multigrid_matrices[0]);

  Utils::MGCoarseDirectMUMPS<TrilinosWrappers::SparseMatrix,
                             VectorType,
                             SparseDirectMUMPS>
    mg_coarse;
  mg_coarse.initialize(*multigrid_matrices[min_level]);

  MGLevelObject<TrilinosWrappers::SparseMatrix *> mg_level_transfers(0,
                                                                     mg_levels -
                                                                       1);
  for (unsigned int l = 0; l < mg_levels - 1; ++l)
    mg_level_transfers[l] = &injection_matrices[l];

  std::vector<DoFHandler<dim> *> dof_handlers(support_dof_handlers.size() + 1);
  for (unsigned int i = 0; i < support_dof_handlers.size(); ++i)
    dof_handlers[i] = support_dof_handlers[i].get();
  dof_handlers[support_dof_handlers.size()] = &original_dof_handler;

  MGTransferAgglomeration<dim, VectorType> mg_transfer(mg_level_transfers,
                                                       dof_handlers);

  Multigrid<VectorType> mg(mg_matrix,
                           mg_coarse,
                           mg_transfer,
                           mg_smoother,
                           mg_smoother,
                           0,
                           numbers::invalid_unsigned_int,
                           Multigrid<VectorType>::v_cycle);

  PreconditionMG<dim, VectorType, MGTransferAgglomeration<dim, VectorType>>
    preconditioner(original_dof_handler, mg, mg_transfer);

  VectorType dist_solution(locally_owned_dofs, comm);
  dist_solution = 0.0;
  SolverCG<VectorType> cg(solver_control);

  cg.connect_condition_number_slot([this](double input) {
    if (my_rank == 0)
      std::cout << "Condition number estimate: " << std::setprecision(6)
                << input << std::endl;
  });

  if (my_rank == 0)
    std::cout
      << "-----------PCG with cells agglomeration AMG preconditioner start-----------"
      << std::endl;

  cg.solve(system_matrix, dist_solution, system_rhs, preconditioner);

  if (my_rank == 0)
    std::cout << "Point agglo CG converged in " << solver_control.last_step()
              << " steps, final residual = " << solver_control.last_value()
              << std::endl;

  constraints.distribute(dist_solution);
  locally_relevant_solution = dist_solution;
  locally_relevant_solution.update_ghost_values();

  Vector<double> error_per_cell(distributed_tria.n_active_cells());
  VectorTools::integrate_difference(mapping,
                                    original_dof_handler,
                                    locally_relevant_solution,
                                    *analytical_solution,
                                    error_per_cell,
                                    QGauss<dim>(fe_q.get_degree() + 2),
                                    VectorTools::L2_norm);

  double l2_error =
    std::sqrt(Utilities::MPI::sum(error_per_cell.norm_sqr(), comm));
  if (my_rank == 0)
    std::cout << "Solution L2 error against analytical solution: " << l2_error
              << std::endl;
}

template <int dim, unsigned int rtree_m_points, unsigned int rtree_m_cells>
void
Poisson<dim, rtree_m_points, rtree_m_cells>::solve_with_trilinos_amg()
{
  using VectorType = LinearAlgebra::distributed::Vector<double>;

  const IndexSet locally_owned_dofs = original_dof_handler.locally_owned_dofs();
  const IndexSet locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(original_dof_handler);

  TrilinosWrappers::PreconditionAMG                 prec_amg;
  TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;

  amg_data.aggregation_threshold = 1e-2; // AMG aggregation threshold
  amg_data.smoother_type         = "Chebyshev";
  amg_data.smoother_sweeps       = parameters.smoother_steps;
  amg_data.output_details        = false;

  if (fe_q.get_degree() > 1)
    amg_data.higher_order_elements = true;

  prec_amg.initialize(system_matrix, amg_data);

  VectorType dist_solution(locally_owned_dofs, comm);
  dist_solution = 0.0;
  SolverCG<VectorType> cg(solver_control);

  if (my_rank == 0)
    std::cout
      << "-----------PCG with Trilinos AMG preconditioner start-----------"
      << std::endl;

  cg.solve(system_matrix, dist_solution, system_rhs, prec_amg);

  if (my_rank == 0)
    std::cout << "Trilinos AMG CG converged in " << solver_control.last_step()
              << " steps, final residual = " << solver_control.last_value()
              << std::endl;

  constraints.distribute(dist_solution);
  locally_relevant_solution = dist_solution;
  locally_relevant_solution.update_ghost_values();

  Vector<double> error_per_cell(distributed_tria.n_active_cells());
  VectorTools::integrate_difference(mapping,
                                    original_dof_handler,
                                    locally_relevant_solution,
                                    *analytical_solution,
                                    error_per_cell,
                                    QGauss<dim>(fe_q.get_degree() + 2),
                                    VectorTools::L2_norm);

  double l2_error =
    std::sqrt(Utilities::MPI::sum(error_per_cell.norm_sqr(), comm));
  if (my_rank == 0)
    std::cout << "Solution L2 error against analytical solution: " << l2_error
              << std::endl;
}

template <int dim, unsigned int rtree_m_points, unsigned int rtree_m_cells>
void
Poisson<dim, rtree_m_points, rtree_m_cells>::run()
{
  if (my_rank == 0)
    {
      std::cout << "Running with " << n_mpi_processes << " MPI processes"
                << std::endl;
      std::cout << "Dimension: " << dim << std::endl;
      std::cout << "Grid type: " << grid_type << std::endl;
      std::cout << "Partitioner type: " << partitioner_type << std::endl;
      std::cout << "FE degree: " << parameters.fe_degree << std::endl;
      std::cout << "Coarse FE degree: " << parameters.coarse_fe_degree
                << std::endl;
      std::cout << "Number of MG levels: " << mg_levels << std::endl;
      std::cout << "Smoother steps: " << parameters.smoother_steps << std::endl;
      std::cout << "Keep ratio constant: "
                << (parameters.keep_ratio_constant ? "true" : "false")
                << std::endl;
      std::cout << "Number of refinements: " << parameters.n_refinements
                << std::endl;
      std::cout << "Skip leaves level: "
                << (parameters.skip_leaves_level ? "true" : "false")
                << std::endl;
      std::cout << std::string(80, '=') << std::endl;
    }
  make_grid();
  assemble_system();

  if (my_rank == 0)
    std::cout << std::string(80, '=') << std::endl;

  if (parameters.do_points_agglo)
    solve_with_point_agglo_amg();

  if (my_rank == 0)
    std::cout << std::string(80, '=') << std::endl;

  if (parameters.do_cells_agglo)
    solve_with_cell_agglo_amg();

  if (my_rank == 0)
    std::cout << std::string(80, '=') << std::endl;

  if (parameters.do_trilinos_amg)
    solve_with_trilinos_amg();
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  deallog.depth_console(0);

  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    dealii::deallog.depth_console(10);

  static constexpr unsigned int dim = 3;
  ProblemParameters<dim>        parameters;
  std::string                   parameter_file;
  static constexpr unsigned int rtree_m_points = 4;
  static constexpr unsigned int rtree_m_cells  = 4;

  if (argc > 1)
    parameter_file = argv[1];
  else
    parameter_file = "continuous_agglo_amg_parameters.prm";
  ParameterAcceptor::initialize(parameter_file, "used_parameters.prm");

  for (unsigned int cycle = 0; cycle < parameters.n_ref_cycles; ++cycle)
    {
      Poisson<dim, rtree_m_points, rtree_m_cells> poisson_problem{parameters};
      poisson_problem.run();
      parameters.n_refinements++;
      if (!parameters.keep_ratio_constant)
        parameters.mg_levels++;
    }

  std::cout << std::endl;
  return 0;
}