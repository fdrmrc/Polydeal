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

#include <algorithm>

#include "../tests.h"

#include "../include/agglomeration_handler.h"

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
  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

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
class MyFunction : public Function<dim>
{
public:
  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;
};

template <int dim>
double
MyFunction<dim>::value(const Point<dim> &p, const unsigned int) const
{
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
  SparseMatrix<double>                   system_matrix;
  Vector<double>                         solution;
  Vector<double>                         system_rhs;
  std::unique_ptr<GridTools::Cache<dim>> cached_tria;
  std::unique_ptr<const Function<dim>>   rhs_function;

public:
  Poisson(const unsigned int, const unsigned int fe_degree = 1);
  void
  run();

  double penalty_constant = 10.; // =1 => bad; = 10 => 4e-2; // 100 => 4e-1
  unsigned int n_subdomains;
};



template <int dim>
Poisson<dim>::Poisson(const unsigned int n_subdomains,
                      const unsigned int fe_degree)
  : mapping(1)
  , dg_fe(fe_degree)
  , n_subdomains(n_subdomains)
{}

template <int dim>
void
Poisson<dim>::make_grid()
{
  // GridGenerator::hyper_ball(tria);
  // tria.refine_global(5); // 4
  GridGenerator::hyper_cube(tria, 0., 1.);
  tria.refine_global(6); // 3
  std::cout << "Size of tria: " << tria.n_active_cells() << std::endl;
  cached_tria  = std::make_unique<GridTools::Cache<dim>>(tria, mapping);
  rhs_function = std::make_unique<const RightHandSide<dim>>();
  // Partition the triangulation with graph partitioner.

  GridTools::partition_triangulation(n_subdomains,
                                     tria,
                                     SparsityTools::Partitioner::metis);
  std::cout << "N subdomains: " << n_subdomains << std::endl;
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
      if (cell->subdomain_id() == 264)
        {
          std::cout << "Element 264 is made by cells: "
                    << cell->active_cell_index() << std::endl;
        }
      cells_per_subdomain[cell->subdomain_id()].push_back(cell);
    }

  // Agglomerate elements together
  // For every subdomain
  // for (const auto i : {0,   1,   54,  55,  57, 40,  81,  82,  208, 119, 187,
  //                     69,  238, 132, 130, 40, 52,  101, 60,  105, 209, 35,
  //                     141, 143, 128, 19,  20, 150, 123, 124, 118, 220})
  // for (const auto i : {27, 28, 29})
  for (std::size_t i = 0; i < cells_per_subdomain.size(); ++i)
    {
      std::cout << "Subdomain " << i << std::endl;
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
                  std::cout << "Neighbor idx: " << neigh->active_cell_index()
                            << std::endl;
                }

              std::cout << "Face idx from outside: " << local_face_idx_out
                        << std::endl;
            }
            std::cout << std::endl;
          }
      }
  }
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

  for (const auto &cell : ah->agglomeration_cell_iterators())
    {
      std::cout << "Cell with idx: " << cell->active_cell_index() << std::endl;
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
                  cell_matrix(i, j) += agglo_values.shape_grad(i, q_index) *
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
      std::cout << "Number of (generalized) faces: " << n_faces << std::endl;


      // const double agglo_measure =
      // bboxes[cell->active_cell_index()].volume();
      unsigned int n_jumps = 0;
      for (unsigned int f = 0; f < n_faces; ++f)
        {
          // double       hf                   = cell->face(0)->measure();
          // const double current_element_size = std::fabs(ah->volume(cell));
          const double current_element_diameter = std::fabs(ah->diameter(cell));
          const double penalty =
            penalty_constant * (1. / current_element_diameter);

          if (ah->at_boundary(cell, f))
            {
              std::cout << "at boundary!" << std::endl;
              const auto &fe_face = ah->reinit(cell, f);

              const unsigned int dofs_per_cell = fe_face.dofs_per_cell;
              std::cout << "With dofs_per_cell =" << fe_face.dofs_per_cell
                        << std::endl;
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
                             (penalty)*fe_face.shape_value(i, q_index) *
                               fe_face.shape_value(j, q_index)) *
                            fe_face.JxW(q_index);
                        }
                    }
                }

              // distribute DoFs
              cell->get_dof_indices(local_dof_indices_bdary_cell);
              constraints.distribute_local_to_global(
                cell_matrix, local_dof_indices_bdary_cell, system_matrix);
            }
          else
            {
              const auto &neigh_cell = ah->agglomerated_neighbor(cell, f);
#ifdef AGGLO_DEBUG
              std::cout << "Neighbor is " << neigh_cell->active_cell_index()
                        << std::endl;
#endif
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
#ifdef AGGLO_DEBUG
                  std::cout << "Neighbor of neighbor is:" << nofn << std::endl;
#endif
                  const auto &fe_faces =
                    ah->reinit_interface(cell, neigh_cell, f, nofn);
#ifdef AGGLO_DEBUG
                  std::cout << "Reinited the interface:" << nofn << std::endl;
#endif
                  const auto &fe_faces0 = fe_faces.first;
                  const auto &fe_faces1 = fe_faces.second;

#ifdef AGGLO_DEBUG
                  std::cout << "Local from current: " << f << std::endl;
                  std::cout << "Local from neighbor: " << nofn << std::endl;

                  std::cout << "Jump between " << cell->active_cell_index()
                            << " and " << neigh_cell->active_cell_index()
                            << std::endl;
                  {
                    std::cout << "Dalla prima i qpoints sono: " << std::endl;
                    for (const auto &q : fe_faces0.get_quadrature_points())
                      std::cout << q << std::endl;
                    std::cout << "Dalla seconda i qpoints sono: " << std::endl;
                    for (const auto &q : fe_faces1.get_quadrature_points())
                      std::cout << q << std::endl;
                  }
#endif
                  // Assert(std::fabs(
                  //          std::accumulate(fe_faces0.get_JxW_values().begin(),
                  //                          fe_faces0.get_JxW_values().end(),
                  //                          0.) -
                  //          .125) < 1e-15,
                  //        ExcMessage("Just testing!"));

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

                  // distribute DoFs accordingly
                  std::cout << "Neighbor is " << neigh_cell->active_cell_index()
                            << std::endl;
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
    const std::string filename = "interpolated_solution_ball_metis.vtu";
    std::ofstream     output(filename);

    DataOut<dim> data_out;
    ah->setup_output_interpolation_matrix();
    Vector<double> interpolated_solution(ah->output_dh.n_dofs());
    ah->output_interpolation_matrix.vmult(interpolated_solution, solution);
    data_out.attach_dof_handler(ah->output_dh);
    data_out.add_data_vector(interpolated_solution,
                             "u",
                             DataOut<dim>::type_dof_data);

    Vector<float> agglomerated(tria.n_active_cells());
    Vector<float> agglo_idx(tria.n_active_cells());
    for (const auto &cell : tria.active_cell_iterators())
      {
        agglomerated[cell->active_cell_index()] =
          ah->master_slave_relationships[cell->active_cell_index()];
        agglo_idx[cell->active_cell_index()] = cell->subdomain_id();
      }
    data_out.add_data_vector(agglomerated,
                             "agglo_relationships",
                             DataOut<dim>::type_cell_data);
    data_out.add_data_vector(agglo_idx,
                             "agglo_idx",
                             DataOut<dim>::type_cell_data);
    data_out.build_patches(mapping);
    data_out.write_vtu(output);



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
  }

  {
    // Sanity check: v(x,y)=x
    Vector<double>  interpx(ah->get_dof_handler().n_dofs());
    Vector<double>  interpone(ah->get_dof_handler().n_dofs());
    MyFunction<dim> xfunction{};
    VectorTools::interpolate(*(ah->euler_mapping),
                             ah->get_dof_handler(),
                             xfunction,
                             interpx);
    double value = system_matrix.matrix_scalar_product(interpx, interpx);
    std::cout << "Test with f(x,y)=x:" << value << std::endl;

    interpone = 1.;
    double value_one =
      system_matrix.matrix_scalar_product(interpone, interpone);
    std::cout << "Test with 1: " << value_one << std::endl;
  }
  {
    ah->setup_output_interpolation_matrix();
    Vector<double>  interpx(ah->get_dof_handler().n_dofs());
    MyFunction<dim> xfunction{};
    VectorTools::interpolate(*(ah->euler_mapping),
                             ah->get_dof_handler(),
                             xfunction,
                             interpx);


    const auto     opP  = linear_operator(ah->output_interpolation_matrix);
    const auto     opA  = linear_operator(system_matrix);
    const auto     opPT = transpose_operator(opP);
    const auto     basis_change   = opP * opA * opPT;
    Vector<double> interpx_mapped = opP * interpx;
    Vector<double> result         = basis_change * interpx_mapped;
    // Vector<double> result(ah->output_dh.n_dofs());

    double sum = 0.;
    for (size_t i = 0; i < result.size(); i++)
      sum += result[i] * result[i];

    std::cout << "Test basis change = " << sum << std::endl;
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
  // {
  //   Poisson<2> poisson_problem{50};
  //   poisson_problem.run();
  // }
  // {
  //   Poisson<2> poisson_problem{75};
  //   poisson_problem.run();
  // }
  // {
  //   Poisson<2> poisson_problem{100};
  //   poisson_problem.run();
  // }

  // {
  //   Poisson<2> poisson_problem{120};
  //   poisson_problem.run();
  // }
  // {
  //   Poisson<2> poisson_problem{180};
  //   poisson_problem.run();
  // }

  // {
  //   Poisson<2> poisson_problem{10};
  //   poisson_problem.run();
  // }
  // {
  //   Poisson<2> poisson_problem{250};
  //   poisson_problem.run();
  // }
  // {
  //   Poisson<2> poisson_problem{150}; //L2 error:0.0179101
  //   poisson_problem.run();
  // }
  // {
  //   Poisson<2> poisson_problem{250}; // L2 error:0.0176499
  //   poisson_problem.run();
  // }
  {
    Poisson<2> poisson_problem{300}; // L2 error:L2 error:0.0102053
    poisson_problem.run();
  }

  return 0;
}