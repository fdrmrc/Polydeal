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
  tria.refine_global(6); // 3
  cached_tria = std::make_unique<GridTools::Cache<dim>>(tria, mapping);
  // Partition the triangulation with graph partitioner.

  GridTools::partition_triangulation(n_subdomains,
                                     tria,
                                     SparsityTools::Partitioner::metis);
  std::cout << "N subdomains: " << n_subdomains << std::endl;
  constraints.close();


  // Check number of agglomerates
  {
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
LaplaceOperator<dim>::setup_agglomeration()
{
  ah = std::make_unique<AgglomerationHandler<dim>>(*cached_tria);

  std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>>
    cells_per_subdomain(n_subdomains);
  for (const auto &cell : tria.active_cell_iterators())
    cells_per_subdomain[cell->subdomain_id()].push_back(cell);

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



  // #ifdef AGGLO_DEBUG
  //   {
  //     for (const auto &cell : ah->agglomeration_cell_iterators())
  //       {
  //         std::cout << "Cell with idx: " << cell->active_cell_index()
  //                   << std::endl;
  //         unsigned int n_agglomerated_faces_per_cell = ah->n_faces(cell);
  //         std::cout << "Number of faces for the agglomeration: "
  //                   << n_agglomerated_faces_per_cell << std::endl;
  //         for (unsigned int f = 0; f < n_agglomerated_faces_per_cell; ++f)
  //           {
  //             std::cout << "Agglomerated face with idx: " << f << std::endl;
  //             const auto &[local_face_idx, neigh, local_face_idx_out, dummy]
  //             =
  //               ah->get_agglomerated_connectivity()[{cell, f}];
  //             {
  //               std::cout << "Face idx: " << local_face_idx << std::endl;
  //               if (neigh.state() == IteratorState::valid)
  //                 {
  //                   std::cout << "Neighbor idx: " <<
  //                   neigh->active_cell_index()
  //                             << std::endl;
  //                 }

  //               std::cout << "Face idx from outside: " << local_face_idx_out
  //                         << std::endl;
  //             }
  //             std::cout << std::endl;
  //           }
  //       }
  //   }
  // #endif
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
  // const auto &                         bboxes = ah->get_bboxes();

  for (const auto &cell : ah->agglomeration_cell_iterators())
    {
      // std::cout << "Cell with idx: " << cell->active_cell_index() <<
      // std::endl;
      cell_matrix              = 0;
      cell_rhs                 = 0;
      const auto &agglo_values = ah->reinit(cell);

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
      // unsigned int n_jumps = 0;
      for (unsigned int f = 0; f < n_faces; ++f)
        {
          // double       hf                   = cell->face(0)->measure();
          // const double current_element_size = std::fabs(ah->volume(cell));
          const double current_element_diameter = std::fabs(ah->diameter(cell));
          // const double penalty =
          //   penalty_constant * (1. / current_element_diameter);

          if (ah->at_boundary(cell, f))
            {
              const auto &fe_face = ah->reinit(cell, f);

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
              cell->get_dof_indices(local_dof_indices_bdary_cell);
              constraints.distribute_local_to_global(
                cell_matrix, local_dof_indices_bdary_cell, system_matrix);
            }
          else
            {
              const auto &neigh_cell = ah->agglomerated_neighbor(cell, f);

              const double neigh_element_diameter =
                std::fabs(ah->diameter(neigh_cell));
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

                  // #ifdef AGGLO_DEBUG
                  //                   std::cout << "Local from current: " << f
                  //                   << std::endl; std::cout << "Local from
                  //                   neighbor: " << nofn << std::endl;

                  //                   std::cout << "Jump between " <<
                  //                   cell->active_cell_index()
                  //                             << " and " <<
                  //                             neigh_cell->active_cell_index()
                  //                             << std::endl;
                  //                   {
                  //                     std::cout << "Dalla prima i qpoints
                  //                     sono: " << std::endl; for (const auto
                  //                     &q : fe_faces0.get_quadrature_points())
                  //                       std::cout << q << std::endl;
                  //                     std::cout << "Dalla seconda i qpoints
                  //                     sono: " << std::endl; for (const auto
                  //                     &q : fe_faces1.get_quadrature_points())
                  //                       std::cout << q << std::endl;
                  //                   }
                  // #endif

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
                  // std::cout << "Neighbor is " <<
                  // neigh_cell->active_cell_index()
                  //           << std::endl;
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
LaplaceOperator<dim>::perform_sanity_check()
{
  {
    // Sanity check: v(x,y)=x
    Vector<double>      interpx(ah->get_dof_handler().n_dofs());
    Vector<double>      interpxplusy(ah->get_dof_handler().n_dofs());
    Vector<double>      interpone(ah->get_dof_handler().n_dofs());
    LinearFunction<dim> xfunction{{1, 0}};
    LinearFunction<dim> xplusfunction{{1, 1}};
    VectorTools::interpolate(*(ah->euler_mapping),
                             ah->get_dof_handler(),
                             xfunction,
                             interpx);
    VectorTools::interpolate(*(ah->euler_mapping),
                             ah->get_dof_handler(),
                             xplusfunction,
                             interpxplusy);
    const double valuex = system_matrix.matrix_scalar_product(interpx, interpx);
    const double valuexplusy =
      system_matrix.matrix_scalar_product(interpxplusy, interpxplusy);
    std::cout << "Test with f(x,y)=x:" << valuex << std::endl;
    std::cout << "Test with f(x,y)=x+y:" << valuexplusy << std::endl;

    interpone = 1.;
    double value_one =
      system_matrix.matrix_scalar_product(interpone, interpone);
    std::cout << "Test with 1: " << value_one << std::endl;
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
  for (const unsigned int n_agglomerates : {50, 100, 120})
    {
      LaplaceOperator<2> sanity_check{n_agglomerates};
      sanity_check.run();
    }

  return 0;
}