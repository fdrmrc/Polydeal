#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/data_out.h>

#include <agglomeration_handler.h>

#include <algorithm>


template <int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide()
    : Function<dim>()
  {}

  virtual void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<double>           &values,
             const unsigned int /*component*/) const override;
};


template <int dim>
void
RightHandSide<dim>::value_list(const std::vector<Point<dim>> &points,
                               std::vector<double>           &values,
                               const unsigned int /*component*/) const
{
  for (unsigned int i = 0; i < values.size(); ++i)
    values[i] = 8. * numbers::PI * numbers::PI *
                std::sin(2. * numbers::PI * points[i][0]) *
                std::sin(2. * numbers::PI * points[i][1]);
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
  distribute_jumps_and_averages(
    FEFaceValues<dim>                                    &fe_face,
    FEFaceValues<dim>                                    &fe_face1,
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    const unsigned int                                    f);
  void
  assemble_system();
  void
  solve();
  void
  output_results();


  Triangulation<dim>                         tria;
  MappingQ<dim>                              mapping;
  FE_DGQ<dim>                                dg_fe;
  DoFHandler<dim>                            classical_dh;
  std::unique_ptr<AgglomerationHandler<dim>> ah;
  SparsityPattern                            sparsity;
  SparseMatrix<double>                       system_matrix;
  Vector<double>                             solution;
  Vector<double>                             system_rhs;
  std::unique_ptr<GridTools::Cache<dim>>     cached_tria;
  std::unique_ptr<const Function<dim>>       rhs_function;

public:
  Poisson();
  void
  run();

  double penalty = 3.;
};



template <int dim>
Poisson<dim>::Poisson()
  : mapping(1)
  , dg_fe(1)
  , classical_dh(tria)
{}

template <int dim>
void
Poisson<dim>::make_grid()
{
  GridGenerator::hyper_cube(tria, -1, 1);
  // tria.refine_global(2);
  tria.refine_global(1);
  classical_dh.distribute_dofs(dg_fe);
  cached_tria  = std::make_unique<GridTools::Cache<dim>>(tria, mapping);
  rhs_function = std::make_unique<const RightHandSide<dim>>();
}



template <int dim>
void
Poisson<dim>::setup_agglomeration()
{
  std::vector<types::global_cell_index> idxs_to_be_agglomerated = {0, 1, 2, 3};

  std::vector<typename Triangulation<dim>::active_cell_iterator>
    cells_to_be_agglomerated;

  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated,
                                             cells_to_be_agglomerated);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated2 = {4, 5, 6, 7};

  std::vector<typename Triangulation<dim>::active_cell_iterator>
    cells_to_be_agglomerated2;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated2,
                                             cells_to_be_agglomerated2);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated3 = {8,
                                                                    9,
                                                                    10,
                                                                    11};

  std::vector<typename Triangulation<dim>::active_cell_iterator>
    cells_to_be_agglomerated3;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated3,
                                             cells_to_be_agglomerated3);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated4 = {12,
                                                                    13,
                                                                    14,
                                                                    15};

  std::vector<typename Triangulation<dim>::active_cell_iterator>
    cells_to_be_agglomerated4;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated4,
                                             cells_to_be_agglomerated4);
  // Agglomerate the cells just stored
  ah = std::make_unique<AgglomerationHandler<dim>>(*cached_tria);
  ah->define_agglomerate(cells_to_be_agglomerated);
  ah->define_agglomerate(cells_to_be_agglomerated2);
  ah->define_agglomerate(cells_to_be_agglomerated3);
  ah->define_agglomerate(cells_to_be_agglomerated4);
  ah->distribute_agglomerated_dofs(dg_fe);
  ah->create_agglomeration_sparsity_pattern(sparsity);
  // // sanity check
  // for (const auto &cell : ah->agglo_dh.active_cell_iterators())
  //   {
  //     const unsigned int agglo_faces_per_cell =
  //       ah->n_agglomerated_faces_per_cell(cell);
  //     std::cout << "Number of agglomerated faces for cell "
  //               << cell->active_cell_index() << " is " <<
  //               agglo_faces_per_cell
  // << std::endl;

  std::ofstream out("grid_coarsen.vtk");
  GridOut       grid_out;
  grid_out.write_vtk(tria, out);
  std::cout << "Grid written " << std::endl;
}

template <int dim>
void
Poisson<dim>::distribute_jumps_and_averages(
  FEFaceValues<dim>                                    &fe_face,
  FEFaceValues<dim>                                    &fe_face1,
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  const unsigned int                                    f)
{
  if (cell->face(f)->at_boundary())
    {
      std::cout << "at boundary" << std::endl;
      fe_face.reinit(cell, f);
      const unsigned int dofs_per_cell = fe_face.dofs_per_cell;
      FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices_bdary_cell(
        dofs_per_cell);
      const double hf = cell->face(f)->measure();

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
                       fe_face.shape_grad(j, q_index) * normals[q_index] -
                     fe_face.shape_grad(i, q_index) * normals[q_index] *
                       fe_face.shape_value(j, q_index) +
                     (penalty / hf) * fe_face.shape_value(i, q_index) *
                       fe_face.shape_value(j, q_index)) *
                    fe_face.JxW(q_index);
                }
            }
        }

      // distribute DoFs
      cell->get_dof_indices(local_dof_indices_bdary_cell);
      system_matrix.add(local_dof_indices_bdary_cell, cell_matrix);
    }
  else
    {
      fe_face.reinit(cell, f);
      fe_face1.reinit(cell->neighbor(f), cell->neighbor_of_neighbor(f));

      const unsigned int dofs_per_cell = fe_face.dofs_per_cell;
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices1(dofs_per_cell);

      const double       hf = cell->face(f)->measure();
      FullMatrix<double> A_00(dofs_per_cell, dofs_per_cell);
      FullMatrix<double> A_01(dofs_per_cell, dofs_per_cell);
      FullMatrix<double> A_10(dofs_per_cell, dofs_per_cell);
      FullMatrix<double> A_11(dofs_per_cell, dofs_per_cell);


      // Get normal vectors seen from each agglomeration.
      const auto &normals = fe_face1.get_normal_vectors();
      // A00
      for (unsigned int q_index : fe_face.quadrature_point_indices())
        {
          std::cout << normals[q_index] << std::endl;
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  A_00(i, j) +=
                    (-0.5 * fe_face.shape_grad(i, q_index) * normals[q_index] *
                       fe_face.shape_value(j, q_index) -
                     0.5 * fe_face.shape_grad(j, q_index) * normals[q_index] *
                       fe_face.shape_value(i, q_index) +
                     (penalty / hf) * fe_face.shape_value(i, q_index) *
                       fe_face.shape_value(j, q_index)) *
                    fe_face.JxW(q_index);
                }
            }
        }
      // A10
      for (unsigned int q_index : fe_face.quadrature_point_indices())
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  A_01(i, j) +=
                    (+0.5 * fe_face.shape_grad(i, q_index) * normals[q_index] *
                       fe_face1.shape_value(j, q_index) -
                     0.5 * fe_face1.shape_grad(j, q_index) * normals[q_index] *
                       fe_face.shape_value(i, q_index) -
                     (penalty / hf) * fe_face.shape_value(i, q_index) *
                       fe_face1.shape_value(j, q_index)) *
                    fe_face1.JxW(q_index);
                }
            }
        }
      // A01
      for (unsigned int q_index : fe_face.quadrature_point_indices())
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  A_10(i, j) +=
                    (+0.5 * fe_face1.shape_grad(i, q_index) * normals[q_index] *
                       fe_face.shape_value(j, q_index) -
                     0.5 * fe_face.shape_grad(j, q_index) * normals[q_index] *
                       fe_face1.shape_value(i, q_index) -
                     (penalty / hf) * fe_face1.shape_value(i, q_index) *
                       fe_face.shape_value(j, q_index)) *
                    fe_face1.JxW(q_index);
                }
            }
        }
      // A11
      for (unsigned int q_index : fe_face.quadrature_point_indices())
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  A_11(i, j) +=
                    (-0.5 * fe_face1.shape_grad(i, q_index) * normals[q_index] *
                       fe_face1.shape_value(j, q_index) -
                     0.5 * fe_face1.shape_grad(j, q_index) * normals[q_index] *
                       fe_face1.shape_value(i, q_index) +
                     (penalty / hf) * fe_face1.shape_value(i, q_index) *
                       fe_face1.shape_value(j, q_index)) *
                    fe_face1.JxW(q_index);
                }
            }
        }

      // distribute DoFs accordingly
      cell->get_dof_indices(local_dof_indices);
      // Need to convert the neighbor to a valid "dh" iterator
      typename DoFHandler<dim>::cell_iterator neigh_dh(*(cell->neighbor(f)),
                                                       &classical_dh);
      // typename DoFHandler<dim>::cell_iterator neigh_dh(*(cell->neighbor(f)),
      //                                                  &(ah->agglo_dh));
      std::cout << "Neighbor is " << neigh_dh->active_cell_index() << std::endl;
      neigh_dh->get_dof_indices(local_dof_indices1);

      system_matrix.add(local_dof_indices, A_00);
      system_matrix.add(local_dof_indices, local_dof_indices1, A_01);
      system_matrix.add(local_dof_indices1, local_dof_indices, A_10);
      system_matrix.add(local_dof_indices1, A_11);
    }
}

template <int dim>
void
Poisson<dim>::assemble_system()
{
  DynamicSparsityPattern dsp(classical_dh.n_dofs());
  DoFTools::make_flux_sparsity_pattern(classical_dh, dsp);
  sparsity.copy_from(dsp);

  system_matrix.reinit(sparsity);
  solution.reinit(classical_dh.n_dofs());
  system_rhs.reinit(classical_dh.n_dofs());

  const unsigned int quadrature_degree = 3;
  // ah->set_quadrature_degree(quadrature_degree);
  // ah->set_agglomeration_flags(update_values | update_JxW_values |
  //                             update_gradients | update_quadrature_points);
  FEFaceValues<dim> fe_face(mapping,
                            dg_fe,
                            QGauss<dim - 1>(quadrature_degree),
                            update_values | update_JxW_values |
                              update_gradients | update_quadrature_points |
                              update_normal_vectors);


  FEValues<dim> fe_values(mapping,
                          dg_fe,
                          QGauss<dim>(quadrature_degree),
                          update_values | update_JxW_values | update_gradients |
                            update_quadrature_points);

  FEFaceValues<dim>  fe_face1(mapping,
                             dg_fe,
                             QGauss<dim - 1>(quadrature_degree),
                             update_values | update_JxW_values |
                               update_gradients | update_quadrature_points |
                               update_normal_vectors);
  const unsigned int dofs_per_cell = dg_fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


  // Master cells of the mesh
  // for (const auto &cell :
  //      ah->agglo_dh.active_cell_iterators() |
  //        IteratorFilters::ActiveFEIndexEqualTo(ah->AggloIndex::master))
  //   {
  //     std::cout << "Cell with idx: " << cell->active_cell_index() <<
  //     std::endl; cell_matrix              = 0; cell_rhs                 = 0;
  //     const auto &agglo_values = ah->reinit(cell);

  //     const auto         &q_points  = agglo_values.get_quadrature_points();
  //     const unsigned int  n_qpoints = q_points.size();
  //     std::vector<double> rhs(n_qpoints);
  //     rhs_function->value_list(q_points, rhs);

  //     for (unsigned int q_index : agglo_values.quadrature_point_indices())
  //       {
  //         for (unsigned int i = 0; i < dofs_per_cell; ++i)
  //           {
  //             for (unsigned int j = 0; j < dofs_per_cell; ++j)
  //               {
  //                 cell_matrix(i, j) += agglo_values.shape_grad(i, q_index) *
  //                                      agglo_values.shape_grad(j, q_index) *
  //                                      agglo_values.JxW(q_index);
  //               }
  //             cell_rhs(i) += agglo_values.shape_value(i, q_index) *
  //                            rhs[q_index] * agglo_values.JxW(q_index);
  //           }
  //       }
  //     cell->get_dof_indices(local_dof_indices);

  //     // distribute DoFs
  //     for (unsigned int i = 0; i < dofs_per_cell; ++i)
  //       {
  //         for (unsigned int j = 0; j < dofs_per_cell; ++j)
  //           {
  //             system_matrix.add(local_dof_indices[i],
  //                               local_dof_indices[j],
  //                               cell_matrix(i, j));
  //           }
  //         system_rhs(local_dof_indices[i]) += cell_rhs(i);
  //       }



  //     // Face terms for master cells, i.e.: terms that are agglomerated
  //     const unsigned int n_agglomerated_faces =
  //       ah->n_faces(cell);
  //     std::cout << "Number of agglomerated faces per agglomeration: "
  //               << n_agglomerated_faces << std::endl;
  //     for (unsigned int f = 0; f < n_agglomerated_faces; ++f)
  //       {
  //         // We need to retrieve information about the deal.II face shared
  //         with
  //         // this agglomerated face. This can be asked to master_neighbors,
  //         that
  //         // takes precisely the master cell of the agglomeration and the
  //         index
  //         // of the **agglomerated** face.
  //         const auto &[local_face_idx,
  //                      neighboring_cell,
  //                      local_face_idx_outside] =
  //           ah->master_neighbors[{cell, f}];

  //         const auto &fe_isv = ah->reinit(cell, f);
  //         fe_face.reinit(neighboring_cell, local_face_idx_outside);

  //         std::vector<types::global_dof_index> local_dof_indices1(
  //           dofs_per_cell);

  //         const double hf =
  //           neighboring_cell->face(local_face_idx_outside)->measure();
  //         FullMatrix<double> A_00(dofs_per_cell, dofs_per_cell);
  //         FullMatrix<double> A_01(dofs_per_cell, dofs_per_cell);
  //         FullMatrix<double> A_10(dofs_per_cell, dofs_per_cell);
  //         FullMatrix<double> A_11(dofs_per_cell, dofs_per_cell);



  //         // Get normal vectors seen from each agglomeration.
  //         const auto &normals = fe_isv.get_normal_vectors();
  //         // A00
  //         for (unsigned int q_index : fe_isv.quadrature_point_indices())
  //           {
  //             std::cout << normals[q_index] << std::endl;
  //             for (unsigned int i = 0; i < dofs_per_cell; ++i)
  //               {
  //                 for (unsigned int j = 0; j < dofs_per_cell; ++j)
  //                   {
  //                     A_00(i, j) +=
  //                       (-0.5 * fe_isv.shape_grad(i, q_index) *
  //                          normals[q_index] * fe_isv.shape_value(j, q_index)
  //                          -
  //                        0.5 * fe_isv.shape_grad(j, q_index) *
  //                          normals[q_index] * fe_isv.shape_value(i, q_index)
  //                          +
  //                        (penalty / hf) * fe_isv.shape_value(i, q_index) *
  //                          fe_isv.shape_value(j, q_index)) *
  //                       fe_isv.JxW(q_index);
  //                   }
  //               }
  //           }
  //         // A10
  //         for (unsigned int q_index : fe_isv.quadrature_point_indices())
  //           {
  //             for (unsigned int i = 0; i < dofs_per_cell; ++i)
  //               {
  //                 for (unsigned int j = 0; j < dofs_per_cell; ++j)
  //                   {
  //                     A_01(i, j) +=
  //                       (-0.5 * fe_face.shape_grad(i, q_index) *
  //                          normals[q_index] * fe_isv.shape_value(j, q_index)
  //                          +
  //                        0.5 * fe_isv.shape_grad(j, q_index) *
  //                          normals[q_index] * fe_face.shape_value(i, q_index)
  //                          -
  //                        (penalty / hf) * fe_face.shape_value(i, q_index) *
  //                          fe_isv.shape_value(j, q_index)) *
  //                       fe_isv.JxW(q_index);
  //                   }
  //               }
  //           }
  //         // A01
  //         for (unsigned int q_index : fe_isv.quadrature_point_indices())
  //           {
  //             for (unsigned int i = 0; i < dofs_per_cell; ++i)
  //               {
  //                 for (unsigned int j = 0; j < dofs_per_cell; ++j)
  //                   {
  //                     A_10(i, j) +=
  //                       (-0.5 * fe_isv.shape_grad(i, q_index) *
  //                          normals[q_index] * fe_face.shape_value(j, q_index)
  //                          -
  //                        0.5 * fe_face.shape_grad(j, q_index) *
  //                          normals[q_index] * fe_isv.shape_value(i, q_index)
  //                          -
  //                        (penalty / hf) * fe_isv.shape_value(i, q_index) *
  //                          fe_face.shape_value(j, q_index)) *
  //                       fe_isv.JxW(q_index);
  //                   }
  //               }
  //           }
  //         // A11
  //         for (unsigned int q_index : fe_isv.quadrature_point_indices())
  //           {
  //             for (unsigned int i = 0; i < dofs_per_cell; ++i)
  //               {
  //                 for (unsigned int j = 0; j < dofs_per_cell; ++j)
  //                   {
  //                     A_11(i, j) +=
  //                       (-0.5 * fe_face.shape_grad(i, q_index) *
  //                          normals[q_index] * fe_face.shape_value(j, q_index)
  //                          -
  //                        0.5 * fe_face.shape_grad(j, q_index) *
  //                          normals[q_index] * fe_face.shape_value(i, q_index)
  //                          +
  //                        (penalty / hf) * fe_face.shape_value(i, q_index) *
  //                          fe_face.shape_value(j, q_index)) *
  //                       fe_isv.JxW(q_index);
  //                   }
  //               }
  //           }

  //         // distribute DoFs accordingly
  //         cell->get_dof_indices(local_dof_indices);
  //         // Need to convert the neighbor to a valid "dh" iterator
  //         typename DoFHandler<dim>::cell_iterator neigh_dh(*neighboring_cell,
  //                                                          &(ah->agglo_dh));
  //         std::cout << "Neighbor is " << neigh_dh->active_cell_index()
  //                   << std::endl;
  //         neigh_dh->get_dof_indices(local_dof_indices1);

  //         system_matrix.add(local_dof_indices, A_00);
  //         system_matrix.add(local_dof_indices, local_dof_indices1, A_01);
  //         system_matrix.add(local_dof_indices1, local_dof_indices, A_10);
  //         system_matrix.add(local_dof_indices1, A_11);
  //       }
  //   }



  // Loop over standard deal.II cells
  for (const auto &cell : classical_dh.active_cell_iterators())
    {
      std::cout << "Standard cell with idx: " << cell->active_cell_index()
                << std::endl;
      cell_matrix = 0;
      cell_rhs    = 0;
      // const auto &agglo_values = ah->reinit(cell);
      fe_values.reinit(cell);

      const auto         &q_points  = fe_values.get_quadrature_points();
      const unsigned int  n_qpoints = q_points.size();
      std::vector<double> rhs(n_qpoints);
      rhs_function->value_list(q_points, rhs);

      for (unsigned int q_index : fe_values.quadrature_point_indices())
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  cell_matrix(i, j) += fe_values.shape_grad(i, q_index) *
                                       fe_values.shape_grad(j, q_index) *
                                       fe_values.JxW(q_index);
                }
              cell_rhs(i) += fe_values.shape_value(i, q_index) * rhs[q_index] *
                             fe_values.JxW(q_index);
            }
        }

      // distribute DoFs
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              system_matrix.add(local_dof_indices[i],
                                local_dof_indices[j],
                                cell_matrix(i, j));
            }
          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }


      // // Add the contributions from the faces of the standard deal.II mesh.
      // const auto &inadmissible_faces_indices = ah->inadmissible_faces[cell];
      // if (inadmissible_faces_indices.empty())
      //   {
      //     std::cout << "All faces are admissible for this cell" << std::endl;
      //     // all faces are admissible, you can add contributions.
      //     for (const auto f : cell->face_indices())
      //       {
      //         distribute_jumps_and_averages(fe_face, fe_face1, cell, f);
      //       }
      //   }
      // else
      //   {
      // Some faces are constrained, you need to loop only over the ones
      // that are unconstrained.
      for (const auto f : cell->face_indices())
        {
          // if (std::find(inadmissible_faces_indices.begin(),
          //               inadmissible_faces_indices.end(),
          //               f) == inadmissible_faces_indices.end())
          //   {
          std::cout << "Face index " << f << " is admissible!" << std::endl;

          distribute_jumps_and_averages(fe_face, fe_face1, cell, f);
          // }
        }
      // }
    }
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
  const std::string filename = "agglomerated_Poisson.vtu";
  std::ofstream     output(filename);

  DataOut<dim> data_out;
  // data_out.attach_dof_handler(ah->agglo_dh);
  data_out.attach_dof_handler(classical_dh);
  data_out.add_data_vector(solution, "u", DataOut<dim>::type_dof_data);
  data_out.build_patches(mapping);
  data_out.write_vtu(output);
}

template <int dim>
void
Poisson<dim>::run()
{
  make_grid();
  // setup_agglomeration();
  assemble_system();
  {
    const std::string filename = "standar_matrix.txt";
    std::ofstream     output(filename);
    system_matrix.print(output);
  }
  solve();
  output_results();
}

int
main()
{
  Poisson<2> poisson_problem;
  poisson_problem.run();

  return 0;
}
