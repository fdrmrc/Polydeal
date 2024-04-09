// #include <deal.II/grid/grid_generator.h>
// #include <deal.II/grid/grid_out.h>

// #include <deal.II/lac/sparse_direct.h>
// #include <deal.II/lac/sparse_matrix.h>

// #include <deal.II/numerics/data_out.h>
// #include <deal.II/numerics/vector_tools_integrate_difference.h>

// #include <agglomeration_handler.h>
// #include <poly_utils.h>

// #include <algorithm>

// template <int dim>
// class RightHandSide : public Function<dim>
// {
// public:
//   RightHandSide()
//     : Function<dim>()
//   {}

//   virtual void
//   value_list(const std::vector<Point<dim>> &points,
//              std::vector<double> &          values,
//              const unsigned int /*component*/) const override;
// };

// template <int dim>
// class Solution : public Function<dim>
// {
// public:
//   virtual double
//   value(const Point<dim> &p, const unsigned int component = 0) const
//   override;

//   virtual Tensor<1, dim>
//   gradient(const Point<dim> & p,
//            const unsigned int component = 0) const override;
// };

// template <int dim>
// void
// RightHandSide<dim>::value_list(const std::vector<Point<dim>> &points,
//                                std::vector<double> &          values,
//                                const unsigned int /*component*/) const
// {
//   for (unsigned int i = 0; i < values.size(); ++i)
//     values[i] = 8. * numbers::PI * numbers::PI *
//                 std::sin(2. * numbers::PI * points[i][0]) *
//                 std::sin(2. * numbers::PI * points[i][1]);
// }


// template <int dim>
// double
// Solution<dim>::value(const Point<dim> &p, const unsigned int) const
// {
//   return std::sin(2. * numbers::PI * p[0]) * std::sin(2. * numbers::PI *
//   p[1]);
// }

// template <int dim>
// Tensor<1, dim>
// Solution<dim>::gradient(const Point<dim> &p, const unsigned int) const
// {
//   Tensor<1, dim> return_value;
//   Assert(false, ExcNotImplemented());
//   return return_value;
// }


// template <int dim>
// class TestIterator
// {
// private:
//   void
//   make_grid();
//   void
//   setup_agglomeration();
//   void
//   test1();
//   void
//   test2();
//   void
//   test3();



//   Triangulation<dim>                         tria;
//   MappingQ<dim>                              mapping;
//   FE_DGQ<dim>                                dg_fe;
//   std::unique_ptr<AgglomerationHandler<dim>> ah;
//   std::unique_ptr<GridTools::Cache<dim>>     cached_tria;

// public:
//   TestIterator();
//   void
//   run();
// };



// template <int dim>
// TestIterator<dim>::TestIterator()
//   : mapping(1)
//   , dg_fe(1)
// {
//   std::cout << "dim = " << dim << std::endl;
// }

// template <int dim>
// void
// TestIterator<dim>::make_grid()
// {
//   GridGenerator::hyper_cube(tria, -1, 1);
//   tria.refine_global(3);
//   cached_tria = std::make_unique<GridTools::Cache<dim>>(tria, mapping);
// }



// template <int dim>
// void
// TestIterator<dim>::setup_agglomeration()
// {
//   std::vector<typename Triangulation<2>::active_cell_iterator>
//     cells; // each cell = an agglomerate
//   for (const auto &cell : tria.active_cell_iterators())
//     cells.push_back(cell);


//   std::vector<types::global_cell_index> flagged_cells;
//   // Helper lambda
//   const auto store_flagged_cells =
//     [&flagged_cells](
//       const std::vector<types::global_cell_index> &idxs_to_be_agglomerated) {
//       for (const int idx : idxs_to_be_agglomerated)
//         flagged_cells.push_back(idx);
//     };

//   // std::vector<types::global_cell_index> idxs_to_be_agglomerated = {
//   //   3, 9}; //{8, 9, 10, 11};
//   std::vector<types::global_cell_index> idxs_to_be_agglomerated = {36,
//                                                                    37,
//                                                                    38,
//                                                                    39};

//   std::vector<typename Triangulation<dim>::active_cell_iterator>
//     cells_to_be_agglomerated;
//   PolyUtils::collect_cells_for_agglomeration(tria,
//                                              idxs_to_be_agglomerated,
//                                              cells_to_be_agglomerated);
//   store_flagged_cells(idxs_to_be_agglomerated);

//   std::vector<types::global_cell_index> idxs_to_be_agglomerated2 = {18, 24,
//   25};

//   std::vector<typename Triangulation<dim>::active_cell_iterator>
//     cells_to_be_agglomerated2;
//   PolyUtils::collect_cells_for_agglomeration(tria,
//                                              idxs_to_be_agglomerated2,
//                                              cells_to_be_agglomerated2);
//   store_flagged_cells(idxs_to_be_agglomerated2);


//   std::vector<types::global_cell_index> idxs_to_be_agglomerated3 = {3, 6};
//   std::vector<typename Triangulation<dim>::active_cell_iterator>
//     cells_to_be_agglomerated3;
//   PolyUtils::collect_cells_for_agglomeration(tria,
//                                              idxs_to_be_agglomerated3,
//                                              cells_to_be_agglomerated3);
//   store_flagged_cells(idxs_to_be_agglomerated3);


//   // Agglomerate the cells just stored
//   ah = std::make_unique<AgglomerationHandler<dim>>(*cached_tria);

//   for (std::size_t i = 0; i < tria.n_active_cells(); ++i)
//     {
//       // If not present, agglomerate all the singletons
//       if (std::find(flagged_cells.begin(),
//                     flagged_cells.end(),
//                     cells[i]->active_cell_index()) ==
//                     std::end(flagged_cells))
//         ah->define_agglomerate({cells[i]});
//     }

//   ah->define_agglomerate(cells_to_be_agglomerated);
//   ah->define_agglomerate(cells_to_be_agglomerated2);
//   ah->define_agglomerate(cells_to_be_agglomerated3);
//   ah->distribute_agglomerated_dofs(dg_fe);
// }



// template <int dim>
// void
// TestIterator<dim>::test1()
// {
//   const unsigned int                   dofs_per_cell = ah->n_dofs_per_cell();
//   std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
//   auto                                 polytope = ah->begin();
//   {
//     for (; polytope != ah->end(); ++polytope)
//       {
//         const unsigned int n_faces = polytope->n_faces();
//         std::cout << "n_faces =" << n_faces << std::endl;
//         polytope->get_dof_indices(local_dof_indices);
//         std::cout << "Global DoF indices for polytope " << polytope->index()
//                   << std::endl;
//         std::cout << "Master cell polytope = "
//                   << polytope.master_cell()->active_cell_index() <<
//                   std::endl;
//         for (const types::global_dof_index idx : local_dof_indices)
//           std::cout << idx << std::endl;
//       } // Loop over polytopes
//   }
//   std::cout << std::endl;
// }


// template <int dim>
// void
// TestIterator<dim>::run()
// {
//   make_grid();
//   setup_agglomeration();
//   test1();
// }

// int
// main()
// {
//   TestIterator<2> test;
//   test.run();

//   return 0;
// }

/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2022 by the polyDEAL authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 */


// Similar to continuous_face_02.cc, but with standard cells seen as
// agglomerates.


#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/data_out.h>

#include <agglomeration_handler.h>
#include <poly_utils.h>


void
perimeter_test(AgglomerationHandler<2> &ah)
{
  double perimeter = 0.;
  for (const auto &polytope : ah.polytope_iterators())
    {
      std::cout << "Master cell index = "
                << polytope.master_cell()->active_cell_index() << std::endl;
      const auto &info = ah.get_interface();

      unsigned int n_faces = polytope->n_faces();
      std::cout << "Number of agglomerated faces = " << n_faces << std::endl;
      for (unsigned int f = 0; f < n_faces; ++f)
        {
          std::cout << "Agglomerate face index = " << f << std::endl;
          if (!polytope->at_boundary(f))
            {
              const auto &neighbor = polytope->neighbor(f);
              std::cout << "Neighbor = " << neighbor->index() << std::endl;
              std::cout << "Neighbor of neighbor = "
                        << polytope->neighbor_of_agglomerated_neighbor(f)
                        << std::endl;
            }
          else
            {
              const auto &test_feisv = ah.reinit(polytope, f);
              perimeter += std::accumulate(test_feisv.get_JxW_values().begin(),
                                           test_feisv.get_JxW_values().end(),
                                           0.);
            }
        }
      std::cout << std::endl;
    }
  std::cout << "Perimeter = " << perimeter << std::endl;
}



void
test_neighbors(AgglomerationHandler<2> &ah)
{
  std::cout << "Check on neighbors and neighbors of neighbors:" << std::endl;
  for (const auto &polytope : ah.polytope_iterators())
    {
      unsigned int n_faces = polytope->n_faces();
      for (unsigned int f = 0; f < n_faces; ++f)
        {
          if (!polytope->at_boundary(f))
            {
              const auto &neighbor_polytope = polytope->neighbor(f);
              AssertThrow(neighbor_polytope
                              ->neighbor(
                                polytope->neighbor_of_agglomerated_neighbor(f))
                              ->index() == polytope->index(),
                          ExcMessage("Mismatch!"));
            }
        }
    }
  std::cout << "Ok" << std::endl;
}



void
test_face_qpoints(AgglomerationHandler<2> &ah)
{
  std::cout << "Check on quadrature points:" << std::endl;
  for (const auto &polytope : ah.polytope_iterators())
    {
      unsigned int n_faces = polytope->n_faces();
      for (unsigned int f = 0; f < n_faces; ++f)
        {
          if (!polytope->at_boundary(f))
            {
              const auto &       neigh_polytope = polytope->neighbor(f);
              const unsigned int nofn =
                polytope->neighbor_of_agglomerated_neighbor(f);
              const auto &fe_faces =
                ah.reinit_interface(polytope, neigh_polytope, f, nofn);

              const auto &fe_faces0 = fe_faces.first;
              const auto &fe_faces1 = fe_faces.second;

              const auto &points0 = fe_faces0.get_quadrature_points();
              const auto &points1 = fe_faces1.get_quadrature_points();
              for (size_t i = 0; i < fe_faces1.get_quadrature_points().size();
                   ++i)
                {
                  double d = (points0[i] - points1[i]).norm();
                  Assert(d < 1e-15,
                         ExcMessage(
                           "Face qpoints at the interface do not match!"));
                }
            }
        }
    }
  std::cout << "Ok" << std::endl;
}


void
test(const Triangulation<2> &tria, AgglomerationHandler<2> &ah)
{
  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells; // each cell = an agglomerate
  for (const auto &cell : tria.active_cell_iterators())
    cells.push_back(cell);


  std::vector<types::global_cell_index> flagged_cells;
  // Helper lambda
  const auto store_flagged_cells =
    [&flagged_cells](
      const std::vector<types::global_cell_index> &idxs_to_be_agglomerated) {
      for (const int idx : idxs_to_be_agglomerated)
        flagged_cells.push_back(idx);
    };

  // std::vector<types::global_cell_index> idxs_to_be_agglomerated = {
  //   3, 9}; //{8, 9, 10, 11};
  std::vector<types::global_cell_index> idxs_to_be_agglomerated = {36,
                                                                   37,
                                                                   38,
                                                                   39};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated,
                                             cells_to_be_agglomerated);
  store_flagged_cells(idxs_to_be_agglomerated);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated2 = {18, 24, 25};

  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated2;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated2,
                                             cells_to_be_agglomerated2);
  store_flagged_cells(idxs_to_be_agglomerated2);


  std::vector<types::global_cell_index> idxs_to_be_agglomerated3 = {3, 6};
  std::vector<typename Triangulation<2>::active_cell_iterator>
    cells_to_be_agglomerated3;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated3,
                                             cells_to_be_agglomerated3);
  store_flagged_cells(idxs_to_be_agglomerated3);


  // Agglomerate the cells just stored

  for (std::size_t i = 0; i < tria.n_active_cells(); ++i)
    {
      // If not present, agglomerate all the singletons
      if (std::find(flagged_cells.begin(),
                    flagged_cells.end(),
                    cells[i]->active_cell_index()) == std::end(flagged_cells))
        ah.define_agglomerate({cells[i]});
    }

  ah.define_agglomerate(cells_to_be_agglomerated);
  ah.define_agglomerate(cells_to_be_agglomerated2);
  ah.define_agglomerate(cells_to_be_agglomerated3);
}



int
main()
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria, -1, 1);
  MappingQ<2> mapping(1);
  tria.refine_global(3);

  GridTools::Cache<2>     cached_tria(tria, mapping);
  AgglomerationHandler<2> ah(cached_tria);
  FE_DGQ<2>               fe_dg(1);

  test(tria, ah);

  ah.distribute_agglomerated_dofs(fe_dg);
  ah.initialize_fe_values(QGauss<2>(1),
                          update_JxW_values | update_quadrature_points);

  perimeter_test(ah);
  std::cout << "- - - - - - - - - - - -" << std::endl;
  test_neighbors(ah);
  std::cout << "- - - - - - - - - - - -" << std::endl;
  test_face_qpoints(ah);
  std::cout << "End Test" << std::endl;
}
