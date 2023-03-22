// ---------------------------------------------------------------------
//
// Copyright (C) 2018 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/function_lib.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <non_nested_transfer.h>

#include <algorithm>

using namespace dealii;


template <int dim, int spacedim = dim>
void
test_prolongation(const unsigned int        n_refinements,
                  const Function<dim>      &function,
                  const FiniteElement<dim> &fe_space = FE_Q<dim>(1))
{
  Triangulation<dim, spacedim> coarse_tria, tria;
  GridGenerator::hyper_cube(coarse_tria, -1. + .2, 1. + .2);
  GridTools::rotate(numbers::PI_4 * 0.5, coarse_tria);
  coarse_tria.refine_global(static_cast<unsigned int>(n_refinements / 2));
  DoFHandler<dim, spacedim> coarse_dh(coarse_tria);
  coarse_dh.distribute_dofs(fe_space);
  GridTools::Cache<dim, spacedim> coarse_cache(coarse_tria);

  GridGenerator::hyper_cube(tria, -2, +2);
  tria.refine_global(n_refinements);
  GridTools::Cache<dim, spacedim> fine_cache(tria);
  DoFHandler<dim, spacedim>       fine_dh(tria);
  fine_dh.distribute_dofs(fe_space);

  {
    // Just print the grids.
    std::ofstream coarse_filename("coarse_tria.vtk");
    std::ofstream fine_filename("fine_tria.vtk");
    GridOut       go;
    go.write_vtk(coarse_tria, coarse_filename);
    go.write_vtk(tria, fine_filename);
  }


  Vector<double> vec(coarse_dh.n_dofs());
  VectorTools::interpolate(coarse_dh, function, vec);
  Vector<double> dst(fine_dh.n_dofs());
  // Prolongate
  non_nested_prolongation(
    coarse_cache, fine_cache, coarse_dh, fine_dh, fe_space, vec, dst);

  DataOut<2> data_out;
  data_out.attach_dof_handler(coarse_dh);
  data_out.add_data_vector(vec, "solution");
  data_out.build_patches();
  std::ofstream output_coarse("coarse_sol.vtk");
  data_out.write_vtk(output_coarse);
  data_out.clear();
  std::ofstream output_fine("prolonged_sol.vtk");
  data_out.attach_dof_handler(fine_dh);
  data_out.add_data_vector(dst, "prolonged_solution");
  data_out.build_patches();
  data_out.write_vtk(output_fine);
}

int
main()
{
  const unsigned int n_refinements = 5; // 3
  // FE_DGQ<2>          fe_space(2);
  FE_Q<2>                        fe_space(2);
  Functions::ConstantFunction<2> constant(1.);
  Functions::Monomial<2> bilinear_function(Tensor<1, 2>({1, 1})); // f(x,y)=xy
  Functions::Monomial<2> linear_function(Tensor<1, 2>({1, 0}));   // f(x,y)= x
  test_prolongation<2>(n_refinements, constant, fe_space);
  test_prolongation<2>(n_refinements, linear_function, fe_space);
  test_prolongation<2>(n_refinements, bilinear_function, fe_space);
}
