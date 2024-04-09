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

// Compute the h_orthogonal quantity for a some polygonal shapes.

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <agglomeration_handler.h>
#include <poly_utils.h>

template <int dim>
void
test()
{
  std::cout << "Testing with dim = " << dim << std::endl;
  std::vector<std::pair<std::string, std::string>> names_and_args;
  if constexpr (dim == 2)
    {
      names_and_args.emplace_back("hyper_cube", "0.0 : 1.0 : false");
      names_and_args.emplace_back("hyper_ball", "0.,0. : 1. : false");
      names_and_args.emplace_back("hyper_L", "0.0 : 1.0 : false");
    }
  else
    {
      names_and_args.emplace_back("hyper_cube", "0.0 : 1.0 : false");
      names_and_args.emplace_back("hyper_L", "0.0 : 1.0 : false");
      names_and_args.emplace_back("hyper_ball", "0.,0.,0. : 1. : false");
    }



  for (const auto &[name, arg] : names_and_args)
    {
      Triangulation<dim> tria;
      GridGenerator::generate_from_name_and_arguments(tria, name, arg);
      std::ofstream out(name + ".vtk");
      GridOut       grid_out;
      grid_out.write_vtk(tria, out);
      FE_Nothing<dim> dummy_fe;
      DoFHandler<dim> dh(tria);
      dh.distribute_dofs(dummy_fe);
      FEFaceValues<dim> face_values(dummy_fe,
                                    QGauss<dim - 1>{1},
                                    update_normal_vectors);

      auto                           face_it = tria.begin_active_face();
      Tensor<1, dim>                 normal;
      std::vector<decltype(face_it)> polygon_boundary;


      auto first_cell = dh.begin_active();
      //  0-th cell of the hyper ball is not on the boundary
      if (name.compare("hyper_ball") == 0 && dim == 3)
        ++first_cell;

      unsigned int face_index =
        0; // the face number of the triangulations seen as a polytope.
      for (unsigned int f : first_cell->face_indices())
        {
          if (first_cell->face(f)->at_boundary())
            {
              face_values.reinit(first_cell, f);
              normal  = face_values.normal_vector(0);
              face_it = first_cell->face(f);
              break;
            }
        }

      for (const auto &cell : tria.active_cell_iterators())
        {
          for (unsigned int f : cell->face_indices())
            {
              if (cell->face(f)->at_boundary())
                polygon_boundary.push_back(cell->face(f));
            }
        }

      std::cout << "h_f for " + name + " = "
                << PolyUtils::compute_h_orthogonal(face_index,
                                                   polygon_boundary,
                                                   normal)
                << std::endl;
    }
}

int
main()
{
  test<2>();
  test<3>();
}
