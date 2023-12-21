/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2022 by the deal.II authors
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

#include <agglomeration_handler.h>
#include <poly_utils.h>


void
test()
{
  { // Pentagon
    Point<2> p0{0., 0.};
    Point<2> p1{1., 0.};
    Point<2> p2{1., 1.};
    Point<2> p3{0.5, 2.};
    Point<2> p4{0., 1.};

    std::pair<Point<2>, Point<2>> face{p2, p3};

    std::vector<std::pair<Point<2>, Point<2>>> polygon_boundary;
    polygon_boundary.push_back({p0, p1});
    polygon_boundary.push_back({p1, p2});
    polygon_boundary.push_back({p2, p3});
    polygon_boundary.push_back({p3, p4});
    polygon_boundary.push_back({p4, p0});
    std::cout << "Pentagon, h_f = "
              << PolyUtils::compute_h_orthogonal(face, polygon_boundary)
              << std::endl;
  }

  {
    // Lshape element
    Point<2> p0{0., 0.};
    Point<2> p1{1., 0.};
    Point<2> p2{1., .5};
    Point<2> p3{.5, .5};
    Point<2> p4{5., 1.};
    Point<2> p5{0., 1.};

    std::pair<Point<2>, Point<2>> face{p2, p3};

    std::vector<std::pair<Point<2>, Point<2>>> polygon_boundary;
    polygon_boundary.push_back({p0, p1});
    polygon_boundary.push_back({p1, p2});
    polygon_boundary.push_back({p2, p3});
    polygon_boundary.push_back({p3, p4});
    polygon_boundary.push_back({p4, p5});
    polygon_boundary.push_back({p5, p0});
    std::cout << "Lshape, h_f = "
              << PolyUtils::compute_h_orthogonal(face, polygon_boundary)
              << std::endl;
  }
}

int
main()
{
  test();
}
