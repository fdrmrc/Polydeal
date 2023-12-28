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
    // dx = x2 - x1 and dy = y2 - y1, then the normals are (-dy, dx) and (dy,
    // -dx).
    const double dx = p3[0] - p2[0];
    const double dy = p3[1] - p2[1];
    Tensor<1, 2> normal({dy, -dx});

    std::vector<std::pair<Point<2>, Point<2>>> polygon_boundary;
    polygon_boundary.push_back({p0, p1});
    polygon_boundary.push_back({p1, p2});
    polygon_boundary.push_back({p2, p3});
    polygon_boundary.push_back({p3, p4});
    polygon_boundary.push_back({p4, p0});
    std::cout << "Pentagon, h_f = "
              << PolyUtils::compute_h_orthogonal(face, polygon_boundary, normal)
              << std::endl;
  }

  {
    // p1                p3
    // |      0.25       |
    // |<--------------->|
    // |                 |
    // p0                p2
    Point<2> p0{0., 0.};
    Point<2> p1{0., 0.0625};
    Point<2> p2{0.25, 0.};
    Point<2> p3{0.25, 0.0625};



    std::pair<Point<2>, Point<2>> face{p1, p0};
    const double                  dx = p3[0] - p2[0];
    const double                  dy = p3[1] - p2[1];
    Tensor<1, 2>                  normal({-dy, dx});

    std::vector<std::pair<Point<2>, Point<2>>> polygon_boundary;
    polygon_boundary.push_back({p0, p1});
    polygon_boundary.push_back({p2, p3});

    std::cout << "Lshape, h_f = "
              << PolyUtils::compute_h_orthogonal(face, polygon_boundary, normal)
              << std::endl;
  }

  {
    // General polygon created with METIS
    std::pair<Point<2>, Point<2>> face{{0.625, 0.75}, {0.625, 0.875}};
    Tensor<1, 2>                  normal({1, 0});

    std::vector<std::pair<Point<2>, Point<2>>> polygon_boundary;
    polygon_boundary.push_back({{0.25, 0.5}, {0.25, 0.625}});
    polygon_boundary.push_back({{0.25, 0.5}, {0.375, 0.5}});
    polygon_boundary.push_back({{0.25, 0.625}, {0.375, 0.625}});
    polygon_boundary.push_back({{0.25, 0.625}, {0.375, 0.625}});
    polygon_boundary.push_back({{0.375, 0.5}, {0.5, 0.5}});
    polygon_boundary.push_back({{0.375, 0.625}, {0.375, 0.75}});
    polygon_boundary.push_back({{0.375, 0.75}, {0.375, 0.875}});
    polygon_boundary.push_back({{0.375, 0.875}, {0.375, 1}});
    polygon_boundary.push_back({{0.375, 1}, {0.5, 1}});
    polygon_boundary.push_back({{0.5, 0.5}, {0.625, 0.5}});
    polygon_boundary.push_back({{0.75, 0.5}, {0.75, 0.625}});
    polygon_boundary.push_back({{0.625, 0.5}, {0.75, 0.5}});
    polygon_boundary.push_back({{0.75, 0.625}, {0.75, 0.75}});
    polygon_boundary.push_back({{0.625, 0.75}, {0.75, 0.75}});
    polygon_boundary.push_back({{0.625, 0.75}, {0.625, 0.875}});
    polygon_boundary.push_back({{0.625, 0.875}, {0.625, 1}});
    polygon_boundary.push_back({{0.5, 1}, {0.625, 1}});

    std::cout << "Generic poly, h_f = "
              << PolyUtils::compute_h_orthogonal(face, polygon_boundary, normal)
              << std::endl;
  }
}

int
main()
{
  test();
}
