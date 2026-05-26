// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) XXXX - YYYY by the polyDEAL authors
//
// This file is part of the polyDEAL library.
//
// Detailed license information governing the source code
// can be found in LICENSE.md at the top level directory.
//
// -----------------------------------------------------------------------------

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/point.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <vector>

#include "continuous_agglo_utils.h"

using namespace dealii;

/**
 * Test the sequential implementation of
 * create_triangulation_from_bounding_boxes
 */

// Test for 2D case
void
test_create_tria_2d_single_box()
{
  const int                     dim = 2;
  std::vector<BoundingBox<dim>> boxes;
  Point<dim>                    p1(0.0, 0.0);
  Point<dim>                    p2(1.0, 1.0);
  BoundingBox<dim>              box(std::make_pair(p1, p2));
  boxes.push_back(box);

  Triangulation<dim> tria;
  dealii::ContinuousAggloUtils::PointsAgglo::
    create_triangulation_from_bounding_boxes(tria, boxes);

  assert(tria.n_active_cells() == 1);
  assert(tria.n_vertices() == 4);

  auto cell = tria.begin_active();
  assert(cell->measure() == 1.0);

  std::cout << "2D single box test passed!" << std::endl;
}

// Test for 2D case with multiple non-overlapping boxes
void
test_create_tria_2d_multiple_boxes()
{
  const int                     dim = 2;
  std::vector<BoundingBox<dim>> boxes;

  Point<dim> p1(0.0, 0.0), p2(1.0, 1.0);
  boxes.push_back(BoundingBox<dim>(std::make_pair(p1, p2)));

  Point<dim> p3(1.0, 0.0), p4(2.0, 1.0);
  boxes.push_back(BoundingBox<dim>(std::make_pair(p3, p4)));

  Point<dim> p5(0.0, 1.0), p6(1.0, 2.0);
  boxes.push_back(BoundingBox<dim>(std::make_pair(p5, p6)));

  Point<dim> p7(1.0, 1.0), p8(2.0, 2.0);
  boxes.push_back(BoundingBox<dim>(std::make_pair(p7, p8)));

  Triangulation<dim> tria;
  dealii::ContinuousAggloUtils::PointsAgglo::
    create_triangulation_from_bounding_boxes(tria, boxes);

  assert(tria.n_active_cells() == 4);
  assert(tria.n_vertices() == 16);

  double total_area = 0.0;
  for (const auto &cell : tria.active_cell_iterators())
    total_area += cell->measure();
  assert(std::abs(total_area - 4.0) < 1e-10);

  std::cout << "2D multiple boxes test passed!" << std::endl;
}

// Test for 2D case with overlapping boxes
void
test_create_tria_2d_overlapping_boxes()
{
  const int                     dim = 2;
  std::vector<BoundingBox<dim>> boxes;

  Point<dim> p1(0.0, 0.0), p2(1.5, 1.0);
  boxes.push_back(BoundingBox<dim>(std::make_pair(p1, p2)));

  Point<dim> p3(0.5, 0.0), p4(2.0, 1.0);
  boxes.push_back(BoundingBox<dim>(std::make_pair(p3, p4)));

  Triangulation<dim> tria;
  dealii::ContinuousAggloUtils::PointsAgglo::
    create_triangulation_from_bounding_boxes(tria, boxes);

  assert(tria.n_active_cells() == 2);
  assert(tria.n_vertices() == 8);

  std::cout << "2D overlapping boxes test passed!" << std::endl;
}

// Test for 3D case
void
test_create_tria_3d_single_box()
{
  const int                     dim = 3;
  std::vector<BoundingBox<dim>> boxes;
  Point<dim>                    p1(0.0, 0.0, 0.0);
  Point<dim>                    p2(1.0, 1.0, 1.0);
  BoundingBox<dim>              box(std::make_pair(p1, p2));
  boxes.push_back(box);

  Triangulation<dim> tria;
  dealii::ContinuousAggloUtils::PointsAgglo::
    create_triangulation_from_bounding_boxes(tria, boxes);

  assert(tria.n_active_cells() == 1);
  assert(tria.n_vertices() == 8);

  auto cell = tria.begin_active();
  assert(std::abs(cell->measure() - 1.0) < 1e-10);

  std::cout << "3D single box test passed!" << std::endl;
}

// Test for 3D case with multiple boxes
void
test_create_tria_3d_multiple_boxes()
{
  const int                     dim = 3;
  std::vector<BoundingBox<dim>> boxes;

  // Create 2 unit cubes
  Point<dim> p1(0.0, 0.0, 0.0), p2(1.0, 1.0, 1.0);
  boxes.push_back(BoundingBox<dim>(std::make_pair(p1, p2)));

  Point<dim> p3(1.0, 0.0, 0.0), p4(2.0, 1.0, 1.0);
  boxes.push_back(BoundingBox<dim>(std::make_pair(p3, p4)));

  Triangulation<dim> tria;
  dealii::ContinuousAggloUtils::PointsAgglo::
    create_triangulation_from_bounding_boxes(tria, boxes);

  assert(tria.n_active_cells() == 2);
  assert(tria.n_vertices() == 16);

  double total_volume = 0.0;
  for (const auto &cell : tria.active_cell_iterators())
    total_volume += cell->measure();
  assert(std::abs(total_volume - 2.0) < 1e-10);

  std::cout << "3D multiple boxes test passed!" << std::endl;
}

// Test for 3D case with rectangular boxes (non-cubic)
void
test_create_tria_3d_rectangular_boxes()
{
  const int                     dim = 3;
  std::vector<BoundingBox<dim>> boxes;

  // Create a rectangular box (2x3x4)
  Point<dim> p1(0.0, 0.0, 0.0), p2(2.0, 3.0, 4.0);
  boxes.push_back(BoundingBox<dim>(std::make_pair(p1, p2)));

  Triangulation<dim> tria;
  dealii::ContinuousAggloUtils::PointsAgglo::
    create_triangulation_from_bounding_boxes(tria, boxes);

  assert(tria.n_active_cells() == 1);

  auto cell = tria.begin_active();
  assert(std::abs(cell->measure() - 24.0) < 1e-10); // 2*3*4 = 24

  std::cout << "3D rectangular boxes test passed!" << std::endl;
}

// Test that vertices are correctly positioned
void
test_create_tria_2d_vertex_positions()
{
  const int                     dim = 2;
  std::vector<BoundingBox<dim>> boxes;
  Point<dim>                    p1(0.5, 1.5);
  Point<dim>                    p2(2.5, 3.5);
  boxes.push_back(BoundingBox<dim>(std::make_pair(p1, p2)));

  Triangulation<dim> tria;
  dealii::ContinuousAggloUtils::PointsAgglo::
    create_triangulation_from_bounding_boxes(tria, boxes);

  std::vector<Point<dim>> expected_vertices = {Point<dim>(0.5, 1.5),
                                               Point<dim>(2.5, 1.5),
                                               Point<dim>(0.5, 3.5),
                                               Point<dim>(2.5, 3.5)};

  const auto &vertices = tria.get_vertices();
  assert(vertices.size() == expected_vertices.size());

  for (size_t i = 0; i < expected_vertices.size(); ++i)
    {
      assert(std::abs(vertices[i][0] - expected_vertices[i][0]) < 1e-10);
      assert(std::abs(vertices[i][1] - expected_vertices[i][1]) < 1e-10);
    }

  std::cout << "2D vertex positions test passed!" << std::endl;
}

int
main()
{
  std::cout << "Running tests for create_triangulation_from_bounding_boxes..."
            << std::endl;

  // 2D tests
  test_create_tria_2d_single_box();
  test_create_tria_2d_multiple_boxes();
  test_create_tria_2d_overlapping_boxes();
  test_create_tria_2d_vertex_positions();

  // 3D tests
  test_create_tria_3d_single_box();
  test_create_tria_3d_multiple_boxes();
  test_create_tria_3d_rectangular_boxes();

  std::cout << "\nAll tests passed!" << std::endl;

  return 0;
}
