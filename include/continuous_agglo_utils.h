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

#ifndef continuous_agglo_utils_h
#define continuous_agglo_utils_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/tria_description.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/numerics/rtree.h>

#include <agglomerator.h>


namespace dealii::ContinuousAggloUtils
{
  // TODO: For now these are all sequential implementations, need to check what
  // is different for the parallel implementation
  namespace PointsAgglo
  {
    template <int dim>
    void
    create_triangulation_from_bounding_boxes(
      Triangulation<dim>                  &dummy_tria,
      const std::vector<BoundingBox<dim>> &boxes)
    {
      const unsigned int n_boxes          = boxes.size();
      const unsigned int vertices_per_box = (dim == 2) ? 4 : 8;

      // Pre-allocate space
      std::vector<Point<dim>>    vertices;
      std::vector<CellData<dim>> cells;
      vertices.reserve(n_boxes * vertices_per_box);
      cells.reserve(n_boxes);

      for (const auto &box : boxes)
        {
          const Point<dim> &p_min = box.get_boundary_points().first;
          const Point<dim> &p_max = box.get_boundary_points().second;

          // Each box gets its own set of vertices (allows overlaps!)
          const unsigned int vertex_offset = vertices.size();

          if constexpr (dim == 2)
            {
              // 4 corners of the bounding box
              vertices.push_back(Point<dim>(p_min[0], p_min[1])); // bottom-left
              vertices.push_back(
                Point<dim>(p_max[0], p_min[1])); // bottom-right
              vertices.push_back(Point<dim>(p_min[0], p_max[1])); // top-left
              vertices.push_back(Point<dim>(p_max[0], p_max[1])); // top-right

              // Create a quadrilateral cell from these 4 vertices
              CellData<dim> cell;
              cell.vertices[0] = vertex_offset + 0; // bottom-left
              cell.vertices[1] = vertex_offset + 1; // bottom-right
              cell.vertices[2] = vertex_offset + 2; // top-left
              cell.vertices[3] = vertex_offset + 3; // top-right
              cells.push_back(cell);
            }
          else if constexpr (dim == 3)
            {
              // 8 corners of the 3D bounding box
              vertices.push_back(Point<dim>(p_min[0], p_min[1], p_min[2]));
              vertices.push_back(Point<dim>(p_max[0], p_min[1], p_min[2]));
              vertices.push_back(Point<dim>(p_min[0], p_max[1], p_min[2]));
              vertices.push_back(Point<dim>(p_max[0], p_max[1], p_min[2]));
              vertices.push_back(Point<dim>(p_min[0], p_min[1], p_max[2]));
              vertices.push_back(Point<dim>(p_max[0], p_min[1], p_max[2]));
              vertices.push_back(Point<dim>(p_min[0], p_max[1], p_max[2]));
              vertices.push_back(Point<dim>(p_max[0], p_max[1], p_max[2]));

              // Create a hexahedral cell from these 8 vertices
              CellData<dim> cell;
              for (unsigned int v = 0; v < 8; ++v)
                cell.vertices[v] = vertex_offset + v;
              cells.push_back(cell);
            }
        }

      // Create the triangulation
      dummy_tria.create_triangulation(vertices, cells, SubCellData());
    }



    template <int dim>
    void
    create_distributed_tria_from_local_boxes(
      parallel::fullydistributed::Triangulation<dim, dim> &distributed_tria,
      const std::vector<BoundingBox<dim>>                 &local_boxes,
      MPI_Comm                                             communicator)
    {
      unsigned int my_rank = Utilities::MPI::this_mpi_process(communicator);

      Triangulation<dim> tria_local;
      create_triangulation_from_bounding_boxes(tria_local, local_boxes);

      // Mark all cells as owned by my rank
      for (const auto &cell : tria_local.active_cell_iterators())
        cell->set_subdomain_id(my_rank);

      const TriangulationDescription::Description<dim, dim> description =
        TriangulationDescription::Utilities::
          create_description_from_triangulation(tria_local, communicator);

      distributed_tria.create_triangulation(description);
    }



    template <int dim, typename MatrixType, typename SparsityPatternType>
    void
    fill_injection_matrix(
      const DoFHandler<dim> &coarse_dof_handler,
      const DoFHandler<dim> &fine_dof_handler,
      SparsityPatternType   &sparsity_pattern,
      MatrixType            &transfer_matrix,
      const std::map<
        std::pair<types::global_cell_index, types::global_cell_index>,
        std::vector<types::global_cell_index>> &parent_to_child_info,
      const std::vector<BoundingBox<dim>>      &coarse_level_boxes,
      const std::vector<BoundingBox<dim>>      &fine_level_boxes,
      const unsigned int                        coarse_level)
    {}


    // Note for me: we suppose that the support points are ordered in the same
    // way as the DoFs numbering of the finest level, so that we can use the
    // support points to construct the agglomeration hierarchy. This is not a
    // strong assumption, since we can always reorder the support points to
    // match the DoF ordering. Deal.II automatically does this.
    template <int dim,
              typename MatrixType,
              typename VectorType,
              unsigned int rtree_m,
              unsigned int rtree_M>
    void
    agglomerate_and_compute_injection_matrices(
      VectorType              &support_points_vector,
      bool                     skip_leaves,
      std::vector<MatrixType> &injection_matrices,
      unsigned int             mg_levels)
    {
      namespace bgi = boost::geometry::index;

      if constexpr (std::is_same_v<MatrixType, SparseMatrix<double>>)
        {
          auto tree = pack_rtree_of_indices<bgi::rstar<rtree_m, rtree_M>>(
            support_points_vector);

          unsigned int tree_lvls = n_levels(tree);
          if (skip_leaves)
            tree_lvls = std::max<unsigned int>(tree_lvls - 1, 0);

          Assert(
            tree_lvls > 1,
            ExcMessage(
              "The tree should have at least two levels to perform agglomeration."));

          Assert(mg_levels <= tree_lvls + 1,
                 ExcMessage("You are trying to use " +
                            std::to_string(mg_levels) +
                            " levels, but the hierarchy can only have " +
                            std::to_string(tree_lvls + 1) + " levels."));

          injection_matrices.resize(mg_levels - 1);

          std::vector<std::vector<BoundingBox<dim>>> boxes_per_level(mg_levels -
                                                                     1);
          for (unsigned int i = tree_lvls - mg_levels + 1; i < tree_lvls; ++i)
            {
              unsigned int                                 j = 0;
              CellsAgglomerator<dim, decltype(tree), true> agglomerator{tree,
                                                                        i + 1};
              const std::vector<std::vector<types::global_dof_index>>
                agglomerates = agglomerator.extract_agglomerates();
              boxes_per_level[j].reserve(agglomerates.size());

              for (const std::vector<types::global_dof_index> &agglo :
                   agglomerates)
                {
                  std::vector<Point<dim>> points_in_current_agglomerate;
                  points_in_current_agglomerate.reserve(agglo.size());

                  for (const auto &index : agglo)
                    points_in_current_agglomerate.push_back(
                      support_points_vector[index]);

                  BoundingBox<dim> bbox{points_in_current_agglomerate};
                  boxes_per_level[j].emplace_back(
                    points_in_current_agglomerate);
                }
              j++;
            }
        }
      else
        {
          Assert(false, ExcNotImplemented());
        }
    }

  } // namespace PointsAgglo
  namespace CellsAgglo
  {}
} // namespace dealii::ContinuousAggloUtils

#endif