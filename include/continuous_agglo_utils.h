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

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_description.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/numerics/rtree.h>

#include <agglomerator.h>

#include <map>
#include <memory>
#include <vector>


namespace dealii::ContinuousAggloUtils
{
  template <int dim>
  void
  create_triangulation_from_bounding_boxes(
    Triangulation<dim>                  &dummy_tria,
    const std::vector<BoundingBox<dim>> &boxes)
  {
    const unsigned int n_boxes          = boxes.size();
    const unsigned int vertices_per_box = (dim == 2) ? 4 : 8;

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
            vertices.push_back(Point<dim>(p_max[0], p_min[1])); // bottom-right
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
  {
    if constexpr (std::is_same_v<MatrixType, SparseMatrix<double>> &&
                  std::is_same_v<SparsityPatternType, SparsityPattern>)
      {
        const FiniteElement<dim> &fe_dgq = coarse_dof_handler.get_fe();
        const Triangulation<dim> &fine_tria =
          fine_dof_handler.get_triangulation();
        AffineConstraints<double> constraints;

        const std::vector<Point<dim>> &unit_support_points =
          fe_dgq.get_unit_support_points();

        DynamicSparsityPattern dsp;
        dsp.reinit(fine_dof_handler.n_dofs(), coarse_dof_handler.n_dofs());
        AffineConstraints<double>            dummy_constraints;
        std::vector<types::global_dof_index> coarse_dof_indices(
          fe_dgq.n_dofs_per_cell());
        std::vector<types::global_dof_index> fine_dof_indices(
          fe_dgq.n_dofs_per_cell());


        // Loop over coarse tria and print DoFs
        for (const auto &cell : coarse_dof_handler.active_cell_iterators())
          {
            cell->get_dof_indices(coarse_dof_indices);

            std::vector<types::global_dof_index> indices_of_children =
              parent_to_child_info.at(
                {cell->active_cell_index(), coarse_level + 1});

            for (const auto &idx : indices_of_children)
              {
                DoFAccessor<dim, dim, dim, false> dof_accessor_child(
                  &fine_tria, 0, idx, &fine_dof_handler);
                dof_accessor_child.get_dof_indices(fine_dof_indices);

                for (const types::global_dof_index row : fine_dof_indices)
                  dsp.add_entries(row,
                                  coarse_dof_indices.begin(),
                                  coarse_dof_indices.end());
              }
          }

        // Filled sparsity pattern
        sparsity_pattern.copy_from(dsp);

        // Now onto filling the matrix...
        transfer_matrix.reinit(sparsity_pattern);
        const unsigned int dofs_per_cell = fe_dgq.n_dofs_per_cell();
        FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);

        for (const auto &cell : coarse_dof_handler.active_cell_iterators())
          {
            cell->get_dof_indices(coarse_dof_indices);

            const BoundingBox<dim> &coarse_box =
              coarse_level_boxes[cell->active_cell_index()];

            std::vector<types::global_dof_index> indices_of_children =
              parent_to_child_info.at(
                {cell->active_cell_index(), coarse_level + 1});

            for (const auto &idx : indices_of_children)
              {
                DoFAccessor<dim, dim, dim, false> dof_accessor_child(
                  &fine_tria, 0, idx, &fine_dof_handler);
                dof_accessor_child.get_dof_indices(fine_dof_indices);
                const BoundingBox<dim> &fine_bbox = fine_level_boxes[idx];

                local_matrix = 0.;

                // Now we plot the fine support points
                std::vector<Point<dim>> real_qpoints;
                real_qpoints.reserve(unit_support_points.size());
                for (const Point<dim> &p : unit_support_points)
                  real_qpoints.push_back(fine_bbox.unit_to_real(p));

                for (unsigned int i = 0; i < coarse_dof_indices.size(); ++i)
                  {
                    const auto &p = coarse_box.real_to_unit(real_qpoints[i]);
                    for (unsigned int j = 0; j < fine_dof_indices.size(); ++j)
                      {
                        local_matrix(i, j) = fe_dgq.shape_value(j, p);
                      }
                  }

                constraints.distribute_local_to_global(local_matrix,
                                                       fine_dof_indices,
                                                       coarse_dof_indices,
                                                       transfer_matrix);
              }
          }
      }
    else if constexpr (std::is_same_v<MatrixType,
                                      TrilinosWrappers::SparseMatrix> &&
                       std::is_same_v<SparsityPatternType,
                                      TrilinosWrappers::SparsityPattern>)
      {
        const FiniteElement<dim> &fe_dgq = coarse_dof_handler.get_fe();
        const Triangulation<dim> &fine_tria =
          fine_dof_handler.get_triangulation();

        // Check that the triangulation is distributed
        Assert(
          (dynamic_cast<
             const parallel::fullydistributed::Triangulation<dim, dim> *>(
             &fine_tria) != nullptr),
          ExcMessage(
            "The triangulation should be a parallel::fullydistributed::Triangulation."));

        const MPI_Comm comm = fine_tria.get_mpi_communicator();

        // just for local2global
        AffineConstraints<double> constraints;
        constraints.close();

        const IndexSet &owned_fine   = fine_dof_handler.locally_owned_dofs();
        const IndexSet &owned_coarse = coarse_dof_handler.locally_owned_dofs();

        sparsity_pattern.reinit(owned_fine, owned_coarse, comm);

        std::vector<types::global_dof_index> coarse_dof_indices(
          fe_dgq.n_dofs_per_cell());
        std::vector<types::global_dof_index> fine_dof_indices(
          fe_dgq.n_dofs_per_cell());

        for (const auto &cell : coarse_dof_handler.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(coarse_dof_indices);

              // std::cout << "Debug: Coarse cell " <<
              // cell->active_cell_index()
              //           << " with DoFs: ";
              // for (const auto &idx : coarse_dof_indices)
              //   std::cout << idx << " ";
              // std::cout << std::endl;

              std::vector<types::global_dof_index> indices_of_children =
                parent_to_child_info.at(
                  {cell->active_cell_index(), coarse_level + 1});

              for (const auto child_idx : indices_of_children)
                {
                  DoFAccessor<dim, dim, dim, false> child_accessor(
                    &fine_tria, 0, child_idx, &fine_dof_handler);
                  child_accessor.get_dof_indices(fine_dof_indices);

                  for (const auto row : fine_dof_indices)
                    sparsity_pattern.add_entries(row,
                                                 coarse_dof_indices.begin(),
                                                 coarse_dof_indices.end());
                }
            }

        sparsity_pattern.compress();
        transfer_matrix.reinit(sparsity_pattern);

        FullMatrix<double> local_matrix(fe_dgq.n_dofs_per_cell(),
                                        fe_dgq.n_dofs_per_cell());
        const auto &unit_support_points = fe_dgq.get_unit_support_points();

        for (const auto &cell : coarse_dof_handler.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(coarse_dof_indices);
              const BoundingBox<dim> &coarse_box =
                coarse_level_boxes[cell->active_cell_index()];

              std::vector<types::global_dof_index> indices_of_children =
                parent_to_child_info.at(
                  {cell->active_cell_index(), coarse_level + 1});

              for (const auto child_idx : indices_of_children)
                {
                  DoFAccessor<dim, dim, dim, false> child_accessor(
                    &fine_tria, 0, child_idx, &fine_dof_handler);
                  child_accessor.get_dof_indices(fine_dof_indices);

                  const BoundingBox<dim> &fine_box =
                    fine_level_boxes[child_idx];
                  local_matrix = 0.0;

                  std::vector<Point<dim>> real_qpoints;
                  real_qpoints.reserve(unit_support_points.size());
                  for (const auto &p : unit_support_points)
                    real_qpoints.push_back(fine_box.unit_to_real(p));

                  for (unsigned int i = 0; i < coarse_dof_indices.size(); ++i)
                    {
                      const Point<dim> p =
                        coarse_box.real_to_unit(real_qpoints[i]);
                      for (unsigned int j = 0; j < fine_dof_indices.size(); ++j)
                        local_matrix(i, j) = fe_dgq.shape_value(j, p);
                    }

                  constraints.distribute_local_to_global(local_matrix,
                                                         fine_dof_indices,
                                                         coarse_dof_indices,
                                                         transfer_matrix);
                }
            }

        transfer_matrix.compress(VectorOperation::add);
      }
    else
      {
        Assert(false, ExcNotImplemented());
      }
  }



  namespace PointsAgglo
  {
    // Note for me:
    // This function is not supporting multiple DoFs insisting on the same
    // support point as it uses the size of the vector of support points to
    // determine the number of DoFs on the finest level.
    template <int dim, unsigned int rtree_m, unsigned int rtree_M>
    void
    agglomerate_and_compute_injection_matrices(
      const std::vector<Point<dim>>     &support_points_vector,
      const bool                         skip_leaves,
      std::vector<SparseMatrix<double>> &injection_matrices,
      std::vector<SparsityPattern>      &injection_sparsity_patterns,
      const unsigned int                 mg_levels,
      const std::vector<unsigned int>   &coarse_space_degrees,
      std::vector<std::unique_ptr<Triangulation<dim>>> &triangulations,
      std::vector<std::unique_ptr<DoFHandler<dim>>>    &support_dof_handlers)
    {
      namespace bgi = boost::geometry::index;

      Assert(mg_levels > 1,
             ExcMessage("At least two levels are needed for agglomeration."));
      Assert(coarse_space_degrees.size() == mg_levels - 1,
             ExcMessage(
               "The size of coarse_space_degrees should be mg_levels - 1."));

      auto tree = pack_rtree_of_indices<bgi::rstar<rtree_M, rtree_m>>(
        support_points_vector);

      unsigned int tree_lvls = n_levels(tree);
      if (skip_leaves)
        tree_lvls = std::max<unsigned int>(tree_lvls - 1, 0);

      Assert(
        tree_lvls > 1,
        ExcMessage(
          "The tree should have at least two levels to perform agglomeration."));

      Assert(mg_levels <= tree_lvls + 1,
             ExcMessage("You are trying to use " + std::to_string(mg_levels) +
                        " levels, but the hierarchy can only have " +
                        std::to_string(tree_lvls + 1) + " levels."));

      std::vector<std::vector<BoundingBox<dim>>> boxes_per_level(mg_levels - 1);
      std::vector<
        std::map<std::pair<types::global_cell_index, types::global_cell_index>,
                 std::vector<types::global_cell_index>>>
        hierarchies_per_level(mg_levels - 2);

      unsigned int j = 0;
      for (unsigned int i = tree_lvls - mg_levels + 1; i < tree_lvls; ++i)
        {
          CellsAgglomerator<dim, decltype(tree), true> agglomerator{tree,
                                                                    i + 1};
          const std::vector<std::vector<types::global_dof_index>> agglomerates =
            agglomerator.extract_agglomerates();
          boxes_per_level[j].reserve(agglomerates.size());

          // Store hierarchy for reuse later (not needed for last level)
          if (j < mg_levels - 2)
            hierarchies_per_level[j] = agglomerator.get_hierarchy();

          for (const std::vector<types::global_dof_index> &agglo : agglomerates)
            {
              std::vector<Point<dim>> points_in_current_agglomerate;
              points_in_current_agglomerate.reserve(agglo.size());

              for (const auto &index : agglo)
                points_in_current_agglomerate.push_back(
                  support_points_vector[index]);

              BoundingBox<dim> bbox{points_in_current_agglomerate};
              boxes_per_level[j].emplace_back(points_in_current_agglomerate);
            }
          j++;
        }

      triangulations.clear();
      support_dof_handlers.clear();
      triangulations.resize(mg_levels - 1);
      support_dof_handlers.resize(mg_levels - 1);

      for (unsigned int i = 0; i < mg_levels - 1; ++i)
        {
          triangulations[i] = std::make_unique<Triangulation<dim>>();
          dealii::ContinuousAggloUtils::
            create_triangulation_from_bounding_boxes(*triangulations[i],
                                                     boxes_per_level[i]);
          support_dof_handlers[i] =
            std::make_unique<DoFHandler<dim>>(*triangulations[i]);

          FE_DGQ<dim> fe_dgq(coarse_space_degrees[i]);
          support_dof_handlers[i]->distribute_dofs(fe_dgq);
        }
      injection_matrices.clear();
      injection_sparsity_patterns.clear();
      injection_matrices.resize(mg_levels - 1);
      injection_sparsity_patterns.resize(mg_levels - 1);

      for (unsigned int j = 0; j < mg_levels - 2; ++j)
        dealii::ContinuousAggloUtils::fill_injection_matrix(
          *support_dof_handlers[j],
          *support_dof_handlers[j + 1],
          injection_sparsity_patterns[j],
          injection_matrices[j],
          hierarchies_per_level[j],
          boxes_per_level[j],
          boxes_per_level[j + 1],
          tree_lvls - mg_levels + 1 + j);

      // We now have to fill the last injection matrix
      {
        CellsAgglomerator<dim, decltype(tree), true> agglomerator{tree,
                                                                  tree_lvls};

        const std::vector<std::vector<types::global_dof_index>> agglomerates =
          agglomerator.extract_agglomerates();

        FE_DGQ<dim> fe_dgq(coarse_space_degrees[mg_levels - 2]);

        std::vector<types::global_dof_index> dof_indices_agglo_tria(
          fe_dgq.n_dofs_per_cell());

        DynamicSparsityPattern dsp_agglo_to_original_tria;
        dsp_agglo_to_original_tria.reinit(
          support_points_vector
            .size(), // should be equal to the number of DoFs if only  1
                     // DoFs insist per support point
          support_dof_handlers[mg_levels - 2]->n_dofs());

        unsigned int agglo_index = 0;
        for (const auto &cell :
             support_dof_handlers[mg_levels - 2]->active_cell_iterators())
          {
            cell->get_dof_indices(dof_indices_agglo_tria);

            for (const types::global_dof_index dof_idx :
                 agglomerates[agglo_index])
              dsp_agglo_to_original_tria.add_entries(
                dof_idx,
                dof_indices_agglo_tria.begin(),
                dof_indices_agglo_tria.end());

            agglo_index++;
          }

        injection_sparsity_patterns[mg_levels - 2].copy_from(
          dsp_agglo_to_original_tria);
        injection_matrices[mg_levels - 2].reinit(
          injection_sparsity_patterns[mg_levels - 2]);

        AffineConstraints<double> dummy_constraints; // for loc2glob

        agglo_index = 0;
        for (const auto &cell :
             support_dof_handlers[mg_levels - 2]->active_cell_iterators())
          {
            cell->get_dof_indices(dof_indices_agglo_tria);

            const BoundingBox<dim> &coarse_box =
              boxes_per_level[mg_levels - 2][cell->active_cell_index()];

            const unsigned int n_fine_support_points =
              agglomerates[agglo_index].size();

            const std::vector<types::global_dof_index> fine_indices =
              agglomerates[agglo_index];

            FullMatrix<double> local_matrix(n_fine_support_points,
                                            fe_dgq.n_dofs_per_cell());
            local_matrix = 0.0;

            for (unsigned int i = 0; i < n_fine_support_points; ++i)
              {
                const Point<dim> p = coarse_box.real_to_unit(
                  support_points_vector[fine_indices[i]]);

                for (unsigned int j = 0; j < dof_indices_agglo_tria.size(); ++j)
                  local_matrix(i, j) = fe_dgq.shape_value(j, p);
              }

            dummy_constraints.distribute_local_to_global(
              local_matrix,
              fine_indices,
              dof_indices_agglo_tria,
              injection_matrices[mg_levels - 2]);

            ++agglo_index;
          }
      }
    }



    // For the parallel version there are way too many issues with sending just
    // the map. Using the DoFHandler of the fine original tria is easier. This
    // also means that we need a mapping I am using a generic mapping since i
    // don't know if there are weird geometries.
    template <int dim, unsigned int rtree_m, unsigned int rtree_M>
    void
    parallel_agglomerate_and_compute_injection_matrices(
      const DoFHandler<dim>                       &fine_dh,
      const Mapping<dim, dim>                     &mapping,
      const bool                                   skip_leaves,
      std::vector<TrilinosWrappers::SparseMatrix> &injection_matrices,
      std::vector<TrilinosWrappers::SparsityPattern>
                                      &injection_sparsity_patterns,
      const unsigned int               mg_levels,
      const std::vector<unsigned int> &coarse_space_degrees,
      std::vector<
        std::unique_ptr<parallel::fullydistributed::Triangulation<dim, dim>>>
                                                    &triangulations,
      std::vector<std::unique_ptr<DoFHandler<dim>>> &support_dof_handlers)
    {
      namespace bgi = boost::geometry::index;

      MPI_Comm           comm    = fine_dh.get_mpi_communicator();
      const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);

      Assert(mg_levels > 1,
             ExcMessage("At least two levels are needed for agglomeration."));
      Assert(coarse_space_degrees.size() == mg_levels - 1,
             ExcMessage(
               "The size of coarse_space_degrees should be mg_levels - 1."));

      std::map<types::global_dof_index, Point<dim>> support_points_map =
        DoFTools::map_dofs_to_support_points(mapping, fine_dh);

      const unsigned int locally_owned_dofs_number =
        fine_dh.locally_owned_dofs().n_elements();

      const IndexSet &locally_owned_dofs_set = fine_dh.locally_owned_dofs();
      std::vector<types::global_dof_index> local_dof_indices;
      std::vector<Point<dim>>              local_support_points_vector;
      local_support_points_vector.reserve(locally_owned_dofs_number);
      local_dof_indices.reserve(locally_owned_dofs_number);
      for (const auto &entry : support_points_map)
        if (locally_owned_dofs_set.is_element(entry.first))
          {
            local_dof_indices.push_back(entry.first);
            local_support_points_vector.push_back(entry.second);
          }

      auto local_tree = pack_rtree_of_indices<bgi::rstar<rtree_M, rtree_m>>(
        local_support_points_vector);
      // The indices used are tied to the indices of
      // local_support_poinst_vector. To access the original DoFs, one needs to
      // use local_dof_indices

      unsigned int local_tree_lvls = n_levels(local_tree);
      if (skip_leaves)
        local_tree_lvls = std::max<unsigned int>(local_tree_lvls - 1, 0);

      Assert(local_tree_lvls > 1,
             ExcMessage(
               "The tree on rank " + std::to_string(my_rank) +
               " should have at least two levels to perform agglomeration."));

      Assert(mg_levels <= local_tree_lvls + 1,
             ExcMessage("You are trying to use " + std::to_string(mg_levels) +
                        " levels, but the hierarchy on rank " +
                        std::to_string(my_rank) + " can only have " +
                        std::to_string(local_tree_lvls + 1) + " levels."));

      // To deal with trees of different heights: we simply take the levels from
      // the finest level and the bound is given by the shortest tree
      // We could probably allow up to the highest local tree by padding with
      // identity but for now this is fine

      std::vector<std::vector<BoundingBox<dim>>> local_boxes_per_level(
        mg_levels - 1);
      std::vector<
        std::map<std::pair<types::global_cell_index, types::global_cell_index>,
                 std::vector<types::global_cell_index>>>
        local_hierarchies_per_level(mg_levels - 2);

      unsigned int j = 0;
      for (unsigned int i = local_tree_lvls - mg_levels + 1;
           i < local_tree_lvls;
           ++i)
        {
          CellsAgglomerator<dim, decltype(local_tree), true> agglomerator{
            local_tree, i + 1};
          const std::vector<std::vector<types::global_dof_index>> agglomerates =
            agglomerator.extract_agglomerates();
          local_boxes_per_level[j].reserve(agglomerates.size());

          // Store hierarchy for reuse later (not needed for last level)
          if (j < mg_levels - 2)
            local_hierarchies_per_level[j] = agglomerator.get_hierarchy();

          for (const std::vector<types::global_dof_index> &agglo : agglomerates)
            {
              std::vector<Point<dim>> points_in_current_agglomerate;
              points_in_current_agglomerate.reserve(agglo.size());

              for (const auto &index : agglo)
                points_in_current_agglomerate.push_back(
                  local_support_points_vector[index]);

              BoundingBox<dim> bbox{points_in_current_agglomerate};
              local_boxes_per_level[j].emplace_back(
                points_in_current_agglomerate);
            }
          j++;
        }

      triangulations.clear();
      support_dof_handlers.clear();
      triangulations.resize(mg_levels - 1);
      support_dof_handlers.resize(mg_levels - 1);

      for (unsigned int i = 0; i < mg_levels - 1; ++i)
        {
          triangulations[i] = std::make_unique<
            parallel::fullydistributed::Triangulation<dim, dim>>(comm);
          dealii::ContinuousAggloUtils::
            create_distributed_tria_from_local_boxes(*triangulations[i],
                                                     local_boxes_per_level[i],
                                                     comm);
          support_dof_handlers[i] =
            std::make_unique<DoFHandler<dim>>(*triangulations[i]);

          FE_DGQ<dim> fe_dgq(coarse_space_degrees[i]);
          support_dof_handlers[i]->distribute_dofs(fe_dgq);
        }

      injection_matrices.clear();
      injection_sparsity_patterns.clear();
      injection_matrices.resize(mg_levels - 1);
      injection_sparsity_patterns.resize(mg_levels - 1);

      for (unsigned int j = 0; j < mg_levels - 2; ++j)
        dealii::ContinuousAggloUtils::fill_injection_matrix(
          *support_dof_handlers[j],
          *support_dof_handlers[j + 1],
          injection_sparsity_patterns[j],
          injection_matrices[j],
          local_hierarchies_per_level[j],
          local_boxes_per_level[j],
          local_boxes_per_level[j + 1],
          local_tree_lvls - mg_levels + 1 + j);

      // We now have to fill the last injection matrix
      {
        CellsAgglomerator<dim, decltype(local_tree), true> agglomerator{
          local_tree, local_tree_lvls};

        const std::vector<std::vector<types::global_dof_index>> agglomerates =
          agglomerator.extract_agglomerates();

        FE_DGQ<dim> fe_dgq(coarse_space_degrees[mg_levels - 2]);

        std::vector<types::global_dof_index> dof_indices_agglo_tria(
          fe_dgq.n_dofs_per_cell());

        const IndexSet &locally_owned_dofs_coarse =
          support_dof_handlers[mg_levels - 2]->locally_owned_dofs();
        const IndexSet &locally_owned_dofs_fine = fine_dh.locally_owned_dofs();

        injection_sparsity_patterns[mg_levels - 2].reinit(
          locally_owned_dofs_fine, locally_owned_dofs_coarse, comm);

        unsigned int agglo_index = 0;
        for (const auto &cell :
             support_dof_handlers[mg_levels - 2]->active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(dof_indices_agglo_tria);

              // agglomerates return indices in the local tree, we need to map
              // them back to the original DoF indices using local_dof_indices
              for (const types::global_dof_index dof_idx :
                   agglomerates[agglo_index])
                injection_sparsity_patterns[mg_levels - 2].add_entries(
                  local_dof_indices[dof_idx],
                  dof_indices_agglo_tria.begin(),
                  dof_indices_agglo_tria.end());

              agglo_index++;
            }

        injection_sparsity_patterns[mg_levels - 2].compress();
        injection_matrices[mg_levels - 2].reinit(
          injection_sparsity_patterns[mg_levels - 2]);

        AffineConstraints<double> dummy_constraints; // for loc2glob
        agglo_index = 0;
        for (const auto &cell :
             support_dof_handlers[mg_levels - 2]->active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(dof_indices_agglo_tria);

              const BoundingBox<dim> &coarse_box =
                local_boxes_per_level[mg_levels - 2][agglo_index];

              const unsigned int n_fine_support_points =
                agglomerates[agglo_index].size();

              const std::vector<types::global_dof_index> &fine_indices =
                agglomerates[agglo_index];

              FullMatrix<double> local_matrix(n_fine_support_points,
                                              fe_dgq.n_dofs_per_cell());
              local_matrix = 0.0;

              for (unsigned int i = 0; i < n_fine_support_points; ++i)
                {
                  const Point<dim> p = coarse_box.real_to_unit(
                    local_support_points_vector[fine_indices[i]]);

                  for (unsigned int j = 0; j < dof_indices_agglo_tria.size();
                       ++j)
                    local_matrix(i, j) = fe_dgq.shape_value(j, p);
                }

              std::vector<types::global_dof_index> fine_indices_global(
                n_fine_support_points);
              for (unsigned int i = 0; i < n_fine_support_points; ++i)
                fine_indices_global[i] = local_dof_indices[fine_indices[i]];

              dummy_constraints.distribute_local_to_global(
                local_matrix,
                fine_indices_global,
                dof_indices_agglo_tria,
                injection_matrices[mg_levels - 2]);

              ++agglo_index;
            }

        injection_matrices[mg_levels - 2].compress(VectorOperation::add);
      }
    }
  } // namespace PointsAgglo
  namespace CellsAgglo
  {} // namespace CellsAgglo
} // namespace dealii::ContinuousAggloUtils

#endif