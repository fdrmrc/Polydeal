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


#ifndef poly_utils_h
#define poly_utils_h


#include <deal.II/base/config.h>

#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/std_cxx20/iota_view.h>

#include <deal.II/boost_adaptors/bounding_box.h>
#include <deal.II/boost_adaptors/point.h>
#include <deal.II/boost_adaptors/segment.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <boost/geometry/algorithms/distance.hpp>
#include <boost/geometry/index/detail/rtree/utilities/print.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/geometry/strategies/strategies.hpp>

#include <deal.II/cgal/point_conversion.h>

#ifdef DEAL_II_WITH_TRILINOS
#  include <EpetraExt_RowMatrixOut.h>
#endif

#ifdef DEAL_II_WITH_CGAL

#  include <CGAL/Constrained_Delaunay_triangulation_2.h>
#  include <CGAL/Constrained_triangulation_plus_2.h>
#  include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#  include <CGAL/Exact_predicates_exact_constructions_kernel_with_sqrt.h>
#  include <CGAL/Polygon_2.h>
#  include <CGAL/Polygon_with_holes_2.h>
#  include <CGAL/Segment_Delaunay_graph_2.h>
#  include <CGAL/Segment_Delaunay_graph_traits_2.h>
#  include <CGAL/intersections.h>
#  include <CGAL/squared_distance_2.h>
#  include <CGAL/squared_distance_3.h>


#endif

#include <memory>


namespace dealii::PolyUtils::internal
{
  /**
   * Helper function to compute the position of index @p index in vector @p v.
   */
  inline types::global_cell_index
  get_index(const std::vector<types::global_cell_index> &v,
            const types::global_cell_index               index)
  {
    return std::distance(v.begin(), std::find(v.begin(), v.end(), index));
  }



  /**
   * Compute the connectivity graph for locally owned regions of a distributed
   * triangulation.
   */
  template <int dim, int spacedim>
  void
  get_face_connectivity_of_cells(
    const parallel::fullydistributed::Triangulation<dim, spacedim>
                                               &triangulation,
    DynamicSparsityPattern                     &cell_connectivity,
    const std::vector<types::global_cell_index> locally_owned_cells)
  {
    cell_connectivity.reinit(triangulation.n_locally_owned_active_cells(),
                             triangulation.n_locally_owned_active_cells());


    // loop over all cells and their neighbors to build the sparsity
    // pattern. note that it's a bit hard to enter all the connections when
    // a neighbor has children since we would need to find out which of its
    // children is adjacent to the current cell. this problem can be omitted
    // if we only do something if the neighbor has no children -- in that
    // case it is either on the same or a coarser level than we are. in
    // return, we have to add entries in both directions for both cells
    for (const auto &cell : triangulation.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            const unsigned int index = cell->active_cell_index();
            cell_connectivity.add(get_index(locally_owned_cells, index),
                                  get_index(locally_owned_cells, index));
            for (auto f : cell->face_indices())
              if ((cell->at_boundary(f) == false) &&
                  (cell->neighbor(f)->has_children() == false) &&
                  cell->neighbor(f)->is_locally_owned())
                {
                  const unsigned int other_index =
                    cell->neighbor(f)->active_cell_index();

                  cell_connectivity.add(get_index(locally_owned_cells, index),
                                        get_index(locally_owned_cells,
                                                  other_index));
                  cell_connectivity.add(get_index(locally_owned_cells,
                                                  other_index),
                                        get_index(locally_owned_cells, index));
                }
          }
      }
  }
} // namespace dealii::PolyUtils::internal


namespace dealii::PolyUtils
{
  template <typename Value,
            typename Options,
            typename Translator,
            typename Box,
            typename Allocators>
  struct Rtree_visitor : public boost::geometry::index::detail::rtree::visitor<
                           Value,
                           typename Options::parameters_type,
                           Box,
                           Allocators,
                           typename Options::node_tag,
                           true>::type
  {
    inline Rtree_visitor(
      const Translator &translator,
      unsigned int      target_level,
      std::vector<std::vector<typename Triangulation<
        boost::geometry::dimension<Box>::value>::active_cell_iterator>> &boxes,
      std::vector<std::vector<unsigned int>>                            &csr);


    /**
     * An alias that identifies an InternalNode of the tree.
     */
    using InternalNode =
      typename boost::geometry::index::detail::rtree::internal_node<
        Value,
        typename Options::parameters_type,
        Box,
        Allocators,
        typename Options::node_tag>::type;

    /**
     * An alias that identifies a Leaf of the tree.
     */
    using Leaf = typename boost::geometry::index::detail::rtree::leaf<
      Value,
      typename Options::parameters_type,
      Box,
      Allocators,
      typename Options::node_tag>::type;

    /**
     * Implements the visitor interface for InternalNode objects. If the node
     * belongs to the level next to @p target_level, then fill the bounding box vector for that node.
     */
    inline void
    operator()(const InternalNode &node);

    /**
     * Implements the visitor interface for Leaf objects.
     */
    inline void
    operator()(const Leaf &);

    /**
     * Translator interface, required by the boost implementation of the rtree.
     */
    const Translator &translator;

    /**
     * Store the level we are currently visiting.
     */
    size_t level;

    /**
     * Index used to keep track of the number of different visited nodes during
     * recursion/
     */
    size_t node_counter;

    size_t next_level_leafs_processed;
    /**
     * The level where children are living.
     * Before: "we want to extract from the RTree object."
     */
    const size_t target_level;

    /**
     * A reference to the input vector of vector of BoundingBox objects. This
     * vector v has the following property: v[i] = vector with all
     * of the BoundingBox bounded by the i-th node of the Rtree.
     */
    std::vector<std::vector<typename Triangulation<
      boost::geometry::dimension<Box>::value>::active_cell_iterator>>
      &agglomerates;

    std::vector<std::vector<unsigned int>> &row_ptr;
  };



  template <typename Value,
            typename Options,
            typename Translator,
            typename Box,
            typename Allocators>
  Rtree_visitor<Value, Options, Translator, Box, Allocators>::Rtree_visitor(
    const Translator  &translator,
    const unsigned int target_level,
    std::vector<std::vector<typename Triangulation<
      boost::geometry::dimension<Box>::value>::active_cell_iterator>>
                                           &bb_in_boxes,
    std::vector<std::vector<unsigned int>> &csr)
    : translator(translator)
    , level(0)
    , node_counter(0)
    , next_level_leafs_processed(0)
    , target_level(target_level)
    , agglomerates(bb_in_boxes)
    , row_ptr(csr)
  {}



  template <typename Value,
            typename Options,
            typename Translator,
            typename Box,
            typename Allocators>
  void
  Rtree_visitor<Value, Options, Translator, Box, Allocators>::operator()(
    const Rtree_visitor::InternalNode &node)
  {
    using elements_type =
      typename boost::geometry::index::detail::rtree::elements_type<
        InternalNode>::type; //  pairs of bounding box and pointer to child node
    const elements_type &elements =
      boost::geometry::index::detail::rtree::elements(node);

    if (level < target_level)
      {
        size_t level_backup = level;
        ++level;

        for (typename elements_type::const_iterator it = elements.begin();
             it != elements.end();
             ++it)
          {
            boost::geometry::index::detail::rtree::apply_visitor(*this,
                                                                 *it->second);
          }

        level = level_backup;
      }
    else if (level == target_level)
      {
        // const unsigned int n_children = elements.size();
        const auto offset = agglomerates.size();
        agglomerates.resize(offset + 1);
        row_ptr.resize(row_ptr.size() + 1);
        next_level_leafs_processed = 0;
        row_ptr.back().push_back(
          next_level_leafs_processed); // convention: row_ptr[0]=0
        size_t level_backup = level;

        ++level;
        for (const auto &child : elements)
          {
            boost::geometry::index::detail::rtree::apply_visitor(*this,
                                                                 *child.second);
          }
        // Done with node number 'node_counter'

        ++node_counter; // visited all children of an internal node

        level = level_backup;
      }
    else if (level > target_level)
      {
        // Keep visiting until you go to the leafs.
        size_t level_backup = level;

        ++level;

        for (const auto &child : elements)
          {
            boost::geometry::index::detail::rtree::apply_visitor(*this,
                                                                 *child.second);
          }
        level = level_backup;
        row_ptr[node_counter].push_back(next_level_leafs_processed);
      }
  }



  template <typename Value,
            typename Options,
            typename Translator,
            typename Box,
            typename Allocators>
  void
  Rtree_visitor<Value, Options, Translator, Box, Allocators>::operator()(
    const Rtree_visitor::Leaf &leaf)
  {
    using elements_type =
      typename boost::geometry::index::detail::rtree::elements_type<
        Leaf>::type; //  pairs of bounding box and pointer to child node
    const elements_type &elements =
      boost::geometry::index::detail::rtree::elements(leaf);


    for (const auto &it : elements)
      {
        agglomerates[node_counter].push_back(it.second);
      }
    next_level_leafs_processed += elements.size();
  }

  template <typename Rtree>
  inline std::pair<
    std::vector<std::vector<unsigned int>>,
    std::vector<std::vector<typename Triangulation<boost::geometry::dimension<
      typename Rtree::indexable_type>::value>::active_cell_iterator>>>
  extract_children_of_level(const Rtree &tree, const unsigned int level)
  {
    using RtreeView =
      boost::geometry::index::detail::rtree::utilities::view<Rtree>;
    RtreeView rtv(tree);

    std::vector<std::vector<unsigned int>> csrs;
    std::vector<std::vector<typename Triangulation<boost::geometry::dimension<
      typename Rtree::indexable_type>::value>::active_cell_iterator>>
      agglomerates;

    if (rtv.depth() == 0)
      {
        // The below algorithm does not work for `rtv.depth()==0`, which might
        // happen if the number entries in the tree is too small.
        // In this case, simply return a single bounding box.
        agglomerates.resize(1);
        agglomerates[0].resize(1);
        csrs.resize(1);
        csrs[0].resize(1);
      }
    else
      {
        const unsigned int target_level =
          std::min<unsigned int>(level, rtv.depth());

        Rtree_visitor<typename RtreeView::value_type,
                      typename RtreeView::options_type,
                      typename RtreeView::translator_type,
                      typename RtreeView::box_type,
                      typename RtreeView::allocators_type>
          node_visitor(rtv.translator(), target_level, agglomerates, csrs);
        rtv.apply_visitor(node_visitor);
      }
    AssertDimension(agglomerates.size(), csrs.size());

    return {csrs, agglomerates};
  }


  template <int dim, typename Number = double>
  Number
  compute_h_orthogonal(
    const unsigned int face_index,
    const std::vector<typename Triangulation<dim>::active_face_iterator>
                         &polygon_boundary,
    const Tensor<1, dim> &deal_normal)
  {
#ifdef DEAL_II_WITH_CGAL

    using Kernel = CGAL::Exact_predicates_exact_constructions_kernel;
    std::vector<typename Kernel::FT> candidates;
    candidates.reserve(polygon_boundary.size() - 1);

    // Initialize the range of faces to be checked for intersection: they are
    // {0,..,n_faces-1}\setminus the current face index face_index.
    std::vector<unsigned int> face_indices(polygon_boundary.size());
    std::iota(face_indices.begin(), face_indices.end(), 0); // fill the range
    face_indices.erase(face_indices.cbegin() +
                       face_index); // remove current index

    if constexpr (dim == 2)
      {
        typename Kernel::Segment_2 face_segm(
          {polygon_boundary[face_index]->vertex(0)[0],
           polygon_boundary[face_index]->vertex(0)[1]},
          {polygon_boundary[face_index]->vertex(1)[0],
           polygon_boundary[face_index]->vertex(1)[1]});

        // Shoot a ray from the midpoint of the face in the orthogonal direction
        // given by deal.II normals
        const auto &midpoint = CGAL::midpoint(face_segm);
        // deal.II normal is always outward, flip the direction
        const typename Kernel::Vector_2 orthogonal_direction{-deal_normal[0],
                                                             -deal_normal[1]};
        const typename Kernel::Ray_2    ray(midpoint, orthogonal_direction);
        for (const auto f : face_indices)
          {
            typename Kernel::Segment_2 segm({polygon_boundary[f]->vertex(0)[0],
                                             polygon_boundary[f]->vertex(0)[1]},
                                            {polygon_boundary[f]->vertex(1)[0],
                                             polygon_boundary[f]->vertex(
                                               1)[1]});

            if (CGAL::do_intersect(ray, segm))
              candidates.push_back(CGAL::squared_distance(midpoint, segm));
          }
        return std::sqrt(CGAL::to_double(
          *std::min_element(candidates.cbegin(), candidates.cend())));
      }
    else if constexpr (dim == 3)
      {
        const typename Kernel::Point_3 &center{
          polygon_boundary[face_index]->center()[0],
          polygon_boundary[face_index]->center()[1],
          polygon_boundary[face_index]->center()[2]};
        // deal.II normal is always outward, flip the direction
        const typename Kernel::Vector_3 orthogonal_direction{-deal_normal[0],
                                                             -deal_normal[1],
                                                             -deal_normal[2]};
        const typename Kernel::Ray_3    ray(center, orthogonal_direction);

        for (const auto f : face_indices)
          {
            // split the face into 2 triangles and compute distances
            typename Kernel::Triangle_3 first_triangle(
              {polygon_boundary[f]->vertex(0)[0],
               polygon_boundary[f]->vertex(0)[1],
               polygon_boundary[f]->vertex(0)[2]},
              {polygon_boundary[f]->vertex(1)[0],
               polygon_boundary[f]->vertex(1)[1],
               polygon_boundary[f]->vertex(1)[2]},
              {polygon_boundary[f]->vertex(3)[0],
               polygon_boundary[f]->vertex(3)[1],
               polygon_boundary[f]->vertex(3)[2]});
            typename Kernel::Triangle_3 second_triangle(
              {polygon_boundary[f]->vertex(0)[0],
               polygon_boundary[f]->vertex(0)[1],
               polygon_boundary[f]->vertex(0)[2]},
              {polygon_boundary[f]->vertex(3)[0],
               polygon_boundary[f]->vertex(3)[1],
               polygon_boundary[f]->vertex(3)[2]},
              {polygon_boundary[f]->vertex(2)[0],
               polygon_boundary[f]->vertex(2)[1],
               polygon_boundary[f]->vertex(2)[2]});

            // compute point-triangle distance only if the orthogonal ray
            // hits the triangle
            if (CGAL::do_intersect(ray, first_triangle))
              candidates.push_back(
                CGAL::squared_distance(center, first_triangle));
            if (CGAL::do_intersect(ray, second_triangle))
              candidates.push_back(
                CGAL::squared_distance(center, second_triangle));
          }

        return std::sqrt(CGAL::to_double(
          *std::min_element(candidates.cbegin(), candidates.cend())));
      }
    else
      {
        Assert(false, ExcImpossibleInDim(dim));
        (void)face_index;
        (void)polygon_boundary;
        return {};
      }

#else

    Assert(false, ExcNeedsCGAL());
    (void)face_index;
    (void)polygon_boundary;
    return {};
#endif
  }


  /**
   * Given a vector @p src, typically the solution stemming after the
   * agglomerate problem has been solved, this function interpolates @p src
   * onto the finer grid and stores the result in vector @p dst.
   *
   * @note Supported parallel types are TrilinosWrappers::SparseMatrix and TrilinosWrappers::MPI::Vector
   */
  template <int dim, int spacedim, typename VectorType>
  void
  interpolate_to_fine_grid(
    const AgglomerationHandler<dim, spacedim> &agglomeration_handler,
    VectorType                                &dst,
    const VectorType                          &src)
  {
    Assert((dim == spacedim), ExcNotImplemented());
    Assert(
      dst.size() == 0,
      ExcMessage(
        "The destination vector must the empt upon calling this function."));

    using NumberType = typename VectorType::value_type;
    constexpr bool is_trilinos_vector =
      std::is_same_v<VectorType, TrilinosWrappers::MPI::Vector>;
    using MatrixType = std::conditional_t<is_trilinos_vector,
                                          TrilinosWrappers::SparseMatrix,
                                          SparseMatrix<NumberType>>;

    MatrixType interpolation_matrix;

    [[maybe_unused]]
    typename std::conditional_t<!is_trilinos_vector, SparsityPattern, void *>
      sp;

    // Get some info from the handler
    const DoFHandler<dim, spacedim> &agglo_dh = agglomeration_handler.agglo_dh;

    DoFHandler<dim, spacedim> *output_dh =
      const_cast<DoFHandler<dim, spacedim> *>(&agglomeration_handler.output_dh);
    const FiniteElement<dim, spacedim> &fe = agglomeration_handler.get_fe();
    const Triangulation<dim, spacedim> &tria =
      agglomeration_handler.get_triangulation();
    const auto &bboxes = agglomeration_handler.get_local_bboxes();

    // Setup an auxiliary DoFHandler for output purposes
    output_dh->reinit(tria);
    output_dh->distribute_dofs(fe);

    const IndexSet &locally_owned_dofs       = output_dh->locally_owned_dofs();
    const IndexSet &locally_owned_dofs_agglo = agglo_dh.locally_owned_dofs();

    std::conditional_t<is_trilinos_vector,
                       TrilinosWrappers::SparsityPattern,
                       DynamicSparsityPattern>
      dsp;

    if constexpr (is_trilinos_vector)
      dsp.reinit(locally_owned_dofs,
                 locally_owned_dofs_agglo,
                 tria.get_communicator());
    else
      dsp.reinit(output_dh->n_dofs(),
                 agglo_dh.n_dofs(),
                 output_dh->locally_owned_dofs());

    std::vector<types::global_dof_index> agglo_dof_indices(fe.dofs_per_cell);
    std::vector<types::global_dof_index> standard_dof_indices(fe.dofs_per_cell);
    std::vector<types::global_dof_index> output_dof_indices(fe.dofs_per_cell);

    Quadrature<dim>         quad(fe.get_unit_support_points());
    FEValues<dim, spacedim> output_fe_values(fe,
                                             quad,
                                             update_quadrature_points);

    for (const auto &cell : agglo_dh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          if (agglomeration_handler.is_master_cell(cell))
            {
              auto slaves = agglomeration_handler.get_slaves_of_idx(
                cell->active_cell_index());
              slaves.emplace_back(cell);

              cell->get_dof_indices(agglo_dof_indices);

              for (const auto &slave : slaves)
                {
                  // addd master-slave relationship
                  const auto slave_output =
                    slave->as_dof_handler_iterator(*output_dh);
                  slave_output->get_dof_indices(output_dof_indices);
                  for (const auto row : output_dof_indices)
                    dsp.add_entries(row,
                                    agglo_dof_indices.begin(),
                                    agglo_dof_indices.end());
                }
            }
        }


    const auto assemble_interpolation_matrix = [&]() {
      FullMatrix<NumberType>  local_matrix(fe.dofs_per_cell, fe.dofs_per_cell);
      std::vector<Point<dim>> reference_q_points(fe.dofs_per_cell);

      // Dummy AffineConstraints, only needed for loc2glb
      AffineConstraints<NumberType> c;
      c.close();

      for (const auto &cell : agglo_dh.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            if (agglomeration_handler.is_master_cell(cell))
              {
                auto slaves = agglomeration_handler.get_slaves_of_idx(
                  cell->active_cell_index());
                slaves.emplace_back(cell);

                cell->get_dof_indices(agglo_dof_indices);

                const types::global_cell_index polytope_index =
                  agglomeration_handler.cell_to_polytope_index(cell);

                // Get the box of this agglomerate.
                const BoundingBox<dim> &box = bboxes[polytope_index];

                for (const auto &slave : slaves)
                  {
                    // add master-slave relationship
                    const auto slave_output =
                      slave->as_dof_handler_iterator(*output_dh);

                    slave_output->get_dof_indices(output_dof_indices);
                    output_fe_values.reinit(slave_output);

                    local_matrix = 0.;

                    const auto &q_points =
                      output_fe_values.get_quadrature_points();
                    for (const auto i : output_fe_values.dof_indices())
                      {
                        const auto &p = box.real_to_unit(q_points[i]);
                        for (const auto j : output_fe_values.dof_indices())
                          {
                            local_matrix(i, j) = fe.shape_value(j, p);
                          }
                      }
                    c.distribute_local_to_global(local_matrix,
                                                 output_dof_indices,
                                                 agglo_dof_indices,
                                                 interpolation_matrix);
                  }
              }
          }
    };


    if constexpr (std::is_same_v<MatrixType, TrilinosWrappers::SparseMatrix>)
      {
        dsp.compress();
        interpolation_matrix.reinit(dsp);
        dst.reinit(locally_owned_dofs);
        assemble_interpolation_matrix();
      }
    else if constexpr (std::is_same_v<MatrixType, SparseMatrix<NumberType>>)
      {
        sp.copy_from(dsp);
        interpolation_matrix.reinit(sp);
        dst.reinit(output_dh->n_dofs());
        assemble_interpolation_matrix();
      }
    else
      {
        // PETSc, LA::d::v options not implemented.
        (void)agglomeration_handler;
        (void)dst;
        (void)src;
        AssertThrow(false, ExcNotImplemented());
      }

    // If tria is distributed
    if (dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
          &tria) != nullptr)
      interpolation_matrix.compress(VectorOperation::add);

    // Finally, perform the interpolation.
    interpolation_matrix.vmult(dst, src);
  }


  /**
   * Agglomerate cells together based on their global index. This function is
   * **not** efficient and should be used for testing purposes only.
   */
  template <int dim, int spacedim = dim>
  void
  collect_cells_for_agglomeration(
    const Triangulation<dim, spacedim>          &tria,
    const std::vector<types::global_cell_index> &cell_idxs,
    std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>
      &cells_to_be_agglomerated)
  {
    Assert(cells_to_be_agglomerated.size() == 0,
           ExcMessage(
             "The vector of cells is supposed to be filled by this function."));
    for (const auto &cell : tria.active_cell_iterators())
      if (std::find(cell_idxs.begin(),
                    cell_idxs.end(),
                    cell->active_cell_index()) != cell_idxs.end())
        {
          cells_to_be_agglomerated.push_back(cell);
        }
  }



  /**
   * Partition with METIS the locally owned regions of the given
   * triangulation.
   *
   * @note The given triangulation must be a parallel::fullydistributed::Triangulation. This is
   * required as the partitions generated by p4est, the partitioner for
   * parallell::distributed::Triangulation, can generate discontinuous
   * partitions which are not supported by the METIS partitioner.
   *
   */
  template <int dim, int spacedim>
  void
  partition_locally_owned_regions(const unsigned int            n_partitions,
                                  Triangulation<dim, spacedim> &triangulation,
                                  const SparsityTools::Partitioner partitioner)
  {
    AssertDimension(dim, spacedim);
    Assert(n_partitions > 0,
           ExcMessage("Invalid number of partitions, you provided " +
                      std::to_string(n_partitions)));

    auto parallel_triangulation =
      dynamic_cast<parallel::fullydistributed::Triangulation<dim, spacedim> *>(
        &triangulation);
    Assert(
      (parallel_triangulation != nullptr),
      ExcMessage(
        "Only fully distributed triangulations are supported. If you are using"
        "a parallel::distributed::triangulation, you must convert it to a fully"
        "distributed as explained in the documentation."));

    // check for an easy return
    if (n_partitions == 1)
      {
        for (const auto &cell : parallel_triangulation->active_cell_iterators())
          if (cell->is_locally_owned())
            cell->set_material_id(0);
        return;
      }

    // collect all locally owned cells
    std::vector<types::global_cell_index> locally_owned_cells;
    for (const auto &cell : triangulation.active_cell_iterators())
      if (cell->is_locally_owned())
        locally_owned_cells.push_back(cell->active_cell_index());

    DynamicSparsityPattern cell_connectivity;
    internal::get_face_connectivity_of_cells(*parallel_triangulation,
                                             cell_connectivity,
                                             locally_owned_cells);

    SparsityPattern sp_cell_connectivity;
    sp_cell_connectivity.copy_from(cell_connectivity);

    // partition each locally owned connection graph and get
    // back a vector of indices, one per degree
    // of freedom (which is associated with a
    // cell)
    std::vector<unsigned int> partition_indices(
      parallel_triangulation->n_locally_owned_active_cells());
    SparsityTools::partition(sp_cell_connectivity,
                             n_partitions,
                             partition_indices,
                             partitioner);


    // finally loop over all cells and set the material ids
    for (const auto &cell : parallel_triangulation->active_cell_iterators())
      if (cell->is_locally_owned())
        cell->set_material_id(
          partition_indices[internal::get_index(locally_owned_cells,
                                                cell->active_cell_index())]);
  }



  template <int dim>
  std::
    tuple<std::vector<double>, std::vector<double>, std::vector<double>, double>
    compute_quality_metrics(const AgglomerationHandler<dim> &ah)
  {
    static_assert(dim == 2); // only 2D case is implemented.
#ifdef DEAL_II_WITH_CGAL
    using Kernel = CGAL::Exact_predicates_exact_constructions_kernel_with_sqrt;
    using Polygon_with_holes = typename CGAL::Polygon_with_holes_2<Kernel>;
    using Gt    = typename CGAL::Segment_Delaunay_graph_traits_2<Kernel>;
    using SDG2  = typename CGAL::Segment_Delaunay_graph_2<Gt>;
    using CDT   = typename CGAL::Constrained_Delaunay_triangulation_2<Kernel>;
    using CDTP  = typename CGAL::Constrained_triangulation_plus_2<CDT>;
    using Point = typename CDTP::Point;
    using Cid   = typename CDTP::Constraint_id;
    using Vertex_handle = typename CDTP::Vertex_handle;


    const auto compute_radius_inscribed_circle =
      [](const CGAL::Polygon_2<Kernel> &polygon) -> double {
      SDG2 sdg;

      sdg.insert_segments(polygon.edges_begin(), polygon.edges_end());

      double                               sd = 0, sqdist = 0;
      typename SDG2::Finite_faces_iterator fit = sdg.finite_faces_begin();
      for (; fit != sdg.finite_faces_end(); ++fit)
        {
          typename Kernel::Point_2 pp = sdg.primal(fit);
          for (int i = 0; i < 3; ++i)
            {
              assert(!sdg.is_infinite(fit->vertex(i)));
              if (fit->vertex(i)->site().is_segment())
                {
                  typename Kernel::Segment_2 s =
                    fit->vertex(i)->site().segment();
                  sqdist = CGAL::to_double(CGAL::squared_distance(pp, s));
                }
              else
                {
                  typename Kernel::Point_2 p = fit->vertex(i)->site().point();
                  sqdist = CGAL::to_double(CGAL::squared_distance(pp, p));
                }
            }

          if (polygon.bounded_side(pp) == CGAL::ON_BOUNDED_SIDE)
            sd = std::max(sqdist, sd);
        }

      return std::sqrt(sd);
    };

    const auto mesh_size = [&ah]() -> double {
      double hmax = 0.;
      for (const auto &polytope : ah.polytope_iterators())
        if (polytope->is_locally_owned())
          {
            const double diameter = polytope->diameter();
            if (diameter > hmax)
              hmax = diameter;
          }
      return hmax;
    }();


    // vectors holding quality metrics

    // ration between radius of radius_inscribed_circle and circumscribed circle
    std::vector<double> circle_ratios;
    std::vector<double> unformity_factors; // diameter of element over mesh size
    std::vector<double>
      box_ratio; // ratio between measure of bbox and measure of element.

    const std::vector<BoundingBox<dim>> &bboxes = ah.get_local_bboxes();
    // Loop over all polytopes and compute metrics.
    for (const auto &polytope : ah.polytope_iterators())
      {
        if (polytope->is_locally_owned())
          {
            const std::vector<typename Triangulation<dim>::active_face_iterator>
              &boundary = polytope->polytope_boundary();

            const double diameter                    = polytope->diameter();
            const double radius_circumscribed_circle = .5 * diameter;

            CDTP cdtp;
            for (unsigned int f = 0; f < boundary.size(); f += 1)
              {
                // polyline
                cdtp.insert_constraint(
                  {boundary[f]->vertex(0)[0], boundary[f]->vertex(0)[1]},
                  {boundary[f]->vertex(1)[0], boundary[f]->vertex(1)[1]});
              }
            cdtp.split_subconstraint_graph_into_constraints();

            CGAL::Polygon_2<Kernel> outer_polygon;
            auto                    it = outer_polygon.vertices_begin();
            for (typename CDTP::Constraint_id cid : cdtp.constraints())
              {
                for (typename CDTP::Vertex_handle vh :
                     cdtp.vertices_in_constraint(cid))
                  {
                    it = outer_polygon.insert(outer_polygon.vertices_end(),
                                              vh->point());
                  }
              }
            outer_polygon.erase(it); // remove duplicate final point

            const double radius_inscribed_circle =
              compute_radius_inscribed_circle(outer_polygon);

            circle_ratios.push_back(radius_inscribed_circle /
                                    radius_circumscribed_circle);
            unformity_factors.push_back(diameter / mesh_size);

            // box_ratio

            const auto  &agglo_values = ah.reinit(polytope);
            const double measure_element =
              std::accumulate(agglo_values.get_JxW_values().cbegin(),
                              agglo_values.get_JxW_values().cend(),
                              0.);
            box_ratio.push_back(measure_element /
                                bboxes[polytope->index()].volume());
          }
      }



    // Get all of the local bounding boxes
    double covering_bboxes = 0.;
    for (unsigned int i = 0; i < bboxes.size(); ++i)
      covering_bboxes += bboxes[i].volume();

    const double overlap_factor =
      Utilities::MPI::sum(covering_bboxes,
                          ah.get_dof_handler().get_communicator()) /
      GridTools::volume(ah.get_triangulation()); // assuming a linear mapping



    return {unformity_factors, circle_ratios, box_ratio, overlap_factor};
#else

    (void)ah;
    return {};
#endif
  }


  /**
   * Export each polygon in a csv file as a collection of segments.
   */
  template <int dim>
  void
  export_polygon_to_csv_file(
    const AgglomerationHandler<dim> &agglomeration_handler,
    const std::string               &filename)
  {
    static_assert(dim == 2); // With 3D, Paraview is much better
    std::ofstream myfile;
    myfile.open(filename + ".csv");

    for (const auto &polytope : agglomeration_handler.polytope_iterators())
      if (polytope->is_locally_owned())
        {
          const std::vector<typename Triangulation<dim>::active_face_iterator>
            &boundary = polytope->polytope_boundary();
          for (unsigned int f = 0; f < boundary.size(); ++f)
            {
              myfile << boundary[f]->vertex(0)[0];
              myfile << ",";
              myfile << boundary[f]->vertex(0)[1];
              myfile << ",";
              myfile << boundary[f]->vertex(1)[0];
              myfile << ",";
              myfile << boundary[f]->vertex(1)[1];
              myfile << "\n";
            }
        }


    myfile.close();
  }


  template <typename T>
  inline constexpr T
  constexpr_pow(T num, unsigned int pow)
  {
    return (pow >= sizeof(unsigned int) * 8) ? 0 :
           pow == 0                          ? 1 :
                                               num * constexpr_pow(num, pow - 1);
  }



  void
  write_to_matrix_market_format(const std::string &filename,
                                const std::string &matrix_name,
                                const TrilinosWrappers::SparseMatrix &matrix)
  {
#ifdef DEAL_II_WITH_TRILINOS
    const Epetra_CrsMatrix &trilinos_matrix = matrix.trilinos_matrix();

    const int ierr =
      EpetraExt::RowMatrixToMatrixMarketFile(filename.c_str(),
                                             trilinos_matrix,
                                             matrix_name.c_str(),
                                             0 /*description field empty*/,
                                             true /*write header*/);
    AssertThrow(ierr == 0, ExcTrilinosError(ierr));
#else
    (void)filename;
    (void)matrix_name;
    (void)matrix;
#endif
  }



  /**
   * Given a vector @p src, typically the solution stemming after the
   * agglomerate problem has been solved, this function interpolates @p src
   * onto the finer grid and stores the result in vector @p dst.
   *
   * @note Supported parallel types are TrilinosWrappers::SparseMatrix and TrilinosWrappers::MPI::Vector
   */
  template <int dim, int spacedim, typename VectorType>
  void
  interpolate_to_fine_grid(
    const AgglomerationHandler<dim, spacedim> &agglomeration_handler,
    VectorType                                &dst,
    const VectorType                          &src)
  {
    Assert((dim == spacedim), ExcNotImplemented());
    Assert(
      dst.size() == 0,
      ExcMessage(
        "The destination vector must the empt upon calling this function."));

    using NumberType = typename VectorType::value_type;
    constexpr bool is_trilinos_vector =
      std::is_same_v<VectorType, TrilinosWrappers::MPI::Vector>;
    using MatrixType = std::conditional_t<is_trilinos_vector,
                                          TrilinosWrappers::SparseMatrix,
                                          SparseMatrix<NumberType>>;

    MatrixType interpolation_matrix;

    [[maybe_unused]]
    typename std::conditional_t<!is_trilinos_vector, SparsityPattern, void *>
      sp;

    // Get some info from the handler
    const DoFHandler<dim, spacedim> &agglo_dh = agglomeration_handler.agglo_dh;

    DoFHandler<dim, spacedim> *output_dh =
      const_cast<DoFHandler<dim, spacedim> *>(&agglomeration_handler.output_dh);
    const FiniteElement<dim, spacedim> &fe = agglomeration_handler.get_fe();
    const Triangulation<dim, spacedim> &tria =
      agglomeration_handler.get_triangulation();
    const auto &bboxes = agglomeration_handler.get_local_bboxes();

    // Setup an auxiliary DoFHandler for output purposes
    output_dh->reinit(tria);
    output_dh->distribute_dofs(fe);

    const IndexSet &locally_owned_dofs = output_dh->locally_owned_dofs();
    const IndexSet  locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(*output_dh);

    const IndexSet &locally_owned_dofs_agglo = agglo_dh.locally_owned_dofs();


    DynamicSparsityPattern dsp(output_dh->n_dofs(),
                               agglo_dh.n_dofs(),
                               output_dh->locally_owned_dofs());

    std::vector<types::global_dof_index> agglo_dof_indices(fe.dofs_per_cell);
    std::vector<types::global_dof_index> standard_dof_indices(fe.dofs_per_cell);
    std::vector<types::global_dof_index> output_dof_indices(fe.dofs_per_cell);

    Quadrature<dim>         quad(fe.get_unit_support_points());
    FEValues<dim, spacedim> output_fe_values(fe,
                                             quad,
                                             update_quadrature_points);

    for (const auto &cell : agglo_dh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          if (agglomeration_handler.is_master_cell(cell))
            {
              auto slaves = agglomeration_handler.get_slaves_of_idx(
                cell->active_cell_index());
              slaves.emplace_back(cell);

              cell->get_dof_indices(agglo_dof_indices);

              for (const auto &slave : slaves)
                {
                  // addd master-slave relationship
                  const auto slave_output =
                    slave->as_dof_handler_iterator(*output_dh);
                  slave_output->get_dof_indices(output_dof_indices);
                  for (const auto row : output_dof_indices)
                    dsp.add_entries(row,
                                    agglo_dof_indices.begin(),
                                    agglo_dof_indices.end());
                }
            }
        }


    const auto assemble_interpolation_matrix = [&]() {
      FullMatrix<NumberType>  local_matrix(fe.dofs_per_cell, fe.dofs_per_cell);
      std::vector<Point<dim>> reference_q_points(fe.dofs_per_cell);

      // Dummy AffineConstraints, only needed for loc2glb
      AffineConstraints<NumberType> c;
      c.close();

      for (const auto &cell : agglo_dh.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            if (agglomeration_handler.is_master_cell(cell))
              {
                auto slaves = agglomeration_handler.get_slaves_of_idx(
                  cell->active_cell_index());
                slaves.emplace_back(cell);

                cell->get_dof_indices(agglo_dof_indices);

                const types::global_cell_index polytope_index =
                  agglomeration_handler.cell_to_polytope_index(cell);

                // Get the box of this agglomerate.
                const BoundingBox<dim> &box = bboxes[polytope_index];

                for (const auto &slave : slaves)
                  {
                    // add master-slave relationship
                    const auto slave_output =
                      slave->as_dof_handler_iterator(*output_dh);

                    slave_output->get_dof_indices(output_dof_indices);
                    output_fe_values.reinit(slave_output);

                    local_matrix = 0.;

                    const auto &q_points =
                      output_fe_values.get_quadrature_points();
                    for (const auto i : output_fe_values.dof_indices())
                      {
                        const auto &p = box.real_to_unit(q_points[i]);
                        for (const auto j : output_fe_values.dof_indices())
                          {
                            local_matrix(i, j) = fe.shape_value(j, p);
                          }
                      }
                    c.distribute_local_to_global(local_matrix,
                                                 output_dof_indices,
                                                 agglo_dof_indices,
                                                 interpolation_matrix);
                  }
              }
          }
    };


    if constexpr (std::is_same_v<MatrixType, TrilinosWrappers::SparseMatrix>)
      {
        const MPI_Comm &communicator = tria.get_communicator();
        SparsityTools::distribute_sparsity_pattern(dsp,
                                                   locally_owned_dofs,
                                                   communicator,
                                                   locally_relevant_dofs);

        interpolation_matrix.reinit(locally_owned_dofs,
                                    locally_owned_dofs_agglo,
                                    dsp,
                                    communicator);
        dst.reinit(locally_owned_dofs);
        assemble_interpolation_matrix();
      }
    else if constexpr (std::is_same_v<MatrixType, SparseMatrix<NumberType>>)
      {
        sp.copy_from(dsp);
        interpolation_matrix.reinit(sp);
        dst.reinit(output_dh->n_dofs());
        assemble_interpolation_matrix();
      }
    else
      {
        // PETSc, LA::d::v options not implemented.
        (void)agglomeration_handler;
        (void)dst;
        (void)src;
        AssertThrow(false, ExcNotImplemented());
      }

    // If tria is distributed
    if (dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
          &tria) != nullptr)
      interpolation_matrix.compress(VectorOperation::add);

    // Finally, perform the interpolation.
    interpolation_matrix.vmult(dst, src);
  }



  /**
   * Construct the interpolation matrix from the DG space defined the polytopic
   * elements
   * defined in @p agglomeration_handler to the DG space defined on the DoFHandler associated
   * to standard shapes. The interpolation matrix is assumed to be
   * default-constructed and is filled inside this function.
   */
  template <int dim, int spacedim, typename MatrixType>
  void
  fill_interpolation_matrix(
    const AgglomerationHandler<dim, spacedim> &agglomeration_handler,
    MatrixType                                &interpolation_matrix)
  {
    Assert((dim == spacedim), ExcNotImplemented());

    using NumberType = typename MatrixType::value_type;
    constexpr bool is_trilinos_matrix =
      std::is_same_v<MatrixType, TrilinosWrappers::MPI::Vector>;

    [[maybe_unused]]
    typename std::conditional_t<!is_trilinos_matrix, SparsityPattern, void *>
      sp;

    // Get some info from the handler
    const DoFHandler<dim, spacedim> &agglo_dh = agglomeration_handler.agglo_dh;

    DoFHandler<dim, spacedim> *output_dh =
      const_cast<DoFHandler<dim, spacedim> *>(&agglomeration_handler.output_dh);
    const FiniteElement<dim, spacedim> &fe = agglomeration_handler.get_fe();
    const Triangulation<dim, spacedim> &tria =
      agglomeration_handler.get_triangulation();
    const auto &bboxes = agglomeration_handler.get_local_bboxes();

    // Setup an auxiliary DoFHandler for output purposes
    output_dh->reinit(tria);
    output_dh->distribute_dofs(fe);

    const IndexSet &locally_owned_dofs = output_dh->locally_owned_dofs();
    const IndexSet  locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(*output_dh);

    const IndexSet &locally_owned_dofs_agglo = agglo_dh.locally_owned_dofs();


    DynamicSparsityPattern dsp(output_dh->n_dofs(),
                               agglo_dh.n_dofs(),
                               locally_relevant_dofs);

    std::vector<types::global_dof_index> agglo_dof_indices(fe.dofs_per_cell);
    std::vector<types::global_dof_index> standard_dof_indices(fe.dofs_per_cell);
    std::vector<types::global_dof_index> output_dof_indices(fe.dofs_per_cell);

    Quadrature<dim>         quad(fe.get_unit_support_points());
    FEValues<dim, spacedim> output_fe_values(fe,
                                             quad,
                                             update_quadrature_points);

    for (const auto &cell : agglo_dh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          if (agglomeration_handler.is_master_cell(cell))
            {
              auto slaves = agglomeration_handler.get_slaves_of_idx(
                cell->active_cell_index());
              slaves.emplace_back(cell);

              cell->get_dof_indices(agglo_dof_indices);

              for (const auto &slave : slaves)
                {
                  // addd master-slave relationship
                  const auto slave_output =
                    slave->as_dof_handler_iterator(*output_dh);
                  slave_output->get_dof_indices(output_dof_indices);
                  for (const auto row : output_dof_indices)
                    dsp.add_entries(row,
                                    agglo_dof_indices.begin(),
                                    agglo_dof_indices.end());
                }
            }
        }


    const auto assemble_interpolation_matrix = [&]() {
      FullMatrix<NumberType>  local_matrix(fe.dofs_per_cell, fe.dofs_per_cell);
      std::vector<Point<dim>> reference_q_points(fe.dofs_per_cell);

      // Dummy AffineConstraints, only needed for loc2glb
      AffineConstraints<NumberType> c;
      c.close();

      for (const auto &cell : agglo_dh.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            if (agglomeration_handler.is_master_cell(cell))
              {
                auto slaves = agglomeration_handler.get_slaves_of_idx(
                  cell->active_cell_index());
                slaves.emplace_back(cell);

                cell->get_dof_indices(agglo_dof_indices);

                const types::global_cell_index polytope_index =
                  agglomeration_handler.cell_to_polytope_index(cell);

                // Get the box of this agglomerate.
                const BoundingBox<dim> &box = bboxes[polytope_index];

                for (const auto &slave : slaves)
                  {
                    // add master-slave relationship
                    const auto slave_output =
                      slave->as_dof_handler_iterator(*output_dh);

                    slave_output->get_dof_indices(output_dof_indices);
                    output_fe_values.reinit(slave_output);

                    local_matrix = 0.;

                    const auto &q_points =
                      output_fe_values.get_quadrature_points();
                    for (const auto i : output_fe_values.dof_indices())
                      {
                        const auto &p = box.real_to_unit(q_points[i]);
                        for (const auto j : output_fe_values.dof_indices())
                          {
                            local_matrix(i, j) = fe.shape_value(j, p);
                          }
                      }
                    c.distribute_local_to_global(local_matrix,
                                                 output_dof_indices,
                                                 agglo_dof_indices,
                                                 interpolation_matrix);
                  }
              }
          }
    };


    if constexpr (std::is_same_v<MatrixType, TrilinosWrappers::SparseMatrix>)
      {
        const MPI_Comm &communicator = tria.get_communicator();
        SparsityTools::distribute_sparsity_pattern(dsp,
                                                   locally_owned_dofs,
                                                   communicator,
                                                   locally_relevant_dofs);

        interpolation_matrix.reinit(locally_owned_dofs,
                                    locally_owned_dofs_agglo,
                                    dsp,
                                    communicator);
        assemble_interpolation_matrix();
      }
    else if constexpr (std::is_same_v<MatrixType, SparseMatrix<NumberType>>)
      {
        sp.copy_from(dsp);
        interpolation_matrix.reinit(sp);
        assemble_interpolation_matrix();
      }
    else
      {
        // PETSc, LA::d::v options not implemented.
        (void)agglomeration_handler;
        AssertThrow(false, ExcNotImplemented());
      }

    // If tria is distributed
    if (dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
          &tria) != nullptr)
      interpolation_matrix.compress(VectorOperation::add);
  }



} // namespace dealii::PolyUtils

#endif
