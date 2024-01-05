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


#ifndef poly_utils_h
#define poly_utils_h


#include <deal.II/base/config.h>

#include <deal.II/base/point.h>
#include <deal.II/base/std_cxx20/iota_view.h>

#include <deal.II/boost_adaptors/bounding_box.h>
#include <deal.II/boost_adaptors/point.h>
#include <deal.II/boost_adaptors/segment.h>

#include <boost/geometry/algorithms/distance.hpp>
#include <boost/geometry/index/detail/rtree/utilities/print.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/geometry/strategies/strategies.hpp>

#include <deal.II/cgal/point_conversion.h>

#ifdef DEAL_II_WITH_CGAL

#  include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#  include <CGAL/intersections.h>
#  include <CGAL/squared_distance_2.h>
#  include <CGAL/squared_distance_3.h>


#endif

#include <memory>



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
      std::vector<std::vector<unsigned int>> &                           csr);


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
    const Translator & translator,
    const unsigned int target_level,
    std::vector<std::vector<typename Triangulation<
      boost::geometry::dimension<Box>::value>::active_cell_iterator>>
      &                                     bb_in_boxes,
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



  template <int dim,
            typename Kernel = CGAL::Exact_predicates_exact_constructions_kernel,
            typename Number = double>
  Number
  compute_h_orthogonal(
    const unsigned int face_index,
    const std::vector<typename Triangulation<dim>::active_face_iterator>
      &                   polygon_boundary,
    const Tensor<1, dim> &deal_normal)
  {
#ifdef DEAL_II_WITH_CGAL

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
    (void)face;
    (void)polygon_boundary;
    return {};
#endif
  }
} // namespace dealii::PolyUtils

#endif
