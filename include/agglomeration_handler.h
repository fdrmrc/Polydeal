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
#ifndef agglomeration_handler_h
#define agglomeration_handler_h

#include <deal.II/base/quadrature.h>
#include <deal.II/base/subscriptor.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <deal.II/meshworker/scratch_data.h>

#include <deal.II/non_matching/fe_immersed_values.h>
#include <deal.II/non_matching/immersed_surface_quadrature.h>

#include <agglomeration_iterator.h>

#include <fstream>
#include <utility>

using namespace dealii;

template <int dim, int spacedim>
class AgglomerationHandler;

namespace dealii
{
  namespace internal
  {
    template <int, int>
    class AgglomerationHandlerImplementation;
  }
} // namespace dealii

namespace dealii
{
  namespace IteratorFilters
  {
    /**
     * This predicate returns true if the present cell is not-slave.
     *
     */
    template <int dim, int spacedim>
    class IsNotSlave
    {
    public:
      IsNotSlave(AgglomerationHandler<dim, spacedim> *agglo_handler_ptr)
      {
        ah = agglo_handler_ptr;
      }

      template <typename CellIterator>
      bool
      operator()(const CellIterator &cell) const
      {
        return !ah->is_slave_cell(cell);
      }

    private:
      AgglomerationHandler<dim, spacedim> *ah;
    };

  } // namespace IteratorFilters
} // namespace dealii

/**
 * Assumption: each cell may have only one master cell
 */
template <int dim, int spacedim = dim>
class AgglomerationHandler : public Subscriptor
{
public:
  using agglomeration_iterator = AgglomerationIterator<dim, spacedim>;

  using AgglomerationContainer =
    typename AgglomerationIterator<dim, spacedim>::AgglomerationContainer;


  enum CellAgglomerationType
  {
    master   = 0,
    slave    = 1,
    standard = 2
  };

  /**
   * Record the number of agglomerations on the grid.
   */
  unsigned int n_agglomerations;

  /**
   * DoFHandler for the agglomerated space
   */
  DoFHandler<dim, spacedim> agglo_dh;

  /**
   * DoFHandler for the finest space: classical deal.II space
   */
  DoFHandler<dim, spacedim> output_dh;

  /**
   * Sparsity to interpolate on the output dh.
   */
  SparsityPattern output_interpolation_sparsity;

  /**
   * Interpolation matrix for the output dh.
   */
  SparseMatrix<double> output_interpolation_matrix;

  explicit AgglomerationHandler(
    const GridTools::Cache<dim, spacedim> &cached_tria);

  AgglomerationHandler() = default;

  ~AgglomerationHandler()
  {
    // disconnect the signal
    tria_listener.disconnect();
  }

  /**
   * Iterator to the first polytope.
   */
  agglomeration_iterator
  begin() const;

  /**
   * Iterator to the first polytope.
   */
  agglomeration_iterator
  begin();

  /**
   * Iterator to one past the last polygonal element.
   */
  agglomeration_iterator
  end() const;

  /**
   * Iterator to one past the last polygonal element.
   */
  agglomeration_iterator
  end();

  /**
   * Iterator to the last polygonal element.
   */
  agglomeration_iterator
  last();

  /**
   * Returns an IteratorRange that makes up all the polygonal elements in the
   * mesh.
   */
  IteratorRange<agglomeration_iterator>
  polytope_iterators() const;

  template <int, int>
  friend class AgglomerationIterator;

  template <int, int>
  friend class AgglomerationAccessor;

  /**
   * Distribute degrees of freedom on a grid where some cells have been
   * agglomerated.
   */
  template <class FiniteElement>
  void
  distribute_agglomerated_dofs(const FiniteElement &fe_space)
  {
    Assert((dynamic_cast<const FE_DGQ<dim, spacedim> *>(&fe_space)),
           ExcNotImplemented(
             "Currently, this interface supports only DG discretizations."));
    fe = std::make_unique<FE_DGQ<dim, spacedim>>(fe_space);

    fe_collection.push_back(*fe);                         // master
    fe_collection.push_back(FE_Nothing<dim, spacedim>()); // slave
    fe_collection.push_back(*fe);                         // standard

    initialize_hp_structure();
    setup_connectivity_of_agglomeration();
  }


  /**
   *
   * Set the degree of the quadrature formula to be used and the proper flags
   * for the FEValues object on the agglomerated cell.
   */
  void
  initialize_fe_values(
    const Quadrature<dim> &    cell_quadrature = QGauss<dim>(1),
    const UpdateFlags &        flags           = UpdateFlags::update_default,
    const Quadrature<dim - 1> &face_quadrature = QGauss<dim - 1>(1),
    const UpdateFlags &        face_flags      = UpdateFlags::update_default)
  {
    agglomeration_quad       = cell_quadrature;
    agglomeration_flags      = flags;
    agglomeration_face_quad  = face_quadrature;
    agglomeration_face_flags = face_flags | internal_agglomeration_face_flags;


    no_values = std::make_unique<FEValues<dim>>(
      *mapping,
      dummy_fe,
      agglomeration_quad,
      update_quadrature_points | update_JxW_values); // only for quadrature
    no_face_values = std::make_unique<FEFaceValues<dim>>(
      *mapping,
      dummy_fe,
      agglomeration_face_quad,
      update_quadrature_points | update_JxW_values |
        update_normal_vectors); // only for quadrature
  }

  /**
   * Given a Triangulation with some agglomerated cells, create the sparsity
   * pattern corresponding to a Discontinuous Galerkin discretization where the
   * agglomerated cells are seen as one **unique** cell, with only the DoFs
   * associated to the master cell of the agglomeration.
   */
  void
  create_agglomeration_sparsity_pattern(SparsityPattern &sparsity_pattern);

  /**
   * Store internally that the given cells are agglomerated. The convenction we
   * take is the following:
   * -2: default value, standard deal.II cell
   * -1: cell is a master cell
   *
   * @note cells are assumed to be adjacent one to each other, and no check
   * about this is done.
   */
  agglomeration_iterator
  insert_agglomerate(const AgglomerationContainer &cells);

  /**
   * Get the connectivity of the agglomeration. TODO: this data structure should
   * be private. Keep the getter just for the time being.
   */
  inline decltype(auto)
  get_agglomerated_connectivity()
  {
    return master_neighbors;
  }

  inline const std::vector<long int> &
  get_relationships() const
  {
    return master_slave_relationships;
  }

  /**
   * TODO: remove this in favour of the accessor version.
   *
   * @param master_cell
   * @return std::vector<
   * typename Triangulation<dim, spacedim>::active_cell_iterator>
   */
  inline std::vector<
    typename Triangulation<dim, spacedim>::active_cell_iterator>
  get_agglomerate(
    const typename Triangulation<dim, spacedim>::active_cell_iterator
      &master_cell) const
  {
    Assert(is_master_cell(master_cell), ExcInternalError());
    auto agglomeration = get_slaves_of_idx(master_cell->active_cell_index());
    agglomeration.push_back(master_cell);
    return agglomeration;
  }

  /**
   * Display the indices of the vector identifying which cell is agglomerated
   * with which master.
   */
  template <class StreamType>
  void
  print_agglomeration(StreamType &out)
  {
    for (const auto &cell : euler_dh.active_cell_iterators())
      out << "Cell with index: " << cell->active_cell_index()
          << " has associated value: "
          << master_slave_relationships[cell->active_cell_index()] << std::endl;
  }



  /**
   *
   * TODO: remove this in favour of the accessor version
   * Return a vector of BoundingBox. Each one of the bounding boxes bounds an
   * agglomeration present in your triangulation.
   */
  inline const std::vector<BoundingBox<spacedim>> &
  get_bboxes() const
  {
    return bboxes;
  }


  /**
   *
   * Return a constant reference to the DoFHandler underlying the
   * agglomeration. It knows which cell have been agglomerated, and which FE
   * spaces are present on each cell of the triangulation.
   */
  inline const DoFHandler<dim, spacedim> &
  get_dof_handler() const
  {
    return agglo_dh;
  }

  /**
   * Return the number of agglomerated faces for a generic deal.II cell. If it's
   * a standard cell, the result is 0.
   */
  unsigned int
  n_agglomerated_faces_per_cell(
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
    const
  {
    unsigned int n_neighbors = 0;
    for (const auto &f : cell->face_indices())
      {
        const auto &neighboring_cell = cell->neighbor(f);
        if ((cell->face(f)->at_boundary()) ||
            (neighboring_cell->is_active() &&
             !are_cells_agglomerated(cell, neighboring_cell)))
          {
            ++n_neighbors;
          }
      }
    return n_neighbors;
  }



  /**
   * Return, for a cell, the number of faces. In case the cell is a standard
   * cell, then the number of faces is the classical one. If it's a master cell,
   * then it returns the number of faces of the agglomeration identified by the
   * master cell itself.
   */
  unsigned int
  n_faces(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell) const
  {
    Assert(!is_slave_cell(cell), ExcMessage("You cannot pass a slave cell."));
    if (is_standard_cell(cell))
      {
        return cell->n_faces();
      }
    else
      {
        const auto & agglomeration = get_agglomerate(cell);
        unsigned int n_neighbors   = 0;
        for (const auto &cell : agglomeration)
          {
            n_neighbors += n_agglomerated_faces_per_cell(cell);
          }
        return n_neighbors;
      }
  }



  /**
   * Return, for each cell belonging to the same agllomeration, a cell belonging
   * to the neighbor agglomeration
   */
  typename DoFHandler<dim, spacedim>::active_cell_iterator
  agglomerated_neighbor(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
    const unsigned int                                              f) const
  {
    Assert(!is_slave_cell(cell),
           ExcMessage("This cells is not supposed to be a slave."));
    if (!at_boundary(cell, f))
      {
        if (is_standard_cell(cell) && is_standard_cell(cell->neighbor(f)))
          {
            return cell->neighbor(f);
          }
        else if (is_master_cell(cell))
          {
            const auto &neigh = std::get<1>(master_neighbors[{cell, f}]);
            typename DoFHandler<dim, spacedim>::active_cell_iterator cell_dh(
              *neigh, &agglo_dh);
            return cell_dh;
          }
        else
          {
            // If I fall here, I want to find the neighbor for a standard cell
            // adjacent to an agglomeration
            const auto &master_cell =
              master_slave_relationships_iterators[cell->neighbor(f)
                                                     ->active_cell_index()];
            typename DoFHandler<dim, spacedim>::active_cell_iterator cell_dh(
              *master_cell, &agglo_dh);
            return cell_dh;
          }
      }
    else
      {
        return {};
      }
  }



  /**
   * Generalize the 'cell->neighbor_of_neighbor(f)' to the case where 'cell' is:
   * - a master cell of an agglomeration.
   * - a standard cell adjacent to an agglomeration.
   */
  unsigned int
  neighbor_of_agglomerated_neighbor(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
    const unsigned int                                              f) const
  {
    Assert(!is_slave_cell(cell),
           ExcMessage("This cells is not supposed to be a slave."));
    // First, make sure it's not a boundary face.
    if (!at_boundary(cell, f))
      {
        if (is_standard_cell(cell) && is_standard_cell(cell->neighbor(f)))
          {
            return cell->neighbor_of_neighbor(f);
          }
        else if (is_master_cell(cell) &&
                 is_master_cell(agglomerated_neighbor(cell, f)))
          {
            const auto &current_agglo_info = master_neighbors[{cell, f}];
            const auto &local_f_idx        = std::get<0>(current_agglo_info);
            const auto &current_cell       = std::get<3>(current_agglo_info);

            const auto &agglo_neigh =
              agglomerated_neighbor(cell,
                                    f); // returns the neighboring master
            const unsigned int n_faces_agglomerated_neighbor =
              n_faces(agglo_neigh);
            // Loop over all cells of neighboring agglomerate
            for (unsigned int f_out = 0; f_out < n_faces_agglomerated_neighbor;
                 ++f_out)
              {
                if (agglomerated_neighbor(agglo_neigh, f_out).state() ==
                      IteratorState::valid &&
                    current_cell->neighbor(local_f_idx).state() ==
                      IteratorState::valid &&
                    current_cell.state() == IteratorState::valid)
                  {
                    const auto &neighboring_agglo_info =
                      master_neighbors[{agglo_neigh, f_out}];
                    const auto &local_f_out_idx =
                      std::get<0>(neighboring_agglo_info);
                    const auto &current_cell_out =
                      std::get<3>(neighboring_agglo_info);
                    const auto &other_standard =
                      std::get<1>(neighboring_agglo_info);

                    // Here, an extra condition is needed because there can be
                    // more than one face index that returns the same neighbor
                    // if you simply check who is f' s.t.
                    // cell->neigh(f)->neigh(f') == cell. Hence, an extra
                    // condition must be added.

                    if (other_standard.state() == IteratorState::valid &&
                        agglomerated_neighbor(agglo_neigh, f_out)
                            ->active_cell_index() ==
                          cell->active_cell_index() &&
                        current_cell->active_cell_index() ==
                          current_cell_out->neighbor(local_f_out_idx)
                            ->active_cell_index() &&
                        current_cell->neighbor(local_f_idx) == current_cell_out)
                      return f_out;
                  }
              }
            Assert(false, ExcInternalError());
            return {}; // just to suppress warnings
          }
        else if (is_master_cell(cell) &&
                 is_standard_cell(agglomerated_neighbor(cell, f)))
          {
            return std::get<2>(master_neighbors[{cell, f}]);
          }
        else
          {
            // If I fall here, I want to find the neighbor of neighbor for a
            // standard cell adjacent to an agglomeration.
            const auto &master_cell = agglomerated_neighbor(
              cell, f); // this is the master of the neighboring agglomeration

            return shared_face_agglomeration_idx[{master_cell, cell, f}];
          }
      }
    else
      {
        // Face is at boundary
        return numbers::invalid_unsigned_int;
      }
  }

  /**
   * Construct a finite element space on the agglomeration.
   */
  const FEValues<dim, spacedim> &
  reinit(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell) const;

  /**
   * Construct a finite element space on the agglomeration.
   */
  const FEValues<dim, spacedim> &
  reinit(const AgglomerationIterator<dim, spacedim> &polytope) const;

  /**
   * For a given master cell `cell` and agglomerated face
   * `face_number`, initialize shape functions, normals and
   * quadratures.
   */
  const FEValuesBase<dim, spacedim> &
  reinit(const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
         const unsigned int face_index) const;

  /**
   * For a given polytope and face index, initialize shape functions, normals
   * and quadratures rules to integrate there.
   */
  const FEValuesBase<dim, spacedim> &
  reinit(const AgglomerationIterator<dim, spacedim> &polytope,
         const unsigned int                          face_index) const;

  /**
   *
   * Return a pair of FEValuesBase object reinited from the two sides of the
   * agglomeration.
   */
  std::pair<const FEValuesBase<dim, spacedim> &,
            const FEValuesBase<dim, spacedim> &>
  reinit_interface(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell_in,
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &neigh_cell,
    const unsigned int                                              local_in,
    const unsigned int local_outside) const;

  /**
   *
   * Return a pair of FEValuesBase object reinited from the two sides of the
   * agglomeration.
   */
  std::pair<const FEValuesBase<dim, spacedim> &,
            const FEValuesBase<dim, spacedim> &>
  reinit_interface(const AgglomerationIterator<dim, spacedim> &polytope_in,
                   const AgglomerationIterator<dim, spacedim> &neigh_polytope,
                   const unsigned int                          local_in,
                   const unsigned int local_outside) const;

  /**
   * Return the agglomerated quadrature for the given agglomeration. This
   * amounts to loop over all cells in an agglomeration and collecting together
   * all the rules.
   */
  Quadrature<dim>
  agglomerated_quadrature(
    const AgglomerationContainer &cells,
    const typename Triangulation<dim, spacedim>::active_cell_iterator
      &master_cell) const;



  /**
   * Find (if any) the cells that have the given master index. Note that `idx`
   * is as it can be equal to -1 (meaning that the cell is a master one).
   */
  inline const std::vector<
    typename Triangulation<dim, spacedim>::active_cell_iterator> &
  get_slaves_of_idx(types::global_cell_index idx) const
  {
    return master2slaves.at(idx);
  }



  /**
   * Helper function to determine whether or not a cell is a master or a slave
   */
  template <typename CellIterator>
  inline bool
  is_master_cell(const CellIterator &cell) const
  {
    return master_slave_relationships[cell->active_cell_index()] == -1;
  }



  /**
   * Helper function to determine if the given cell is a standard deal.II cell,
   * that is: not master, nor slave.
   *
   */
  template <typename CellIterator>
  inline bool
  is_standard_cell(const CellIterator &cell) const
  {
    return master_slave_relationships[cell->active_cell_index()] == -2;
  }



  /**
   * Helper function to determine whether or not a cell is a slave cell, without
   * any information about his parents.
   */
  template <typename CellIterator>
  inline bool
  is_slave_cell(const CellIterator &cell) const
  {
    return master_slave_relationships[cell->active_cell_index()] >= 0;
  }



  /**
   *
   * This function generalizes the behaviour of cell->face(f)->at_boundary()
   * in the case where f is an index out of the range [0,..., n_faces).
   * In practice, if you call this function with a standard deal.II cell, you
   * have precisely the same result as calling cell->face(f)->at_boundary().
   * Otherwise, if the cell is a master one, you have a boolean returning true
   * is that face for the agglomeration is on the boundary or not.
   */
  bool
  at_boundary(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
    const unsigned int                                              f) const
  {
    Assert(!is_slave_cell(cell),
           ExcMessage("This function should not be called for a slave cell."));
    if (is_standard_cell(cell))
      return cell->face(f)->at_boundary();
    else
      return std::get<2>(master_neighbors[{cell, f}]) ==
             std::numeric_limits<unsigned int>::max();
  }


  decltype(auto)
  agglomeration_cell_iterators()
  {
    return agglo_dh.active_cell_iterators() |
           IteratorFilters::IsNotSlave(this); // iterator range
  }

  std::unique_ptr<MappingFEField<dim, spacedim> /*, Vector<double>*/>
    euler_mapping;

  inline unsigned int
  n_dofs_per_cell() const noexcept
  {
    return fe->n_dofs_per_cell();
  }

  inline types::global_dof_index
  n_dofs() const noexcept
  {
    return agglo_dh.n_dofs();
  }

  /**
   * Interpolate the solution defined on the agglomerates onto a classical
   * deal.II DoFHandler.
   */
  void
  setup_output_interpolation_matrix();

  /**
   *
   * Compute the volume of an agglomerate. This @p cell argument takes the
   * deal.II cell that identifies an agglomerate. If it's a standard cell,
   * than the this function is equivalent to cell->volume(). An exception is
   * thrown is the given cell is a slave cell.
   */
  double
  volume(const typename Triangulation<dim>::active_cell_iterator &cell) const;

  /*
   * Compute the diameter $h_K$ for a given agglomerate $K$. If $K$ is a
   * standard cell, this is equivalent to call cell->diameter(). Otherwise, this
   * function computes the diameter of the bounding box associated with $K$.
   */
  double
  diameter(const typename Triangulation<dim>::active_cell_iterator &cell) const;

  /**
   * Return the collection of vertices describing the boundary of the polytope
   * associated to the master cell `cell`. The return type is meant to describe
   * a sequence of edges (in 2D) or faces (in 3D).
   */
  inline const std::vector<typename Triangulation<dim>::active_face_iterator> &
  polytope_boundary(
    const typename Triangulation<dim>::active_cell_iterator &cell)
  {
    return polygon_boundary[cell];
  }

private:
  /**
   * Vector of indices such that v[cell->active_cell_index()] returns
   * { -1 if `cell` is a master cell
   * { -2 if `cell` is a standard deal.II cell
   * { `cell_master->active_cell_index()`, i.e. the index of the master cell if
   * `cell` is a slave cell.
   */
  std::vector<long int> master_slave_relationships;


  /**
   *  Same as the one above, but storing cell iterators rather than indices.
   *
   */
  std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>
    master_slave_relationships_iterators;

  using ScratchData = MeshWorker::ScratchData<dim, spacedim>;

  // In order to enumerate the faces of an agglomeration, we consider a map
  // where the key is represented by an iterator to the (master) cell and the
  // index of the agglomerated face.
  using CellAndFace =
    std::pair<const typename Triangulation<dim, spacedim>::active_cell_iterator,
              types::global_cell_index>;

  using MasterAndNeighborAndFace = std::tuple<
    const typename Triangulation<dim, spacedim>::active_cell_iterator,
    const typename Triangulation<dim, spacedim>::active_cell_iterator,
    types::global_cell_index>;

  // As value, we have a vector where each element identifies:
  // - the local face index from the agglomeration
  // - a cell_iterator to the neighboring cell (which is outside of the
  // agglomeration)
  // - the face index seen from outside
  // This is necessary to compute quadrature rules on each agglomerated face.
  using MasterNeighborInfo = std::tuple<
    types::global_cell_index,
    const typename Triangulation<dim, spacedim>::active_cell_iterator,
    types::global_cell_index,
    const typename Triangulation<dim, spacedim>::active_cell_iterator>;

  /**
   * For each pair (master_cell,agglo_face_number), give the following
   * information:
   * - local face index (standard deal.II indexing)
   * - neighboring cell (standard deal.II cell)
   * - neighbor of neighbor (for standard deal.II cell)
   * - deal.II cell that owns that face of the agglomerate element
   *
   */
  mutable std::map<CellAndFace, MasterNeighborInfo> master_neighbors;

  /**
   * Associate a master cell (hence, a given polytope) to its boundary faces.
   * The boundary is described through a vector of face iterators.
   *
   */
  mutable std::map<
    const typename Triangulation<dim, spacedim>::active_cell_iterator,
    std::vector<typename Triangulation<dim>::active_face_iterator>>
    polygon_boundary;

  mutable std::map<MasterAndNeighborAndFace, types::global_cell_index>
    shared_face_agglomeration_idx;

  /**
   * Vector of `BoundingBoxes` s.t. `bboxes[idx]` equals BBOx associated to the
   * agglomeration with master cell indexed by ìdx`. Othwerwise default BBox is
   * empty
   *
   */
  std::vector<BoundingBox<spacedim>> bboxes;

  SmartPointer<const Triangulation<dim, spacedim>> tria;

  SmartPointer<const Mapping<dim, spacedim>> mapping;

  std::unique_ptr<FESystem<dim, spacedim>> euler_fe;

  std::unique_ptr<GridTools::Cache<dim, spacedim>> cached_tria;

  // The FE_DGQ space we have on each cell
  std::unique_ptr<FE_DGQ<dim, spacedim>> fe;

  hp::FECollection<dim, spacedim> fe_collection;

  /**
   * DoFHandler for the physical space
   */
  DoFHandler<dim, spacedim> euler_dh;

  /**
   * Eulerian vector describing the new cells obtained by the bounding boxes
   */
  Vector<double> euler_vector;


  /**
   * Use this in reinit(cell) for standard (non-agglomerated) standard cells,
   * and return the result of scratch.reinit(cell) for cells
   */
  mutable std::unique_ptr<ScratchData> standard_scratch;

  mutable std::unique_ptr<ScratchData> standard_scratch_face;

  mutable std::unique_ptr<ScratchData> standard_scratch_face_any;

  mutable std::unique_ptr<ScratchData> standard_scratch_face_bdary;

  mutable std::unique_ptr<ScratchData> standard_scratch_face_std;

  mutable std::unique_ptr<ScratchData> standard_scratch_face_std_neigh;

  mutable std::unique_ptr<ScratchData> standard_scratch_face_std_another;

  /**
   * Fill this up in reinit(cell), for agglomerated cells, using the custom
   * quadrature, and return the result of
   * scratch.reinit(cell);
   */
  mutable std::unique_ptr<ScratchData> agglomerated_scratch;


  mutable std::unique_ptr<NonMatching::FEImmersedSurfaceValues<spacedim>>
    agglomerated_isv;

  mutable std::unique_ptr<NonMatching::FEImmersedSurfaceValues<spacedim>>
    agglomerated_isv_neigh;

  mutable std::unique_ptr<NonMatching::FEImmersedSurfaceValues<spacedim>>
    agglomerated_isv_bdary;

  boost::signals2::connection tria_listener;

  UpdateFlags agglomeration_flags;

  UpdateFlags agglomeration_face_flags;

  const UpdateFlags internal_agglomeration_face_flags =
    update_quadrature_points | update_normal_vectors | update_values |
    update_gradients | update_JxW_values | update_inverse_jacobians;

  Quadrature<dim> agglomeration_quad;

  Quadrature<dim - 1> agglomeration_face_quad;

  // Associate the master cell to the slaves.
  std::unordered_map<
    types::global_cell_index,
    std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>>
    master2slaves;

  // Map the master cell index with the polytope index
  std::unordered_map<types::global_cell_index, types::global_cell_index>
    master2polygon;

  // Dummy FiniteElement objects needed only to generate quadratures

  /**
   * Dummy FE_Nothing
   */
  FE_Nothing<dim, spacedim> dummy_fe;

  /**
   * Dummy FEValues, needed for cell quadratures.
   */
  std::unique_ptr<FEValues<dim, spacedim>> no_values;

  /**
   * Dummy FEFaceValues, needed for face quadratures.
   */
  std::unique_ptr<FEFaceValues<dim, spacedim>> no_face_values;

  /**
   * A contiguous container for all of the master cells.
   */
  std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>
    master_cells_container;

  /**
   * Initialize connectivity informations
   */
  void
  initialize_agglomeration_data(
    const std::unique_ptr<GridTools::Cache<dim, spacedim>> &cache_tria)
  {
    tria    = &cache_tria->get_triangulation();
    mapping = &cache_tria->get_mapping();

    agglo_dh.reinit(*tria);
    FE_DGQ<dim, spacedim> dummy_dg(1);
    euler_fe = std::make_unique<FESystem<dim, spacedim>>(dummy_dg, spacedim);
    euler_dh.reinit(*tria);
    euler_dh.distribute_dofs(*euler_fe);
    euler_vector.reinit(euler_dh.n_dofs());

    master_slave_relationships.resize(tria->n_active_cells(), -2);
    master_slave_relationships_iterators.resize(tria->n_active_cells(), {});
    if (n_agglomerations > 0)
      std::fill(master_slave_relationships.begin(),
                master_slave_relationships.end(),
                -2); // identify all the tria with standard deal.II cells.

    master_neighbors.clear();
    // bboxes.resize(tria->n_active_cells());

    // First, update the pointer
    cached_tria = std::make_unique<GridTools::Cache<dim, spacedim>>(
      cache_tria->get_triangulation(), cache_tria->get_mapping());

    connect_to_tria_signals();
    n_agglomerations = 0;
  }

  /**
   * Reinitialize the agglomeration data.
   */
  void
  connect_to_tria_signals()
  {
    // First disconnect existing connections
    tria_listener.disconnect();
    tria_listener = tria->signals.any_change.connect(
      [&]() { this->initialize_agglomeration_data(this->cached_tria); });
  }

  /**
   * Helper function to determine whether or not a cell is a slave cell.
   * Instead of returning a boolean, it gives the index of the master cell. If
   * it's a master cell, then the it returns -1, by construction.
   */

  inline typename Triangulation<dim, spacedim>::active_cell_iterator &
  is_slave_cell_of(
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
  {
    return master_slave_relationships_iterators[cell->active_cell_index()];
  }

  /**
   * Construct bounding boxes for an agglomeration described by a sequence of
   * cells. This fills also the euler vector
   */
  void
  create_bounding_box(const AgglomerationContainer & polytope,
                      const types::global_cell_index master_idx)
  {
    Assert(n_agglomerations > 0,
           ExcMessage("No agglomeration has been performed."));
    Assert(dim > 1, ExcNotImplemented());

    std::vector<types::global_dof_index> dof_indices(euler_fe->dofs_per_cell);
    std::vector<Point<spacedim>>         pts; // store all the vertices
    for (const auto &cell : polytope)
      {
        typename DoFHandler<dim, spacedim>::cell_iterator cell_dh(*cell,
                                                                  &euler_dh);
        cell_dh->get_dof_indices(dof_indices);
        for (const auto i : cell->vertex_indices())
          pts.push_back(cell->vertex(i));
      }

    bboxes.emplace_back(pts);

    const auto &p0 =
      bboxes[master2polygon.at(master_idx)].get_boundary_points().first;
    const auto &p1 =
      bboxes[master2polygon.at(master_idx)].get_boundary_points().second;
    if constexpr (dim == 2)
      {
        euler_vector[dof_indices[0]] = p0[0];
        euler_vector[dof_indices[4]] = p0[1];
        // Lower right
        euler_vector[dof_indices[1]] = p1[0];
        euler_vector[dof_indices[5]] = p0[1];
        // Upper left
        euler_vector[dof_indices[2]] = p0[0];
        euler_vector[dof_indices[6]] = p1[1];
        // Upper right
        euler_vector[dof_indices[3]] = p1[0];
        euler_vector[dof_indices[7]] = p1[1];
      }
    else if constexpr (dim == 3)
      {
        // Lowers

        // left
        euler_vector[dof_indices[0]]  = p0[0];
        euler_vector[dof_indices[8]]  = p0[1];
        euler_vector[dof_indices[16]] = p0[2];

        // right
        euler_vector[dof_indices[1]]  = p1[0];
        euler_vector[dof_indices[9]]  = p0[1];
        euler_vector[dof_indices[17]] = p0[2];

        // left
        euler_vector[dof_indices[2]]  = p0[0];
        euler_vector[dof_indices[10]] = p1[1];
        euler_vector[dof_indices[18]] = p0[2];

        // right
        euler_vector[dof_indices[3]]  = p1[0];
        euler_vector[dof_indices[11]] = p1[1];
        euler_vector[dof_indices[19]] = p0[2];

        // Uppers

        // left
        euler_vector[dof_indices[4]]  = p0[0];
        euler_vector[dof_indices[12]] = p0[1];
        euler_vector[dof_indices[20]] = p1[2];

        // right
        euler_vector[dof_indices[5]]  = p1[0];
        euler_vector[dof_indices[13]] = p0[1];
        euler_vector[dof_indices[21]] = p1[2];

        // left
        euler_vector[dof_indices[6]]  = p0[0];
        euler_vector[dof_indices[14]] = p1[1];
        euler_vector[dof_indices[22]] = p1[2];

        // right
        euler_vector[dof_indices[7]]  = p1[0];
        euler_vector[dof_indices[15]] = p1[1];
        euler_vector[dof_indices[23]] = p1[2];
      }
    else
      {
        Assert(false, ExcNotImplemented());
      }
  }


  inline types::global_cell_index
  get_master_idx_of_cell(
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
    const
  {
    auto idx = master_slave_relationships[cell->active_cell_index()];
    if ((idx == -1) || (idx == -2))
      return cell->active_cell_index();
    else
      return idx;
  }
  /**
   * Returns true if the two given cells are agglomerated together.
   */
  inline bool
  are_cells_agglomerated(
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell,
    const typename Triangulation<dim, spacedim>::active_cell_iterator
      &other_cell) const
  {
    return (get_master_idx_of_cell(cell) == get_master_idx_of_cell(other_cell));
  }



  /**
   * Assign a finite element index on each cell of a triangulation, depending
   * if it is a master cell, a slave cell, or a standard deal.II cell. A user
   * doesn't need to know the internals of this, the only thing that is
   * relevant is that after the call to the present function, DoFs are
   * distributed in a different way if a cell is a master, slave, or standard
   * cell.
   */
  void
  initialize_hp_structure();

  /**
   * Helper function to call reinit on a master cell.
   */
  const FEValuesBase<dim, spacedim> &
  reinit_master(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
    const unsigned int                                              face_number,
    std::unique_ptr<NonMatching::FEImmersedSurfaceValues<spacedim>>
      &agglo_isv_ptr) const;

  /**
   * Given an agglomeration described by the master cell `master_cell`, this
   * function:
   * - enumerates the faces of the agglomeration
   * - stores who is the neighbor, the local face indices from outside and
   * inside
   */
  void
  setup_master_neighbor_connectivity(
    const typename Triangulation<dim, spacedim>::active_cell_iterator
      &master_cell) const
  {
    Assert(master_slave_relationships[master_cell->active_cell_index()] == -1,
           ExcMessage("The present cell is not a master one."));

    const auto &agglomeration = get_agglomerate(master_cell);

    unsigned int n_agglo_faces = 0;
    for (const auto &cell : agglomeration)
      {
        for (const auto f : cell->face_indices())
          {
            const auto &neighboring_cell = cell->neighbor(f);
            // Check if cell is not on the boundary and if it's not
            // agglomerated with the neighbor If so, it's a neighbor of the
            // present agglomeration
            if (neighboring_cell.state() == IteratorState::valid &&
                !are_cells_agglomerated(cell, neighboring_cell))
              {
                // a new face of the agglomeration has been discovered.
                polygon_boundary[master_cell].push_back(cell->face(f));

                const auto &cell_and_face =
                  CellAndFace(master_cell, n_agglo_faces);      //(agglo,f)
                const auto nof = cell->neighbor_of_neighbor(f); // loc(f')

                if (is_slave_cell(neighboring_cell))
                  master_neighbors.emplace(
                    cell_and_face,
                    std::make_tuple(f,
                                    master_slave_relationships_iterators
                                      [neighboring_cell->active_cell_index()],
                                    nof,
                                    cell));
                else
                  master_neighbors.emplace(
                    cell_and_face,
                    std::make_tuple(
                      f,
                      neighboring_cell,
                      nof,
                      cell)); //(agglo,f)
                              //->(loc(f),other_deal_cell,loc(f'),dealcell)

                // Now, link the index of the agglomerated face with the
                // master and the neighboring cell.
                shared_face_agglomeration_idx.emplace(
                  MasterAndNeighborAndFace(master_cell, neighboring_cell, nof),
                  n_agglo_faces);
                ++n_agglo_faces;
              }
            else if (cell->face(f)->at_boundary())
              {
                // Boundary face of a boundary cell.
                // Note that the neighboring cell must be invalid.
                const auto &cell_and_face =
                  CellAndFace(master_cell, n_agglo_faces);

                polygon_boundary[master_cell].push_back(cell->face(f));

                master_neighbors.emplace(
                  cell_and_face,
                  std::make_tuple(f,
                                  neighboring_cell,
                                  std::numeric_limits<unsigned int>::max(),
                                  cell)); // TODO: check what the last element
                                          // should be...
                ++n_agglo_faces;
              }
          }
      }
  }

  /**
   * Initialize all the necessary connectivity information for an
   * agglomeration.
   */
  void
  setup_connectivity_of_agglomeration()
  {
    Assert(
      agglo_dh.n_dofs() > 0,
      ExcMessage(
        "The DoFHandler associated to the agglomeration has not been initialized. It's likely that you forgot to distribute the DoFs, i.e. you may want to check if a call to `initialize_hp_structure()` has been done."));
    for (const auto &cell :
         agglo_dh.active_cell_iterators() |
           IteratorFilters::ActiveFEIndexEqualTo(CellAgglomerationType::master))
      {
        setup_master_neighbor_connectivity(cell);
      }
  }

  friend class internal::AgglomerationHandlerImplementation<dim, spacedim>;
};



template <int dim, int spacedim>
AgglomerationIterator<dim, spacedim>
AgglomerationHandler<dim, spacedim>::begin() const
{
  Assert(n_agglomerations > 0,
         ExcMessage("No agglomeration has been performed."));
  return {*master_cells_container.begin(), this};
}



template <int dim, int spacedim>
AgglomerationIterator<dim, spacedim>
AgglomerationHandler<dim, spacedim>::begin()
{
  Assert(n_agglomerations > 0,
         ExcMessage("No agglomeration has been performed."));
  return {*master_cells_container.begin(), this};
}



template <int dim, int spacedim>
AgglomerationIterator<dim, spacedim>
AgglomerationHandler<dim, spacedim>::end() const
{
  Assert(n_agglomerations > 0,
         ExcMessage("No agglomeration has been performed."));
  return {*master_cells_container.end(), this};
}



template <int dim, int spacedim>
AgglomerationIterator<dim, spacedim>
AgglomerationHandler<dim, spacedim>::end()
{
  Assert(n_agglomerations > 0,
         ExcMessage("No agglomeration has been performed."));
  return {*master_cells_container.end(), this};
}



template <int dim, int spacedim>
AgglomerationIterator<dim, spacedim>
AgglomerationHandler<dim, spacedim>::last()
{
  Assert(n_agglomerations > 0,
         ExcMessage("No agglomeration has been performed."));
  return {master_cells_container.back(), this};
}



template <int dim, int spacedim>
IteratorRange<
  typename AgglomerationHandler<dim, spacedim>::agglomeration_iterator>
AgglomerationHandler<dim, spacedim>::polytope_iterators() const
{
  return IteratorRange<
    typename AgglomerationHandler<dim, spacedim>::agglomeration_iterator>(
    begin(), end());
}



namespace dealii
{
  namespace internal
  {
    template <int dim, int spacedim>
    class AgglomerationHandlerImplementation
    {
    public:
      static const FEValuesBase<dim, spacedim> &
      reinit_master(
        const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
        const unsigned int face_index,
        std::unique_ptr<NonMatching::FEImmersedSurfaceValues<spacedim>>
          &                                        agglo_isv_ptr,
        const AgglomerationHandler<dim, spacedim> &handler)
      {
        Assert(handler.is_master_cell(cell),
               ExcMessage("This cell must be a master one."));
        const auto &info_neighbors =
          handler.master_neighbors[{cell, face_index}];
        const auto &local_face_idx   = std::get<0>(info_neighbors);
        const auto &neighboring_cell = std::get<3>(info_neighbors);

        if (neighboring_cell.state() == IteratorState::valid)
          {
            handler.no_face_values->reinit(neighboring_cell, local_face_idx);
            auto q_points = handler.no_face_values->get_quadrature_points();

            const auto &JxWs    = handler.no_face_values->get_JxW_values();
            const auto &normals = handler.no_face_values->get_normal_vectors();

            const auto &bbox =
              handler
                .bboxes[handler.master2polygon.at(cell->active_cell_index())];
            const double                 bbox_measure = bbox.volume();
            std::vector<Point<spacedim>> unit_q_points;

            std::transform(q_points.begin(),
                           q_points.end(),
                           std::back_inserter(unit_q_points),
                           [&](const Point<spacedim> &p) {
                             return bbox.real_to_unit(p);
                           });

            // Weights must be scaled with det(J)*|J^-t n| for each quadrature
            // point. Use the fact that we are using a BBox, so the jacobi
            // entries are the side_length in each direction. Unfortunately,
            // normal vectors will be scaled internally by deal.II by using a
            // covariant transformation. In order not to change the normals, we
            // multiply by the correct factors in order to obtain the original
            // normal after the call to `reinit(cell)`.
            std::vector<double>         scale_factors(q_points.size());
            std::vector<double>         scaled_weights(q_points.size());
            std::vector<Tensor<1, dim>> scaled_normals(q_points.size());

            for (unsigned int q = 0; q < q_points.size(); ++q)
              {
                for (unsigned int direction = 0; direction < spacedim;
                     ++direction)
                  scaled_normals[q][direction] =
                    normals[q][direction] * (bbox.side_length(direction));

                scaled_weights[q] =
                  (JxWs[q] * scaled_normals[q].norm()) / bbox_measure;
                scaled_normals[q] /= scaled_normals[q].norm();
              }

            NonMatching::ImmersedSurfaceQuadrature<dim, spacedim> surface_quad(
              unit_q_points, scaled_weights, scaled_normals);

            agglo_isv_ptr =
              std::make_unique<NonMatching::FEImmersedSurfaceValues<spacedim>>(
                *(handler.euler_mapping),
                *(handler.fe),
                surface_quad,
                handler.agglomeration_face_flags);

            agglo_isv_ptr->reinit(cell);

            return *agglo_isv_ptr;
          }
        else
          {
            // Then it's a boundary face of an agglomeration living on the
            // boundary of the tria. You need to return an FEFaceValues on the
            // boundary face of a boundary cell.
            handler.no_face_values->reinit(neighboring_cell, local_face_idx);

            // TODO: check if *mapping or *euler_mapping
            handler.standard_scratch_face_bdary = std::make_unique<
              typename AgglomerationHandler<dim, spacedim>::ScratchData>(
              *(handler.euler_mapping),
              handler.fe_collection[2],
              handler.agglomeration_quad,
              handler.agglomeration_flags,
              handler.agglomeration_face_quad,
              handler.agglomeration_face_flags);

            return handler.standard_scratch_face_bdary->reinit(cell);
          }
      }
    };
  } // namespace internal
} // namespace dealii



#endif