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

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/subscriptor.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q.h>

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

using namespace dealii;

// Forward declarations
template <int dim, int spacedim>
class AgglomerationHandler;

namespace dealii
{
  namespace internal
  {
    /**
     * Helper class to reinit finite element spaces on polytopal cells.
     */
    template <int, int>
    class AgglomerationHandlerImplementation;
  } // namespace internal
} // namespace dealii


/**
 * Helper class for the storage of connectivity information of the polytopal
 * grid.
 */
namespace dealii
{
  namespace internal
  {
    template <int dim, int spacedim>
    class PolytopeCache
    {
    public:
      /**
       * Default constructor.
       */
      PolytopeCache() = default;

      /**
       * Destructor. It simply calls clear() for all of its members.
       */
      ~PolytopeCache() = default;

      void
      clear()
      {
        // clear all the members
        cell_face_at_boundary.clear();
        interface.clear();
        visited_cell_and_faces.clear();
      }

      /**
       * Standard std::set for recording the standard cells and faces (in the
       * deal.II lingo) that have been already visited. The first argument of
       * the pair identifies the global index of a deal.II cell, while the
       * second its local face number.
       *
       */
      mutable std::set<std::pair<types::global_cell_index, unsigned int>>
        visited_cell_and_faces;

      /**
       * Map that associate the pair of (polytopal index, polytopal face) to
       * (b,cell). The latter pair indicates whether or not the present face is
       * on boundary. If it's on the boundary, then b is true and cell is an
       * invalid cell iterator. Otherwise, b is false and cell points to the
       * neighboring polytopal cell.
       *
       */
      mutable std::map<
        std::pair<types::global_cell_index, unsigned int>,
        std::pair<bool,
                  typename Triangulation<dim, spacedim>::active_cell_iterator>>
        cell_face_at_boundary;

      /**
       * Standard std::map that associated to a pair of neighboring polytopic
       * cells (current_polytope, neighboring_polytope) a sequence of
       * ({deal_cell,deal_face_index}) which is meant to describe their
       * interface.
       * Indeed, the pair is identified by the two polytopic global indices,
       * while the interface is described by a std::vector of deal.II cells and
       * faces.
       *
       */
      mutable std::map<
        std::pair<types::global_cell_index, types::global_cell_index>,
        std::vector<
          std::pair<typename Triangulation<dim, spacedim>::active_cell_iterator,
                    unsigned int>>>
        interface;
    };
  } // namespace internal
} // namespace dealii


/**
 * #TODO: Documentation.
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
    const UpdateFlags &        face_flags      = UpdateFlags::update_default);

  /**
   * Given a Triangulation with some agglomerated cells, create the sparsity
   * pattern corresponding to a Discontinuous Galerkin discretization where the
   * agglomerated cells are seen as one **unique** cell, with only the DoFs
   * associated to the master cell of the agglomeration.
   */
  template <typename Number = double>
  void
  create_agglomeration_sparsity_pattern(
    SparsityPattern &               sparsity_pattern,
    const AffineConstraints<Number> constraints = AffineConstraints<Number>(),
    const bool                      keep_constrained_dofs = true,
    const types::subdomain_id subdomain_id = numbers::invalid_subdomain_id);

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
  define_agglomerate(const AgglomerationContainer &cells);

  inline decltype(auto)
  get_interface() const;

  inline const std::vector<long int> &
  get_relationships() const;

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
      &master_cell) const;

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
   * Return a constant reference to the DoFHandler underlying the
   * agglomeration. It knows which cell have been agglomerated, and which FE
   * spaces are present on each cell of the triangulation.
   */
  inline const DoFHandler<dim, spacedim> &
  get_dof_handler() const;

  /**
   * Returns the number of agglomerate cells in the grid.
   */
  unsigned int
  n_agglomerates() const;

  /**
   * Return the number of agglomerated faces for a generic deal.II cell. If it's
   * a standard cell, the result is 0.
   */
  unsigned int
  n_agglomerated_faces_per_cell(
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
    const;

  /**
   * Return, for a cell, the number of faces. In case the cell is a standard
   * cell, then the number of faces is the classical one. If it's a master cell,
   * then it returns the number of faces of the agglomeration identified by the
   * master cell itself.
   */
  unsigned int
  n_faces(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell) const;

  /**
   * Construct a finite element space on the agglomeration.
   */
  const FEValues<dim, spacedim> &
  reinit(const AgglomerationIterator<dim, spacedim> &polytope) const;

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
   *
   * This function generalizes the behaviour of cell->face(f)->at_boundary()
   * in the case where f is an index out of the range [0,..., n_faces).
   * In practice, if you call this function with a standard deal.II cell, you
   * have precisely the same result as calling cell->face(f)->at_boundary().
   * Otherwise, if the cell is a master one, you have a boolean returning true
   * is that face for the agglomeration is on the boundary or not.
   */
  inline bool
  at_boundary(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
    const unsigned int                                              f) const;

  inline unsigned int
  n_dofs_per_cell() const noexcept;

  inline types::global_dof_index
  n_dofs() const noexcept;

  /**
   * Interpolate the solution defined on the agglomerates onto a classical
   * deal.II DoFHandler.
   * #TODO: place this in a helper function
   */
  void
  setup_output_interpolation_matrix();

  /**
   * Return the collection of vertices describing the boundary of the polytope
   * associated to the master cell `cell`. The return type is meant to describe
   * a sequence of edges (in 2D) or faces (in 3D).
   */
  inline const std::vector<typename Triangulation<dim>::active_face_iterator> &
  polytope_boundary(
    const typename Triangulation<dim>::active_cell_iterator &cell);


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

  std::unique_ptr<MappingFEField<dim, spacedim> /*, Vector<double>*/>
    euler_mapping;


private:
  /**
   * Initialize connectivity informations
   */
  void
  initialize_agglomeration_data(
    const std::unique_ptr<GridTools::Cache<dim, spacedim>> &cache_tria);

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
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell);

  /**
   * Construct bounding boxes for an agglomeration described by a sequence of
   * cells. This fills also the euler vector
   */
  void
  create_bounding_box(const AgglomerationContainer & polytope,
                      const types::global_cell_index master_idx);


  inline types::global_cell_index
  get_master_idx_of_cell(
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
    const;

  /**
   * Returns true if the two given cells are agglomerated together.
   */
  inline bool
  are_cells_agglomerated(
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell,
    const typename Triangulation<dim, spacedim>::active_cell_iterator
      &other_cell) const;

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
   * Find (if any) the cells that have the given master index. Note that `idx`
   * is as it can be equal to -1 (meaning that the cell is a master one).
   */
  inline const std::vector<
    typename Triangulation<dim, spacedim>::active_cell_iterator> &
  get_slaves_of_idx(types::global_cell_index idx) const;

  /**
   * Helper function to determine whether or not a cell is a master or a slave
   */
  template <typename CellIterator>
  inline bool
  is_master_cell(const CellIterator &cell) const;

  /**
   * Helper function to determine if the given cell is a standard deal.II cell,
   * that is: not master, nor slave.
   *
   */
  template <typename CellIterator>
  inline bool
  is_standard_cell(const CellIterator &cell) const;

  /**
   * Helper function to determine whether or not a cell is a slave cell, without
   * any information about his parents.
   */
  template <typename CellIterator>
  inline bool
  is_slave_cell(const CellIterator &cell) const;


  /**
   * Initialize all the necessary connectivity information for an
   * agglomeration. #TODO: loop over polytopes, avoid using master cells
   * explicitely.
   */
  void
  setup_connectivity_of_agglomeration();

  /**
   * Record the number of agglomerations on the grid.
   */
  unsigned int n_agglomerations;


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

  mutable std::vector<types::global_cell_index> number_of_agglomerated_faces;

  /**
   * Associate a master cell (hence, a given polytope) to its boundary faces.
   * The boundary is described through a vector of face iterators.
   *
   */
  mutable std::map<
    const typename Triangulation<dim, spacedim>::active_cell_iterator,
    std::vector<typename Triangulation<dim>::active_face_iterator>>
    polygon_boundary;


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

  const MPI_Comm communicator;

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

  friend class internal::AgglomerationHandlerImplementation<dim, spacedim>;

  internal::PolytopeCache<dim, spacedim> polytope_cache;

  /**
   * Bool that keeps track whether the mesh is composed also by standard deal.II
   * cells as (trivial) polytopes.
   */
  bool hybrid_mesh;
};



// ------------------------------ inline functions -------------------------
template <int dim, int spacedim>
inline decltype(auto)
AgglomerationHandler<dim, spacedim>::get_interface() const
{
  return polytope_cache.interface;
}



template <int dim, int spacedim>
inline const std::vector<long int> &
AgglomerationHandler<dim, spacedim>::get_relationships() const
{
  return master_slave_relationships;
}



template <int dim, int spacedim>
inline std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>
AgglomerationHandler<dim, spacedim>::get_agglomerate(
  const typename Triangulation<dim, spacedim>::active_cell_iterator
    &master_cell) const
{
  Assert(is_master_cell(master_cell), ExcInternalError());
  auto agglomeration = get_slaves_of_idx(master_cell->active_cell_index());
  agglomeration.push_back(master_cell);
  return agglomeration;
}



template <int dim, int spacedim>
inline const DoFHandler<dim, spacedim> &
AgglomerationHandler<dim, spacedim>::get_dof_handler() const
{
  return agglo_dh;
}



template <int dim, int spacedim>
inline const std::vector<
  typename Triangulation<dim, spacedim>::active_cell_iterator> &
AgglomerationHandler<dim, spacedim>::get_slaves_of_idx(
  types::global_cell_index idx) const
{
  return master2slaves.at(idx);
}



template <int dim, int spacedim>
template <typename CellIterator>
inline bool
AgglomerationHandler<dim, spacedim>::is_master_cell(
  const CellIterator &cell) const
{
  return master_slave_relationships[cell->active_cell_index()] == -1;
}



template <int dim, int spacedim>
template <typename CellIterator>
inline bool
AgglomerationHandler<dim, spacedim>::is_standard_cell(
  const CellIterator &cell) const
{
  return master_slave_relationships[cell->active_cell_index()] == -2;
}



/**
 * Helper function to determine whether or not a cell is a slave cell, without
 * any information about his parents.
 */
template <int dim, int spacedim>
template <typename CellIterator>
inline bool
AgglomerationHandler<dim, spacedim>::is_slave_cell(
  const CellIterator &cell) const
{
  return master_slave_relationships[cell->active_cell_index()] >= 0;
}



template <int dim, int spacedim>
inline bool
AgglomerationHandler<dim, spacedim>::at_boundary(
  const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
  const unsigned int face_index) const
{
  Assert(!is_slave_cell(cell),
         ExcMessage("This function should not be called for a slave cell."));

  return polytope_cache.cell_face_at_boundary
    .at({master2polygon.at(cell->active_cell_index()), face_index})
    .first;
}


template <int dim, int spacedim>
inline unsigned int
AgglomerationHandler<dim, spacedim>::n_dofs_per_cell() const noexcept
{
  return fe->n_dofs_per_cell();
}



template <int dim, int spacedim>
inline types::global_dof_index
AgglomerationHandler<dim, spacedim>::n_dofs() const noexcept
{
  return agglo_dh.n_dofs();
}



template <int dim, int spacedim>
inline const std::vector<typename Triangulation<dim>::active_face_iterator> &
AgglomerationHandler<dim, spacedim>::polytope_boundary(
  const typename Triangulation<dim>::active_cell_iterator &cell)
{
  return polygon_boundary[cell];
}



template <int dim, int spacedim>
inline typename Triangulation<dim, spacedim>::active_cell_iterator &
AgglomerationHandler<dim, spacedim>::is_slave_cell_of(
  const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
{
  return master_slave_relationships_iterators[cell->active_cell_index()];
}



template <int dim, int spacedim>
inline types::global_cell_index
AgglomerationHandler<dim, spacedim>::get_master_idx_of_cell(
  const typename Triangulation<dim, spacedim>::active_cell_iterator &cell) const
{
  auto idx = master_slave_relationships[cell->active_cell_index()];
  if ((idx == -1) || (idx == -2))
    return cell->active_cell_index();
  else
    return idx;
}



template <int dim, int spacedim>
inline bool
AgglomerationHandler<dim, spacedim>::are_cells_agglomerated(
  const typename Triangulation<dim, spacedim>::active_cell_iterator &cell,
  const typename Triangulation<dim, spacedim>::active_cell_iterator &other_cell)
  const
{
  return (get_master_idx_of_cell(cell) == get_master_idx_of_cell(other_cell));
}



template <int dim, int spacedim>
inline unsigned int
AgglomerationHandler<dim, spacedim>::n_agglomerates() const
{
  return n_agglomerations;
}



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



#endif