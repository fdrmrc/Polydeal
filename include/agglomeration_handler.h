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

#include <deal.II/base/bounding_box_data_out.h>
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
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <deal.II/meshworker/scratch_data.h>

#include <deal.II/non_matching/fe_immersed_values.h>
#include <deal.II/non_matching/immersed_surface_quadrature.h>

#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <utility>

using namespace dealii;

/**
 *
 * Assumption: each cell may have only one master cell
 */
template <int dim, int spacedim = dim>
class AgglomerationHandler : public Subscriptor
{
  using ScratchData = MeshWorker::ScratchData<dim, spacedim>;

  // Internal type used to associate agglomerated cells with neighbors. In
  // particular, we store for each cell:
  // - local face index seen by the agglomerated cell
  // - an iterator to the neighboring cell
  // - local face index seen from the neighbor
  // In case the neighbor is a cell of the agglomeration:
  // - iterator is pointing to the master cell of the neighobring agglomeration
  // - face index from outside is set to numbers::invalid_unsigned_int
  // The reason why we use a **set** of tuples is that some cells will be seen
  // and checked multiple times.
  using NeighborsInfos = std::set<std::tuple<
    unsigned int,
    const typename Triangulation<dim, spacedim>::active_cell_iterator,
    unsigned int>>;

  using CellAndFace =
    std::pair<const typename Triangulation<dim, spacedim>::active_cell_iterator,
              unsigned int>;

  using MasterNeighborInfo = std::vector<std::tuple<
    unsigned int,
    const typename Triangulation<dim, spacedim>::active_cell_iterator,
    unsigned int>>;

public:
  enum AggloIndex
  {
    master   = 0,
    slave    = 1,
    standard = 2
  };

  static inline unsigned int n_agglomerations = 0; // only C++17 feature

  /**
   * DoFHandler for the agglomerated space
   *
   */
  DoFHandler<dim, spacedim> agglo_dh;

  explicit AgglomerationHandler(
    const GridTools::Cache<dim, spacedim> &cached_tria,
    const FE_DGQ<dim, spacedim>           &fe_space = FE_DGQ<dim, spacedim>(1));

  ~AgglomerationHandler()
  {
    // disconnect the signal
    tria_listener.disconnect();
  }

  /**
   * Set the proper flags for the FEValues object on the agglomerated space.
   *
   */
  inline void
  set_agglomeration_flags(const UpdateFlags &flags)
  {
    agglomeration_flags = flags;
  }

  inline void
  set_quadrature_degree(const unsigned int degree)
  {
    agglomeration_quadrature_degree = degree;
  }

  /**
   * Set active fe indices on each cell, and store internally the objects used
   * to initialize a hp::FEValues.
   */
  void
  initialize_hp_structure();

  void
  create_agglomeration_sparsity_pattern(SparsityPattern &sparsity_pattern);

  /**
   * Store internally that the given cells are agglomerated. The convenction we
   * take is the following:
   * -2: default value, standard deal.II cell
   * -1: cell is a master cell
   *
   * @note Cells are assumed to be adjacent one to each other, and no check
   * about this is done. @todo
   */

  void
  agglomerate_cells(const std::vector<
                    typename Triangulation<dim, spacedim>::active_cell_iterator>
                      &vec_of_cells);
  /**
   * Get cells agglomerated with the given cell iterator. Return an empty vector
   * if there are no cells agglomerated with it.
   */
  std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>
  get_agglomerated_cells(
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
    const;

  /**
   * Get the agglomerated connectivity object
   */
  inline friend std::map<
    const typename Triangulation<dim, spacedim>::active_cell_iterator,
    NeighborsInfos>
  get_agglomerated_connectivity(const AgglomerationHandler<dim, spacedim> &ah)
  {
    return ah.neighbor_connectivity;
  }

  /**
   * Display the indices of the vector identifying which cell is agglomerated
   * with which master.
   */
  friend void
  print_agglomeration(std::ostream                              &os,
                      const AgglomerationHandler<dim, spacedim> &ah)
  {
    for (const auto &cell : ah.euler_dh.active_cell_iterators())
      os << "Cell with index: " << cell->active_cell_index()
         << " has associated value: "
         << ah.master_slave_relationships[cell->active_cell_index()]
         << std::endl;
  }

  inline friend std::vector<BoundingBox<spacedim>>
  get_bboxes(const AgglomerationHandler<dim, spacedim> &ah)
  {
    return ah.bboxes;
  }

  /**
   * Return, for a cell the number of agglomerated faces. If it's a standard
   * cell, the result is 0.
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
        if (neighboring_cell.state() == IteratorState::valid &&
            !are_cells_agglomerated(cell, neighboring_cell))
          {
            ++n_neighbors;
          }
      }
    return n_neighbors;
  }


  /**
   * Return, for each cell belonging to the same agglomeration, the number of
   * agglomerated faces.
   */
  unsigned int
  n_agglomerated_faces_per_agglomeration(
    const typename Triangulation<dim, spacedim>::active_cell_iterator
      &master_cell) const
  {
    Assert(master_slave_relationships[master_cell->active_cell_index()] == -1,
           ExcMessage("You should pass a master cell."));
    auto agglomeration = get_slaves_of_idx(master_cell->active_cell_index());
    agglomeration.push_back(master_cell);
    unsigned int n_neighbors = 0;
    for (const auto &cell : agglomeration)
      {
        n_neighbors += n_agglomerated_faces_per_cell(cell);
      }
    return n_neighbors;
  }

  /**
   * Return, for each cell belonging to the same agllomeration, a cell belonging
   * to the neighbor agglomeration
   */
  typename Triangulation<dim, spacedim>::active_cell_iterator
  agglomerated_neighbor(
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell,
    const unsigned int agglomerated_face_number) const;

  /**
   * Return, for each cell belonging to the same agllomeration, the index of the
   * agglomerated face of the neighbor
   */
  unsigned int
  neighbor_of_agglomerated_neighbor(
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell,
    const unsigned int agglomerated_face_number) const;

  /**
   * Set the up neighbors info object. This function takes a vector of
   * agglomerations. Each agglomeration is described by a vector of cells.
   * @todo Document better
   */
  void
  setup_neighbors_info(
    const std::vector<
      std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>>
      &agglomerations)
  {
    Assert(agglomerations.size() > 0,
           ExcInternalError("This method is supposed to be called after the "
                            "setup of the agglomeration."));

    for (const auto &agglo : agglomerations)
      setup_neighbors_of_agglomeration(agglo);

    Assert(neighbor_connectivity.size() > 0,
           ExcInternalError("The connectivity should not be empty after the "
                            "execution of this function."));
  }

  /**
   * The argument here should be a vector storing the agglomerated cells.
   */
  void
  setup_neighbors_of_agglomeration(
    const std::vector<
      typename Triangulation<dim, spacedim>::active_cell_iterator>
      &vec_of_cells)
  {
    Assert(vec_of_cells.size() > 0, ExcMessage("The given vector is empty."));
    const bool all_valid_cells = std::all_of(
      vec_of_cells.begin(),
      vec_of_cells.end(),
      [&](const typename Triangulation<dim, spacedim>::active_cell_iterator
            &cell) { return !cell->has_children(); });
    Assert(all_valid_cells,
           ExcMessage(
             "Some iterators are not valid. You probably called this function "
             "with iterators to a Triangulation that has been refined without "
             "updating them."));

    for (const auto &cell : vec_of_cells)
      {
        for (const auto &f : cell->face_indices())
          {
            const auto &neighboring_cell = cell->neighbor(f);
            // Check if cell is not on the boundary and if it's not agglomerated
            // with the neighbor If so, it's a neighbor of the present
            // agglomeration
            if (neighboring_cell.state() ==
                  IteratorState::
                    valid && // @todo Handle the case where the cell
                             // is on the boundary
                !are_cells_agglomerated(cell, neighboring_cell))
              {
                const auto nof = cell->neighbor_of_neighbor(f);
                if (is_slave_cell(neighboring_cell))
                  {
                    neighbor_connectivity[cell].insert(
                      {f,
                       master_slave_relationships_iterators
                         [neighboring_cell->active_cell_index()],
                       numbers::invalid_unsigned_int});
                  }
                else
                  {
                    neighbor_connectivity[cell].insert(
                      {f, neighboring_cell, nof});
                  }
              }
          }
      }
  }


  void
  setup_master_neighbor_connectivity(
    const typename Triangulation<dim, spacedim>::active_cell_iterator
      &master_cell)
  {
    Assert(master_slave_relationships[master_cell->active_cell_index()] == -1,
           ExcMessage("The present cell is not a master one."));
    auto agglomeration = get_slaves_of_idx(master_cell->active_cell_index());
    agglomeration.push_back(master_cell);
    unsigned int n_agglo_faces = 0;
    for (const auto &cell : agglomeration)
      {
        for (const auto f : cell->face_indices())
          {
            const auto &neighboring_cell = cell->neighbor(f);
            // Check if cell is not on the boundary and if it's not agglomerated
            // with the neighbor If so, it's a neighbor of the present
            // agglomeration
            if (neighboring_cell.state() ==
                  IteratorState::
                    valid && // @todo Handle the case where the cell
                             // is on the boundary
                !are_cells_agglomerated(cell, neighboring_cell))
              {
                // a new face of the agglomeration has been discovered.
                const auto &cell_and_face =
                  CellAndFace(master_cell, n_agglo_faces);
                const auto nof = cell->neighbor_of_neighbor(f);
                if (is_slave_cell(neighboring_cell))
                  master_neighbors[cell_and_face].emplace_back(
                    f,
                    master_slave_relationships_iterators
                      [neighboring_cell->active_cell_index()],
                    nof);
                else
                  master_neighbors[cell_and_face].emplace_back(f,
                                                               neighboring_cell,
                                                               nof);
                ++n_agglo_faces;
              }
          }
      }
  }



  /**
   * Where the magic happens!
   */
  const FEValues<dim, spacedim> &
  reinit(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell) const;

  /**
   * agglomeration_face_number \in [0,n_agglomerated_faces)
   */
  /*std::pair<const FEValuesBase<dim, spacedim> &,
            NonMatching::ImmersedSurfaceQuadrature<dim, spacedim>>*/
  const NonMatching::FEImmersedSurfaceValues<dim> &
  reinit(const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
         const unsigned int agglomeration_face_number) const;

  /**
   * Return the agglomerated quadrature for the given agglomeration. This
   * amounts to loop over all cells in an agglomeration and collecting together
   * all the rules.
   */
  Quadrature<dim>
  agglomerated_quadrature(
    const std::vector<
      typename Triangulation<dim, spacedim>::active_cell_iterator> &cells,
    const Quadrature<dim> &quadrature_type) const;

  /**
   * Return the agglomerated face quadrature for the given agglomeration face.
   */
  NonMatching::ImmersedSurfaceQuadrature<dim, spacedim>
  agglomerated_face_quadrature(
    typename Triangulation<dim, spacedim>::active_cell_iterator &cell,
    unsigned int agglomerated_face_number) const;

  void
  test_setup_euler_mapping()
  {
    euler_mapping =
      std::make_unique<MappingFEField<dim, spacedim>>(euler_dh, euler_vector);

    ScratchData scratch(*euler_mapping,
                        agglo_dh.get_fe(),
                        QGauss<dim>(3),
                        update_values | update_JxW_values |
                          update_quadrature_points);

    for (const auto &cell :
         agglo_dh.active_cell_iterators() |
           IteratorFilters::ActiveFEIndexEqualTo(AggloIndex::master))
      {
        std::cout << agglo_dh.get_fe().dofs_per_cell << std::endl;
        std::cout << "BBox has measure: "
                  << euler_mapping->get_bounding_box(cell).volume()
                  << std::endl;
        const auto &fev = scratch.reinit(cell);
        double      sum = 0.;
        for (const auto &q : fev.get_JxW_values())
          {
            sum += q;
          }
        std::cout << "Sum is: " << sum << std::endl;
      }

    std::ofstream           ofile("boxes.vtu");
    BoundingBoxDataOut<dim> data_out;
    data_out.build_patches(bboxes);
    data_out.write_vtu(ofile);
  }

  /**
   * Find (if any) the cells that have the given master index. Note that idx is
   * signed as it can be equal to -1 (meaning that the cell is a master one) or
   * -2.
   */
  std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>
  get_slaves_of_idx(const int idx) const;

  mutable std::map<CellAndFace, MasterNeighborInfo> master_neighbors;


private:
  std::vector<long int> master_slave_relationships;

  // Same as the one above, but storing cell iterators rather than indices.
  std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>
    master_slave_relationships_iterators;

  /**
   * bboxes[idx] = BBOx associated to the agglomeration with master cell indexed
   * by idx. Othwerwise ddefault BBox
   *
   */
  std::vector<BoundingBox<spacedim>> bboxes; //@todo: use map also for BBOxes

  SmartPointer<const Triangulation<dim, spacedim>> tria;

  SmartPointer<const Mapping<dim, spacedim>> mapping;

  std::unique_ptr<FESystem<dim, spacedim>> euler_fe;

  std::unique_ptr<GridTools::Cache<dim, spacedim>> cached_tria;

  // The FE_DGQ space we have on each cell
  std::unique_ptr<FE_DGQ<dim, spacedim>> fe;

  hp::FECollection<dim, spacedim> fe_collection;

  mutable std::map<
    const typename Triangulation<dim, spacedim>::active_cell_iterator,
    NeighborsInfos>
    neighbor_connectivity;


  /**
   * DoFHandler for the physical space
   *
   */
  DoFHandler<dim, spacedim> euler_dh;

  /**
   * Eulerian vector describing the new cells obtained by the bounding boxes
   */
  Vector<double> euler_vector;

  std::unique_ptr<MappingFEField<dim, spacedim> /*, Vector<double>*/>
    euler_mapping;

  /**
   * Fill this up in initialize_hp_structure(dh, ...).
   *
   * Use this in reinit(cell) for standard (non-agglomerated) cells, and return
   * the result of scratch.reinit(cell) for cells
   */
  mutable std::unique_ptr<ScratchData> standard_scratch;

  /**
   * Fill this up in reinit(cell), for agglomerated cells, using the custom
   * quadrature, and return the result of
   * scratch.reinit(cell);
   */
  mutable std::unique_ptr<ScratchData> agglomerated_scratch;

  mutable std::unique_ptr<NonMatching::FEImmersedSurfaceValues<spacedim>>
    agglomerated_face_values;

  boost::signals2::connection tria_listener;

  UpdateFlags agglomeration_flags = update_default;

  UpdateFlags agglomeration_face_flags = update_quadrature_points |
                                         update_normal_vectors | update_values |
                                         update_JxW_values;

  unsigned int agglomeration_quadrature_degree = 1;

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
    euler_fe = std::make_unique<FESystem<dim, spacedim>>(*fe, spacedim);
    euler_dh.reinit(*tria);
    euler_dh.distribute_dofs(*euler_fe);
    euler_vector.reinit(euler_dh.n_dofs());

    master_slave_relationships.resize(tria->n_active_cells(), -2);
    master_slave_relationships_iterators.resize(tria->n_active_cells(), {});
    if (n_agglomerations > 0)
      std::fill(master_slave_relationships.begin(),
                master_slave_relationships.end(),
                -2); // identify all the tria with standard deal.II cells.

    neighbor_connectivity.clear();
    bboxes.resize(tria->n_active_cells());
    // n_agglomerations = 0;

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
   * Helper function to determine whether or not a cell is a master or a slave
   */
  inline bool
  is_master_cell(
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
    const
  {
    return master_slave_relationships[cell->active_cell_index()] == -1;
  }

  /**
   * Helper function to determine if the given cell is a standard deal.II cell,
   * that is: not master, nor slave.
   *
   */
  inline bool
  is_standard_cell(
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
    const
  {
    return master_slave_relationships[cell->active_cell_index()] == -2;
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
   * Helper function to determine whether or not a cell is a slave cell,
   without any information about his parents.
   */
  inline bool
  is_slave_cell(
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
    const
  {
    return master_slave_relationships[cell->active_cell_index()] >= 0;
  }

  /**
   * Construct bounding boxes for an agglomeration described by a sequence of
   * cells. This fills also the euler vector
   */
  void
  create_bounding_box(
    const std::vector<
      typename Triangulation<dim, spacedim>::active_cell_iterator>
                      &vec_of_cells,
    const unsigned int master_idx)
  {
    Assert(n_agglomerations > 0,
           ExcMessage("No agglomeration has been performed."));
    Assert(dim == 2 && spacedim == 2,
           ExcNotImplemented()); //@todo #3 Not working in 3D

    std::vector<types::global_dof_index> dof_indices(euler_fe->dofs_per_cell);
    std::vector<Point<spacedim>>         pts; // store all the vertices
    for (const auto &cell : vec_of_cells)
      {
        typename DoFHandler<dim, spacedim>::cell_iterator cell_dh(*cell,
                                                                  &euler_dh);
        cell_dh->get_dof_indices(dof_indices);
        for (const auto i : cell->vertex_indices())
          pts.push_back(cell->vertex(i));
      }

    bboxes[master_idx] = BoundingBox<spacedim>(pts);

    // @todo: be more general than 2D...
    const auto &p0 = bboxes[master_idx].get_boundary_points().first;
    const auto &p1 = bboxes[master_idx].get_boundary_points().second;

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

  /**
   * Returns true if the two given cells are agglomerated together.
   */
  inline bool
  are_cells_agglomerated(
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell,
    const typename Triangulation<dim, spacedim>::active_cell_iterator
      &other_cell) const
  {
    // check if they refer to same master, OR if it's a master with its slave.
    return master_slave_relationships[cell->active_cell_index()] ==
             master_slave_relationships[other_cell->active_cell_index()] ||
           master_slave_relationships[cell->active_cell_index()] ==
             other_cell->active_cell_index() ||
           master_slave_relationships[other_cell->active_cell_index()] ==
             cell->active_cell_index();
  }
};

#endif
