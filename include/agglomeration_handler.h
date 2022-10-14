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

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <deal.II/meshworker/scratch_data.h>

#include <deal.II/non_matching/immersed_surface_quadrature.h>

#include <utility>

using namespace dealii;

/**
 *
 * Assumption: each cell may have only one master cell, that means it's not
 * possible that a cell has two masters cells.
 */
template <int dim, int spacedim = dim>
class AgglomerationHandler : public Subscriptor
{
  using ScratchData = MeshWorker::ScratchData<dim, spacedim>;

  // Internal type used to associate agglomerated cells with neighbors. In
  // particular, we store for each cell the index of the neighboring cell (hence
  // the first type of the pair is a cell iterator) and the *local* index
  // (classical face_no) in the deal.II lingo of the face shared with that cell.
  // The reason why we use a **set** of pairs is that some cells will be seen
  // and checked multiple times.
  using NeighborsInfos = std::set<
    std::pair<const typename Triangulation<dim, spacedim>::active_cell_iterator,
              unsigned int>>;

public:
  static inline unsigned int n_agglomerated_cells = 0; // only C++17 feature

  explicit AgglomerationHandler(const Triangulation<dim, spacedim> &tria,
                                unsigned int mapping_degree = 0);

  ~AgglomerationHandler()
  {
    // disconnect the signal
    // initialize_listener.disconnect();
  }

  /**
   * Set active fe indices on each cell, and store internally the objects used
   * to initialize a hp::FEValues.
   */
  void
  initialize_hp_structure(DoFHandler<dim, spacedim> &dh /*, ...*/);

  /**
   * Store internally that the given cells are agglomerated. The convenction we
   * take is the following: -2: default value, standard deal.II cell -1: a cell
   * is a master cell
   *
   * @note Cells are assumed to be adjacent one to each other, and no check about this is done. @todo
   */
  void
  agglomerate_cells(const std::vector<
                    typename Triangulation<dim, spacedim>::active_cell_iterator>
                      &vec_of_cells);

  /**
   * Same as above, but deciding which index is the master one
   */
  void
  agglomerate_cells(
    const std::vector<
      typename Triangulation<dim, spacedim>::active_cell_iterator>
                      &vec_of_cells,
    const unsigned int local_master_idx);

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
   * Return, for each cell belonging to the same agglomeration, the number of
   * agglomerated faces.
   */
  unsigned int
  n_agglomerated_faces(
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
  // const
  {
    unsigned int n_neighbors = 0.;
    for (const auto &f : cell->face_indices())
      {
        std::cout << f << std::endl;
        const auto neighboring_cell = cell->neighbor(f);
        if (neighboring_cell.state() != IteratorState::invalid &&
            are_cells_agglomerated(cell, neighboring_cell))
          ++n_neighbors;
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
   * Set the up neighbors info object
   */
  void
  setup_neighbors_info()
  {
    Assert(
      neighbor_connectivity.size() > 0,
      ExcInternalError(
        "This method is supposed to be called after the setup of the agglomeration."));



    Assert(
      neighbor_connectivity.size() > 0,
      ExcInternalError(
        "The connectivity should not be empty after the execution of this function."));
  }

  /**
   * The argument here should be a vector storing the agglomerated cells.
   *
   *
   *
   */
  void
  setup_neighbors_of_agglomeration(
    const std::vector<
      typename Triangulation<dim, spacedim>::active_cell_iterator>
      &vec_of_cells)
  {
    Assert(vec_of_cells.size() > 0, ExcMessage("The given vector is empty."));
    for (const auto &cell : vec_of_cells)
      {
        for (const auto &f : cell->face_indices())
          {
            const auto &neighboring_cell = cell->neighbor(f);

            if (neighboring_cell.state() == IteratorState::valid &&
                !are_cells_agglomerated(cell, neighboring_cell))
              {
                std::cout << "Cell idx: " << cell->active_cell_index()
                          << " Face index: " << f << std::endl;
                neighbor_connectivity[cell].insert({neighboring_cell, f});
              }
          }
      }
  }

  /**
   * Where the magic happens!
   */
  const FEValuesBase<dim, spacedim> &
  reinit(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell) const;

  /**
   * agglomeration_face_number \in [0,n_agglomerated_faces)
   */
  /*(fe_values, immersed_surface_quadrature)*/
  std::pair<const FEValuesBase<dim, spacedim> &,
            NonMatching::ImmersedSurfaceQuadrature<dim, spacedim>>
  reinit(const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
         const unsigned int agglomeration_face_number) const;

  /**
   * Return the agglomerated quadrature for the given agglomeration.
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



private:
  std::vector<long int> master_slave_relationships;

  /**
   * bboxes[idx] = BBOx associated to the agglomeration with master cell indexed
   * by idx. Othwerwise ddefault BBox
   *
   */
  std::vector<BoundingBox<spacedim>> bboxes;

  SmartPointer<const Triangulation<dim, spacedim>> tria;

  std::unique_ptr<FESystem<dim, spacedim>> euler_fe;

  std::map<const typename Triangulation<dim, spacedim>::active_cell_iterator,
           NeighborsInfos>
    neighbor_connectivity;

  NeighborsInfos face_and_neighbor;

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

  boost::signals2::connection initialize_listener;

  /**
   * Make sure we throw unless initialize() is called again.
   */
  void
  invalidate_data()
  {
    // initialize_listener.disconnect();
  }

  /**
   * Helper function to determine whether or not a cell is a master or a slave
   */
  inline bool
  is_master_cell(
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
  {
    return master_slave_relationships[cell->active_cell_index()] == -1;
  }

  /**
   * Helper function to determine whether or not a cell is a slave cell.
   * Instead of returning a boolean, it gives the index of the master cell. If
   * it's a master cell, then the it returns -1, by construction.
   */
  inline unsigned int
  is_slave_cell_of(
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
  {
    return master_slave_relationships[cell->active_cell_index()] != -1;
  }

  /**
   * Find (if any) the cells that have the given master index. Note that idx is
   * signed as it can be equal to -1, meaning that the cell is a master one.
   */
  std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>
  get_slaves_of_idx(const int idx) const;


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
    Assert(n_agglomerated_cells > 0,
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
