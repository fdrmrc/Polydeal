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


#include <deal.II/base/function.h>
#include <deal.II/base/subscriptor.h>

#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/non_matching/fe_values.h>
#include <deal.II/non_matching/mesh_classifier.h>

#include <deal.II/numerics/vector_tools_interpolate.templates.h>

#include <vector>
using namespace dealii;

enum class CutClassification
{
  good,
  small_inside,
  small_outisde
};

template <int dim, int spacedim = dim>
class Agglomerator : public Subscriptor
{
public:
  Agglomerator() = default;

  Agglomerator(const GridTools::Cache<dim, spacedim> &cache_tria,
               const Function<dim>                   &level_set,
               const Quadrature<1>                   &quad_1D = QGauss<1>(3));

  ~Agglomerator() = default;

  inline std::vector<
    std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>>
  get_agglomerates() const
  {
    return agglomerates;
  };


private:
  std::unique_ptr<GridTools::Cache<dim, spacedim>> cached_tria;

  /**
   * 1D base Quadrature rule used internally by NonMatching::FEValues to
   * construct a tensor product rule.
   *
   */
  std::unique_ptr<Quadrature<1>> quadrature_1D;

  typename NonMatching::MeshClassifier<dim> mesh_classifier;

  /**
   * Continuous level set function that describes the geometry of the domain.
   *
   */
  std::unique_ptr<Function<dim>> level_set_function;

  /**
   * Discrete level set that describes the geometry of the domain.
   *
   */
  Vector<double> level_set_vector;

  /**
   * DoFs for the discrete level set function.
   */
  DoFHandler<dim> level_set_dof_handler;

  const FE_Q<dim> level_set_fe;

  std::vector<
    std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>>
    agglomerates;

  /**
   * Identify which cells of the mesh are cut.
   */
  void
  identify_cut_cells();

  /**
   * Identify in which cells of the triangulation small cuts occurs.
   */
  void
  classify_cut_cells();

  /**
   * Detect if intersected cell have small cut or not. If it has, find in
   * which part of the domain the small subcell lives.
   */
  CutClassification
  classify_cell(
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cut_cell,
    const double subcell_measure,
    const double tol = .3);
};
