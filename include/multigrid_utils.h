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


#ifndef multigrid_utils_h
#define multigrid_utils_h


#include <deal.II/base/config.h>

#include <deal.II/lac/linear_operator.h>

#include <deal.II/multigrid/mg_base.h>

#include <agglomeration_handler.h>
#include <poly_utils.h>

namespace dealii
{
  namespace internal
  {
    template <typename VectorType>
    struct MatrixSelectorAgglomeration
    {
      using Sparsity = ::dealii::SparsityPattern;
      using Matrix   = ::dealii::SparseMatrix<typename VectorType::value_type>;
    };
  } // namespace internal
} // namespace dealii


/**
 * Struct storing information such as agglomerates and sub agglomerates.
 */
namespace dealii
{
  template <int dim>
  struct RtreeInfo
  {
    RtreeInfo() = delete;
    RtreeInfo(const std::vector<std::vector<unsigned int>> &crss_,
              const std::vector<
                std::vector<typename Triangulation<dim>::active_cell_iterator>>
                &agglomerates_)
      : crss(crss_)
      , agglomerates(agglomerates_)
    {}

    /**
     * CRS like data structure for every agglomerates.
     */
    const std::vector<std::vector<unsigned int>> &crss;

    /**
     * Agglomerates.
     */
    const std::vector<
      std::vector<typename Triangulation<dim>::active_cell_iterator>>
      &agglomerates;
  };
} // namespace dealii


namespace dealii
{
  /**
   * Coarse grid multigrid operator for an iterative solver.
   *
   * This class provides a wrapper for a deal.II iterative solver where the
   * action of the matrix is given through a LinearOperator.
   */
  template <typename VectorType,
            typename SolverType,
            typename PreconditionerType>
  class MGCoarseGridIterativeSolverLO : public MGCoarseGridBase<VectorType>
  {
  public:
    /**
     * Default constructor.
     */
    MGCoarseGridIterativeSolverLO();

    /**
     * Constructor. Only a reference to these objects is stored, so
     * their lifetime needs to exceed the usage in this class.
     */
    MGCoarseGridIterativeSolverLO(SolverType &                      solver,
                                  const LinearOperator<VectorType> &linear_op,
                                  const PreconditionerType &precondition);

    /**
     * Initialize with new data, see the corresponding constructor for more
     * details.
     */
    void
    initialize(SolverType &                      solver,
               const LinearOperator<VectorType> &linear_op,
               const PreconditionerType &        precondition);

    /**
     * Clear all pointers.
     */
    void
    clear();

    /**
     * Implementation of the abstract function. Calls the solve method of the
     * given solver with linear operator, vectors, and preconditioner.
     */
    virtual void
    operator()(const unsigned int level,
               VectorType &       dst,
               const VectorType & src) const override;

  private:
    /**
     * Reference to the solver.
     */
    SmartPointer<
      SolverType,
      MGCoarseGridIterativeSolverLO<VectorType, SolverType, PreconditionerType>>
      solver;

    /**
     * Reference to the linear operator.
     */
    std::unique_ptr<const LinearOperator<VectorType>> linear_op;

    /**
     * Reference to the preconditioner.
     */
    SmartPointer<
      const PreconditionerType,
      MGCoarseGridIterativeSolverLO<VectorType, SolverType, PreconditionerType>>
      preconditioner;
  };



  /* ------------------ Functions for MGCoarseGridIterativeSolverLO
   * (LinearOperator) ------------ */

  template <typename VectorType,
            typename SolverType,
            typename PreconditionerType>
  MGCoarseGridIterativeSolverLO<VectorType, SolverType, PreconditionerType>::
    MGCoarseGridIterativeSolverLO()
    : solver(0, typeid(*this).name())
    , linear_op(nullptr)
    , preconditioner(0, typeid(*this).name())
  {}



  template <typename VectorType,
            typename SolverType,
            typename PreconditionerType>
  MGCoarseGridIterativeSolverLO<VectorType, SolverType, PreconditionerType>::
    MGCoarseGridIterativeSolverLO(SolverType &                      solver,
                                  const LinearOperator<VectorType> &linear_op,
                                  const PreconditionerType &preconditioner)
    : solver(&solver, typeid(*this).name())
    , linear_op(std::make_unique<LinearOperator<VectorType>>(linear_op))
    , preconditioner(&preconditioner, typeid(*this).name())
  {}



  template <typename VectorType,
            typename SolverType,
            typename PreconditionerType>
  void
  MGCoarseGridIterativeSolverLO<VectorType, SolverType, PreconditionerType>::
    initialize(SolverType &                      solver_,
               const LinearOperator<VectorType> &linear_op_,
               const PreconditionerType &        preconditioner_)
  {
    solver = &solver_;
    linear_op.reset(&linear_op_);
    preconditioner = &preconditioner_;
  }



  template <typename VectorType,
            typename SolverType,
            typename PreconditionerType>
  void
  MGCoarseGridIterativeSolverLO<VectorType, SolverType, PreconditionerType>::
    clear()
  {
    solver         = 0;
    linear_op      = nullptr;
    preconditioner = 0;
  }



  namespace internal
  {
    namespace MGCoarseGridIterativeSolverLO
    {
      template <typename VectorType,
                typename SolverType,
                typename PreconditionerType,
                std::enable_if_t<
                  std::is_same_v<VectorType, typename SolverType::vector_type>,
                  VectorType> * = nullptr>
      void
      solve(SolverType &                      solver,
            const LinearOperator<VectorType> &linear_op,
            const PreconditionerType &        preconditioner,
            VectorType &                      dst,
            const VectorType &                src)
      {
        const auto &A_inv = inverse_operator(linear_op, solver, preconditioner);
        dst               = A_inv * src;
      }

      template <typename VectorType,
                typename SolverType,
                typename PreconditionerType,
                std::enable_if_t<
                  !std::is_same_v<VectorType, typename SolverType::vector_type>,
                  VectorType> * = nullptr>
      void
      solve(SolverType &                      solver,
            const LinearOperator<VectorType> &linear_op,
            const PreconditionerType &        preconditioner,
            VectorType &                      dst,
            const VectorType &                src)
      {
        typename SolverType::vector_type src_;
        typename SolverType::vector_type dst_;

        src_ = src;
        dst_ = dst;

        const auto &A_inv = inverse_operator(linear_op, solver, preconditioner);
        dst_              = A_inv * src_;

        dst = dst_;
      }
    } // namespace MGCoarseGridIterativeSolverLO
  }   // namespace internal



  template <typename VectorType,
            typename SolverType,
            typename PreconditionerType>
  void
  MGCoarseGridIterativeSolverLO<VectorType, SolverType, PreconditionerType>::
  operator()(const unsigned int /*level*/,
             VectorType &      dst,
             const VectorType &src) const
  {
    Assert(solver != nullptr, ExcNotInitialized());
    Assert(linear_op != nullptr, ExcNotInitialized());
    Assert(preconditioner != nullptr, ExcNotInitialized());

    dst = 0;
    internal::MGCoarseGridIterativeSolverLO::solve(
      *solver, *linear_op, *preconditioner, dst, src);
  }



  /**
   * Class implementing transfer between consecutive agglomerated levels.
   */
  template <int dim, typename VectorType>
  class MGTwoLevelTransferAgglomeration
  {
  public:
    /**
     * Constructor.
     */
    MGTwoLevelTransferAgglomeration(const RtreeInfo<dim> &rtree_info);

    /**
     * Initialize transfer operator from coarse to fine level.
     */
    void
    reinit(const AgglomerationHandler<dim> &agglo_handler_fine,
           const AgglomerationHandler<dim> &agglo_handler_coarse);

    /**
     * Destructor.
     */
    ~MGTwoLevelTransferAgglomeration() = default;


    /**
     * Reset the object.
     */
    void
    clear();

    /**
     * Prolongate a vector from level <tt>to_level-1</tt> to level
     * <tt>to_level</tt> using the embedding matrices of the underlying finite
     * element. The previous content of <tt>dst</tt> is overwritten.
     *
     * @param to_level The index of the level to prolongate to, which is the
     * level of @p dst.
     *
     * @param src is a vector with as many elements as there are degrees of
     * freedom on the coarser level involved.
     *
     * @param dst has as many elements as there are degrees of freedom on the
     * finer level.
     */
    virtual void
    prolongate(VectorType &dst, const VectorType &src) const;

    virtual void
    prolongate_and_add(VectorType &dst, const VectorType &src) const;

    /**
     * Restrict a vector from level <tt>from_level</tt> to level
     * <tt>from_level-1</tt> using the transpose operation of the prolongate()
     * method. If the region covered by cells on level <tt>from_level</tt> is
     * smaller than that of level <tt>from_level-1</tt> (local refinement),
     * then some degrees of freedom in <tt>dst</tt> are active and will not be
     * altered. For the other degrees of freedom, the result of the
     * restriction is added.
     *
     * @param from_level The index of the level to restrict from, which is the
     * level of @p src.
     *
     * @param src is a vector with as many elements as there are degrees of
     * freedom on the finer level involved.
     *
     * @param dst has as many elements as there are degrees of freedom on the
     * coarser level.
     */
    virtual void
    restrict_and_add(VectorType &dst, const VectorType &src) const;


  private:
    /**
     * Pointer to coarse AgglomerationHandler
     */
    const AgglomerationHandler<dim> *agglo_handler_coarse;

    /**
     * Pointer to fine AgglomerationHandler
     */
    const AgglomerationHandler<dim> *agglo_handler_fine;

    /**
     * CRS like data structure for every agglomerates.
     */
    std::vector<std::vector<unsigned int>> crss;

    /**
     * Agglomerates.
     */
    std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>>
      agglomerates;

    /**
     * Sparsity patterns for transfer matrices.
     */
    std::shared_ptr<
      typename internal::MatrixSelectorAgglomeration<VectorType>::Sparsity>
      prolongation_sparsity;

    /**
     * The actual prolongation matrix.  column indices belong to the dof
     * indices of the coarse level, while row indices belong to the fine level.
     */
    std::shared_ptr<
      typename internal::MatrixSelectorAgglomeration<VectorType>::Matrix>
      prolongation_matrix;
  };



  template <int dim, typename VectorType>
  MGTwoLevelTransferAgglomeration<dim, VectorType>::
    MGTwoLevelTransferAgglomeration(const RtreeInfo<dim> &info)
    : crss(info.crss)
    , agglomerates(info.agglomerates)
  {
    static_assert(dim != 1);

    prolongation_sparsity = std::make_shared<
      typename internal::MatrixSelectorAgglomeration<VectorType>::Sparsity>();
    prolongation_matrix = std::make_shared<
      typename internal::MatrixSelectorAgglomeration<VectorType>::Matrix>();
  }



  template <int dim, typename VectorType>
  void
  MGTwoLevelTransferAgglomeration<dim, VectorType>::reinit(
    const AgglomerationHandler<dim> &agglo_handler_fine_,
    const AgglomerationHandler<dim> &agglo_handler_coarse_)
  {
    agglo_handler_fine   = &agglo_handler_fine_;
    agglo_handler_coarse = &agglo_handler_coarse_;

    const DoFHandler<dim> &agglo_dh_fine   = agglo_handler_fine->agglo_dh;
    const DoFHandler<dim> &agglo_dh_coarse = agglo_handler_coarse->agglo_dh;
    Assert((agglo_dh_fine.n_dofs() > agglo_dh_coarse.n_dofs()),
           ExcMessage(
             "Coarse DoFHandler has more DoFs than finer DoFHandler."));

    // Setup sparsity pattern
    DynamicSparsityPattern dsp(agglo_dh_fine.n_dofs(),
                               agglo_dh_coarse.n_dofs());

    const auto &       fe            = agglo_dh_coarse.get_fe();
    const unsigned int dofs_per_cell = fe.dofs_per_cell;

    for (std::size_t i = 0; i < agglomerates.size(); ++i)
      {
        // coarse agglomerate
        const auto &                          agglo = agglomerates[i];
        std::vector<types::global_cell_index> cell_indices;
        for (const auto &cell : agglo)
          cell_indices.push_back(cell->active_cell_index());

        const auto &dof_cell_coarse =
          agglo[std::distance(std::begin(cell_indices),
                              std::min_element(std::begin(cell_indices),
                                               std::end(cell_indices)))]
            ->as_dof_handler_iterator(
              agglo_dh_coarse); // min index is master cell

        std::vector<types::global_dof_index> dof_indices_coarse(dofs_per_cell);
        dof_cell_coarse->get_dof_indices(dof_indices_coarse);

        const auto &crs = crss[i]; // vector of indices for current agglomerate

        for (unsigned int k = 0; k < crs.size() - 1; ++k)
          {
            std::vector<types::global_dof_index> dof_indices_fine(
              dofs_per_cell);


            // collect indices of one sub-agglomerate
            std::vector<types::global_cell_index> cell_indices_fine;
            for (unsigned int j = crs[k]; j < crs[k + 1]; ++j)
              cell_indices_fine.push_back(agglo[j]->active_cell_index());

            const unsigned int master_idx =
              std::distance(std::begin(cell_indices_fine),
                            std::min_element(std::begin(cell_indices_fine),
                                             std::end(cell_indices_fine)));

            const auto &dof_cell_fine =
              agglo[crs[k] + master_idx]->as_dof_handler_iterator(
                agglo_dh_fine);

            dof_cell_fine->get_dof_indices(dof_indices_fine);


            for (const auto row : dof_indices_fine)
              dsp.add_entries(row,
                              dof_indices_coarse.begin(),
                              dof_indices_coarse.end());
          }
      }

    prolongation_sparsity->copy_from(dsp);
    prolongation_matrix->reinit(*prolongation_sparsity);



    // Fill prolongation_matrix
    AffineConstraints<typename VectorType::value_type>
      c; // dummy constraint, needed only for loc2glb operations.
    c.close();
    FullMatrix<typename VectorType::value_type> local_matrix(dofs_per_cell,
                                                             dofs_per_cell);
    const std::vector<Point<dim>> &             unit_support_points =
      fe.get_unit_support_points();

    const std::vector<BoundingBox<dim>> bboxes =
      agglo_handler_coarse->get_local_bboxes();

    for (unsigned int i = 0; i < agglomerates.size(); ++i)
      {
        // coarse agglomerate
        const auto &                          agglo = agglomerates[i];
        std::vector<types::global_cell_index> cell_indices;
        for (const auto &cell : agglo)
          cell_indices.push_back(cell->active_cell_index());

        const auto &master_cell_coarse =
          agglo[std::distance(std::begin(cell_indices),
                              std::min_element(std::begin(cell_indices),
                                               std::end(cell_indices)))];
        const auto dof_cell_coarse =
          master_cell_coarse->as_dof_handler_iterator(
            agglo_dh_coarse); // min index is master cell

        std::vector<types::global_dof_index> dof_indices_coarse(dofs_per_cell);
        dof_cell_coarse->get_dof_indices(dof_indices_coarse);



        const auto &crs = crss[i]; // vector of indices for current agglomerate

        for (unsigned int k = 0; k < crs.size() - 1; ++k)
          {
            local_matrix = 0.;
            std::vector<types::global_dof_index> dof_indices_fine(
              dofs_per_cell);

            // collect indices of one sub-agglomerate
            std::vector<types::global_cell_index> cell_indices_fine;
            for (unsigned int j = crs[k]; j < crs[k + 1]; ++j)
              cell_indices_fine.push_back(agglo[j]->active_cell_index());


            const unsigned int master_idx =
              std::distance(std::begin(cell_indices_fine),
                            std::min_element(std::begin(cell_indices_fine),
                                             std::end(cell_indices_fine)));

            const auto &master_cell_fine = agglo[crs[k] + master_idx];

            const auto &dof_cell_fine =
              master_cell_fine->as_dof_handler_iterator(
                agglo_dh_fine); // min index is master cell

            dof_cell_fine->get_dof_indices(dof_indices_fine);

            // compute real location of support points
            std::vector<Point<dim>> real_qpoints;
            for (const Point<dim> &p : unit_support_points)
              real_qpoints.push_back(
                agglo_handler_fine->euler_mapping->transform_unit_to_real_cell(
                  dof_cell_fine, p));
            // real_qpoints.push_back(box_fine.unit_to_real(p));

            for (unsigned int i = 0; i < dof_indices_coarse.size(); ++i)
              {
                const auto &p =
                  agglo_handler_coarse->euler_mapping
                    ->transform_real_to_unit_cell(dof_cell_coarse,
                                                  real_qpoints[i]);
                for (unsigned int j = 0; j < dof_indices_fine.size(); ++j)
                  {
                    local_matrix(i, j) = fe.shape_value(j, p);
                  }
              }

            c.distribute_local_to_global(local_matrix,
                                         dof_indices_fine,
                                         dof_indices_coarse,
                                         *prolongation_matrix);
          }
      }
  }



  template <int dim, typename VectorType>
  void
  MGTwoLevelTransferAgglomeration<dim, VectorType>::prolongate(
    VectorType &      dst,
    const VectorType &src) const
  {
    Assert(prolongation_matrix != nullptr,
           ExcMessage("Matrix has not been initialized."));
    prolongation_matrix->vmult(dst, src);
  }



  template <int dim, typename VectorType>
  void
  MGTwoLevelTransferAgglomeration<dim, VectorType>::prolongate_and_add(
    VectorType &      dst,
    const VectorType &src) const
  {
    Assert(prolongation_matrix != nullptr,
           ExcMessage("Matrix has not been initialized."));
    prolongation_matrix->vmult_add(dst, src);
  }



  template <int dim, typename VectorType>
  void
  MGTwoLevelTransferAgglomeration<dim, VectorType>::restrict_and_add(
    VectorType &      dst,
    const VectorType &src) const
  {
    Assert(prolongation_matrix != nullptr,
           ExcMessage("Matrix has not been initialized."));
    prolongation_matrix->Tvmult_add(dst, src);
  }



  template <int dim, typename VectorType>
  void
  MGTwoLevelTransferAgglomeration<dim, VectorType>::clear()
  {
    prolongation_matrix.reset();
    prolongation_sparsity.reset();
    agglo_handler_fine   = nullptr;
    agglo_handler_coarse = nullptr;
    crss.clear();
    agglomerates.clear();
  }



  template <int dim, typename VectorType>
  class MGTransferAgglomeration : public MGTransferBase<VectorType>
  {
  public:
    /**
     * Constructor. It takes a sequence of two level transfers between
     * agglomerated meshes.
     */
    MGTransferAgglomeration(
      const MGLevelObject<MGTwoLevelTransferAgglomeration<dim, VectorType>>
        &transfer);

    /**
     * Perform prolongation.
     */
    void
    prolongate(const unsigned int to_level,
               VectorType &       dst,
               const VectorType & src) const override;

    /**
     * Perform prolongation.
     */
    void
    prolongate_and_add(const unsigned int to_level,
                       VectorType &       dst,
                       const VectorType & src) const override;

    /**
     * Perform restriction.
     */
    void
    restrict_and_add(const unsigned int from_level,
                     VectorType &       dst,
                     const VectorType & src) const override;

  private:
    MGLevelObject<
      SmartPointer<MGTwoLevelTransferAgglomeration<dim, VectorType>>>
      transfer;
  };


  template <int dim, typename VectorType>
  MGTransferAgglomeration<dim, VectorType>::MGTransferAgglomeration(
    const MGLevelObject<MGTwoLevelTransferAgglomeration<dim, VectorType>>
      &transfer)
  {
    const unsigned int min_level = transfer.min_level();
    const unsigned int max_level = transfer.max_level();

    this->transfer.resize(min_level, max_level);

    for (unsigned int l = min_level; l <= max_level; ++l)
      this->transfer[l] =
        &const_cast<MGTwoLevelTransferAgglomeration<dim, VectorType> &>(
          static_cast<const MGTwoLevelTransferAgglomeration<dim, VectorType> &>(
            Utilities::get_underlying_value(transfer[l])));
  }


  template <int dim, typename VectorType>
  void
  MGTransferAgglomeration<dim, VectorType>::prolongate(
    const unsigned int to_level,
    VectorType &       dst,
    const VectorType & src) const
  {
    dst = typename VectorType::value_type(0.0);
    prolongate_and_add(to_level, dst, src);
  }

  template <int dim, typename VectorType>
  void
  MGTransferAgglomeration<dim, VectorType>::prolongate_and_add(
    const unsigned int to_level,
    VectorType &       dst,
    const VectorType & src) const
  {
    this->transfer[to_level]->prolongate_and_add(dst, src);
  }

  template <int dim, typename VectorType>
  void
  MGTransferAgglomeration<dim, VectorType>::restrict_and_add(
    const unsigned int from_level,
    VectorType &       dst,
    const VectorType & src) const
  {
    this->transfer[from_level]->restrict_and_add(dst, src);
  }



} // namespace dealii

#endif
