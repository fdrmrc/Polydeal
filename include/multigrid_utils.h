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
  const LinearOperator<VectorType> linear_op;

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
    , linear_op()
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
    , linear_op(linear_op)
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
    linear_op = linear_op_;
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
    linear_op      = null_operator(linear_op);
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
      *solver, linear_op, *preconditioner, dst, src);
  }


} // namespace dealii

#endif