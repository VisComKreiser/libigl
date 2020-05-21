// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2020 Xiangyu Kong <xiangyu.kong@mail.utoronto.ca>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#ifndef IGL_DIRECT_DELTA_MUSH_H
#define IGL_DIRECT_DELTA_MUSH_H

#include "igl_inline.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <Eigen/LU>
#include <vector>

namespace igl {
  // Computes Direct Delta Mesh Skinning (Variant 0) from "Direct Delta Mush
  // Skinning and Variants"
  //
  // Inputs:
  //   V  #V by 3 list of rest pose vertex positions
  //   F  #F by 3 list of triangle indices into rows of V
  //   C  #C by 3 list of rest pose bone endpoint positions
  //   E  #T by 2 list of bone edge indices into rows of C
  //   T  #T list of bone pose transformations
  // Outputs:
  //   U  #V by 3 list of output vertex positions
  template <
    typename DerivedV,
    typename DerivedF,
    typename DerivedC,
    typename DerivedE,
    typename DerivedW,
    typename DerivedT,
    typename DerivedTAlloc,
    typename DerivedU>
  IGL_INLINE void direct_delta_mush(
    const Eigen::MatrixBase<DerivedV> &V,
    const Eigen::MatrixBase<DerivedF> &F,
    const Eigen::MatrixBase<DerivedC> &C,
    const Eigen::MatrixBase<DerivedE> &E,
    const Eigen::SparseMatrix<DerivedW> &W,
    const std::vector<DerivedT, DerivedTAlloc> &T,
    Eigen::PlainObjectBase<DerivedU> &U);

  // Precomputation
  //
  // Inputs:
  //   V  #V by 3 list of rest pose vertex positions
  //   F  #F by 3 list of triangle indices into rows of V
  //   C  #C by 3 list of rest pose bone endpoint positions
  //   E  #T by 2 list of bone edge indices into rows of C
  //   W  #V by #C list of weights // TODO: this could be a sparse matrix?
  //   p  number of smoothing iterations
  //   lambda  smoothing step size
  //   kappa  smoothness parameter (section 3.3)
  // Outputs:
  //   Omega  #V by #T*10 list of precomputated matrix values
  template <
    typename DerivedV,
    typename DerivedF,
    typename DerivedC,
    typename DerivedE,
    typename DerivedW,
    typename DerivedOmega>
  IGL_INLINE void direct_delta_mush_precomputation(
    const Eigen::MatrixBase<DerivedV> &V,
    const Eigen::MatrixBase<DerivedF> &F,
    const Eigen::MatrixBase<DerivedC> &C,
    const Eigen::MatrixBase<DerivedE> &E,
    const Eigen::SparseMatrix<DerivedW> &W,
    const int p,
    const typename DerivedV::Scalar lambda,
    const typename DerivedV::Scalar kappa,
    Eigen::PlainObjectBase<DerivedOmega> &Omega);

  // Pose evaluation
  //   Omega  #V by #T*10 list of precomputated matrix values
  //   T  #T list of bone pose transformations
  // Outputs:
  //   U  #V by 3 list of output vertex positions
  template <
    typename DerivedT,
    typename DerivedTAlloc,
    typename DerivedOmega,
    typename DerivedU>
  IGL_INLINE void direct_delta_mush_pose_evaluation(
    const std::vector<DerivedT, DerivedTAlloc> &T,
    const Eigen::MatrixBase<DerivedOmega> &Omega,
    Eigen::PlainObjectBase<DerivedU> &U);
} // namespace igl

#ifndef IGL_STATIC_LIBRARY
#  include "direct_delta_mush.cpp"
#endif

#endif