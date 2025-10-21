/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "State_JPLQuatLocal.h"

#include "utils/quat_ops.h"

using namespace ov_init;

bool State_JPLQuatLocal::Plus(const double *x, const double *delta, double *x_plus_delta) const {

  // Apply the standard JPL update: q <-- [d_th/2; 1] (x) q
  Eigen::Map<const Eigen::Vector4d> q(x);

  // Get delta into eigen
  Eigen::Map<const Eigen::Vector3d> d_th(delta);
  Eigen::Matrix<double, 4, 1> d_q;
  double theta = d_th.norm();
  if (theta < 1e-8) {
    d_q << .5 * d_th, 1.0;
  } else {
    d_q.block(0, 0, 3, 1) = (d_th / theta) * std::sin(theta / 2);
    d_q(3, 0) = std::cos(theta / 2);
  }
  d_q = ov_core::quatnorm(d_q);

  // Do the update
  Eigen::Map<Eigen::Vector4d> q_plus(x_plus_delta);
  q_plus = ov_core::quat_multiply(d_q, q);
  return true;
}

bool State_JPLQuatLocal::PlusJacobian(const double *x, double *jacobian) const {
  Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> j(jacobian);
  j.topRows<3>().setIdentity();
  j.bottomRows<1>().setZero();
  return true;
}

bool State_JPLQuatLocal::Minus(const double *y, const double *x, double *y_minus_x) const {
  // Compute the "difference" between two quaternions
  // This is the inverse operation of Plus
  Eigen::Map<const Eigen::Vector4d> q_x(x);
  Eigen::Map<const Eigen::Vector4d> q_y(y);

  // Compute q_y * conjugate(q_x) to get the relative rotation
  Eigen::Vector4d q_x_conj = q_x;
  q_x_conj.head<3>() *= -1.0; // conjugate of JPL quaternion

  Eigen::Vector4d q_diff = ov_core::quat_multiply(q_y, q_x_conj);

  // Convert quaternion difference to axis-angle representation
  Eigen::Map<Eigen::Vector3d> delta(y_minus_x);
  if (std::abs(q_diff(3)) >= 1.0) {
    delta.setZero();
  } else {
    double theta = 2.0 * std::acos(std::abs(q_diff(3)));
    if (theta < 1e-8) {
      delta = 2.0 * q_diff.head<3>();
    } else {
      delta = theta * q_diff.head<3>() / q_diff.head<3>().norm();
    }
  }

  return true;
}

bool State_JPLQuatLocal::MinusJacobian(const double *x, double *jacobian) const {
  // For this simple implementation, we use the same jacobian as Plus
  // In practice, this might need more sophisticated computation
  Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> j(jacobian);
  j.leftCols<3>().setIdentity();
  j.rightCols<1>().setZero();
  return true;
}
