#include <iostream>
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth)
{
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  if (estimations.size() == 0 || estimations.size() != ground_truth.size())
  {
    cout << "Invalid estimation or ground_truth vectors" << endl;
    return rmse;
  }

  for (unsigned int i = 0; i < estimations.size(); ++i)
  {
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  rmse = rmse / estimations.size();

  rmse = rmse.array().sqrt();

  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state)
{
  MatrixXd Hj(3, 4);
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  float epsilon = 0.00001;
  float px2 = px * px;
  float py2 = py * py;
  float px2_py2 = px2 + py2 + epsilon;
  float sqrt_px2_py2 = sqrt(px2_py2);
  float sqrt_px2_py2_3 = px2_py2 * sqrt_px2_py2;

  float dp_dpx = px / sqrt_px2_py2;
  float dp_dpy = py / sqrt_px2_py2;
  float dphi_dpx = -py / px2_py2;
  float dphi_dpy = px / px2_py2;
  float dp_dot_dpx = py * (vx * py - vy * px) / sqrt_px2_py2_3;
  float dp_dot_dpy = px * (vy * px - vx * py) / sqrt_px2_py2_3;
  float dp_dot_dvx = px / sqrt_px2_py2;
  float dp_dot_dvy = py / sqrt_px2_py2;

  Hj << dp_dpx, dp_dpy, 0, 0,
      dphi_dpx, dphi_dpy, 0, 0,
      dp_dot_dpx, dp_dot_dpy, dp_dot_dvx, dp_dot_dvy;

  return Hj;
}
