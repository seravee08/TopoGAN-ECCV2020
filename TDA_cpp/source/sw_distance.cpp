#include "sw_distance.h"

#include <stdlib.h>
#include <cstdlib>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <set>
#include <map>
#include <limits>
#include <cmath>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <cstdio>
#include <cassert>
#include <ctime>

PD sliced_wassasertein_distance::PDi;
PD sliced_wassasertein_distance::PDj;

// ===== Public functions =====
cv::Mat sliced_wassasertein_distance::compute_SW_matrix(
	const std::vector<std::vector<std::vector<double>>>& BD_list,
	const bool approx,
	const int slices
) {
	cv::Mat dis_matrix;
	const int instance_num = BD_list.size();
	const int dims = BD_list[0].size();

	switch (dims) {
	case 2:
		dis_matrix = cv::Mat(instance_num, instance_num, CV_64FC2, cv::Scalar(0, 0));
		for (int i = 0; i < instance_num; i++) {
			std::cout << "Processing " << i << "/" << instance_num << " ." << std::endl;
			for (int j = 0; j < instance_num; j++) {
				for (size_t k = 0; k < 2; k++) {
					dis_matrix.at<cv::Vec2d>(i, j)[k] = (approx) ?
						compute_approximate_SW(
							BD_list[i][k], BD_list[j][k], slices) :
						compute_exact_SW(
							BD_list[i][k], BD_list[j][k]);
				}
			}
		}
		break;
	case 3:
		dis_matrix = cv::Mat(instance_num, instance_num, CV_64FC3, cv::Scalar(0, 0, 0));
		for (int i = 0; i < instance_num; i++) {
			std::cout << "Processing " << i << "/" << instance_num << " ." << std::endl;
			for (int j = 0; j < instance_num; j++) {
				for (size_t k = 0; k < 3; k++) {
					dis_matrix.at<cv::Vec3d>(i, j)[k] = (approx) ?
						compute_approximate_SW(
							BD_list[i][k], BD_list[j][k], slices) :
						compute_exact_SW(
							BD_list[i][k], BD_list[j][k]);
				}
			}
		}
		break;
	case 4:
		dis_matrix = cv::Mat(instance_num, instance_num, CV_64FC4, cv::Scalar(0, 0, 0, 0));
		for (int i = 0; i < instance_num; i++) {
			std::cout << "Processing " << i << "/" << instance_num << " ." << std::endl;
			for (int j = 0; j < instance_num; j++) {
				for (size_t k = 0; k < 4; k++) {
					dis_matrix.at<cv::Vec4d>(i, j)[k] = (approx) ?
						compute_approximate_SW(
							BD_list[i][k], BD_list[j][k], slices) :
						compute_exact_SW(
							BD_list[i][k], BD_list[j][k]);
				}
			}
		}
		break;
	default:
		break;
	}
	return dis_matrix;
}

double sliced_wassasertein_distance::compute_approximate_SW(
	const std::vector<double>& PD1_,
	const std::vector<double>& PD2_,
	int N
) {
	clear_PDs();
	PD PD1 = convert_vec2PD(PD1_);
	PD PD2 = convert_vec2PD(PD2_);
	double step = NUMPI_PD / N; double sw = 0;

	// Add projections onto diagonal.
	// ******************************
	int n1, n2; n1 = PD1.size(); n2 = PD2.size();
	for (int i = 0; i < n2; i++)
		PD1.push_back(std::pair<double, double>((PD2[i].first + PD2[i].second) / 2, (PD2[i].first + PD2[i].second) / 2));
	for (int i = 0; i < n1; i++)
		PD2.push_back(std::pair<double, double>((PD1[i].first + PD1[i].second) / 2, (PD1[i].first + PD1[i].second) / 2));
	int n = PD1.size();

	// Sort and compare all projections.
	// *********************************
	//#pragma omp parallel for
	for (int i = 0; i < N; i++) {
		std::vector<std::pair<int, double> > L1, L2;
		for (int j = 0; j < n; j++) {
			L1.push_back(std::pair<int, double>(j, PD1[j].first*cos(-NUMPI_PD / 2 + i*step) + PD1[j].second*sin(-NUMPI_PD / 2 + i*step)));
			L2.push_back(std::pair<int, double>(j, PD2[j].first*cos(-NUMPI_PD / 2 + i*step) + PD2[j].second*sin(-NUMPI_PD / 2 + i*step)));
		}
		std::sort(L1.begin(), L1.end(), myComp); std::sort(L2.begin(), L2.end(), myComp);
		double f = 0; for (int j = 0; j < n; j++)  f += std::abs(L1[j].second - L2[j].second);
		sw += f*step;
	}
	return sw / NUMPI_PD;
}

double sliced_wassasertein_distance::compute_exact_SW(
	const std::vector<double>& PD1_,
	const std::vector<double>& PD2_
) {
	clear_PDs();
	PD PD1 = convert_vec2PD(PD1_);
	PD PD2 = convert_vec2PD(PD2_);

	// Add projections onto diagonal.
	// ******************************
	int n1, n2; n1 = PD1.size(); n2 = PD2.size(); double max_ordinate = std::numeric_limits<double>::min();
	for (int i = 0; i < n2; i++) {
		max_ordinate = std::max(max_ordinate, PD2[i].second);
		PD1.push_back(std::pair<double, double>(((PD2[i].first + PD2[i].second) / 2), ((PD2[i].first + PD2[i].second) / 2)));
	}
	for (int i = 0; i < n1; i++) {
		max_ordinate = std::max(max_ordinate, PD1[i].second);
		PD2.push_back(std::pair<double, double>(((PD1[i].first + PD1[i].second) / 2), ((PD1[i].first + PD1[i].second) / 2)));
	}
	int N = PD1.size(); assert(N == PD2.size());

	// Slightly perturb the points so that the PDs are in generic positions.
	// *********************************************************************
	int mag = 0; while (max_ordinate > 10) { mag++; max_ordinate /= 10; }
	double thresh = pow(10, -5 + mag);
	srand(time(NULL));
	for (int i = 0; i < N; i++) {
		PD1[i].first += thresh*(1.0 - 2.0*rand() / RAND_MAX); PD1[i].second += thresh*(1.0 - 2.0*rand() / RAND_MAX);
		PD2[i].first += thresh*(1.0 - 2.0*rand() / RAND_MAX); PD2[i].second += thresh*(1.0 - 2.0*rand() / RAND_MAX);
	}

	// Compute all angles in both PDs.
	// *******************************
	std::vector<std::pair<double, std::pair<int, int> > > angles1, angles2;
	for (int i = 0; i < N; i++) {
		for (int j = i + 1; j < N; j++) {
			double theta1 = compute_angle(PD1, i, j); double theta2 = compute_angle(PD2, i, j);
			angles1.push_back(std::pair<double, std::pair<int, int> >(theta1, std::pair<int, int>(i, j)));
			angles2.push_back(std::pair<double, std::pair<int, int> >(theta2, std::pair<int, int>(i, j)));
		}
	}

	// Sort angles.
	// ************
	std::sort(angles1.begin(), angles1.end(), sortAngle); std::sort(angles2.begin(), angles2.end(), sortAngle);

	// Initialize orders of the points of both PD (given by ordinates when theta = -pi/2).
	// ***********************************************************************************
	PDi = PD1; PDj = PD2;
	std::vector<int> orderp1, orderp2;
	for (int i = 0; i < N; i++) { orderp1.push_back(i); orderp2.push_back(i); }
	std::sort(orderp1.begin(), orderp1.end(), compOri); std::sort(orderp2.begin(), orderp2.end(), compOrj);

	// Find the inverses of the orders.
	// ********************************
	std::vector<int> order1(N); std::vector<int> order2(N);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
			if (orderp1[j] == i)
				order1[i] = j;
	}
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
			if (orderp2[j] == i)
				order2[i] = j;
	}

	// Record all inversions of points in the orders as theta varies along the positive half-disk.
	// *******************************************************************************************
	std::vector<std::vector<std::pair<int, double> > > anglePerm1(N);
	std::vector<std::vector<std::pair<int, double> > > anglePerm2(N);

	int M1 = angles1.size();
	for (int i = 0; i < M1; i++) {
		double theta = angles1[i].first; int p = angles1[i].second.first; int q = angles1[i].second.second;
		anglePerm1[order1[p]].push_back(std::pair<int, double>(p, theta));
		anglePerm1[order1[q]].push_back(std::pair<int, double>(q, theta));
		int a = order1[p]; int b = order1[q]; order1[p] = b; order1[q] = a;
	}

	int M2 = angles2.size();
	for (int i = 0; i < M2; i++) {
		double theta = angles2[i].first; int p = angles2[i].second.first; int q = angles2[i].second.second;
		anglePerm2[order2[p]].push_back(std::pair<int, double>(p, theta));
		anglePerm2[order2[q]].push_back(std::pair<int, double>(q, theta));
		int a = order2[p]; int b = order2[q]; order2[p] = b; order2[q] = a;
	}

	for (int i = 0; i < N; i++) {
		anglePerm1[order1[i]].push_back(std::pair<int, double>(i, NUMPI_PD / 2));
		anglePerm2[order2[i]].push_back(std::pair<int, double>(i, NUMPI_PD / 2));
	}

	// Compute the SW distance with the list of inversions.
	// ****************************************************
	return compute_sw(anglePerm1, anglePerm2);
}

// ===== Private functions =====
void sliced_wassasertein_distance::clear_PDs() {
	PDi.clear();
	PDj.clear();
}

PD sliced_wassasertein_distance::convert_vec2PD(const std::vector<double>& t) {
	assert(t.size() % 2 == 0);
	PD res(t.size() / 2);
	for (int i = 0; i < res.size(); i++) {
		res[i].first = t[i * 2];
		res[i].second = t[i * 2 + 1];
	}
	return res;
}

bool sliced_wassasertein_distance::compOri(const int& p, const int& q) {
	if (PDi[p].second != PDi[q].second)
		return (PDi[p].second < PDi[q].second);
	else
		return (PDi[p].first > PDi[q].first);
}

bool sliced_wassasertein_distance::compOrj(const int& p, const int& q) {
	if (PDj[p].second != PDj[q].second)
		return (PDj[p].second < PDj[q].second);
	else
		return (PDj[p].first > PDj[q].first);
}

bool sliced_wassasertein_distance::myComp(
	const std::pair<int, double> & P1,
	const std::pair<int, double> & P2
) {
	return P1.second < P2.second;
}

double sliced_wassasertein_distance::compute_int_cos(
	const double& alpha,
	const double& beta
) {
	double res;
	assert((alpha >= 0 && alpha <= NUMPI_PD) || (alpha >= -NUMPI_PD && alpha <= 0));
	if (alpha >= 0 && alpha <= NUMPI_PD) {
		if (cos(alpha) >= 0) {
			if (NUMPI_PD / 2 <= beta) { res = 2 - sin(alpha) - sin(beta); }
			else { res = sin(beta) - sin(alpha); }
		}
		else {
			if (1.5*NUMPI_PD <= beta) { res = 2 + sin(alpha) + sin(beta); }
			else { res = sin(alpha) - sin(beta); }
		}
	}
	if (alpha >= -NUMPI_PD && alpha <= 0) {
		if (cos(alpha) <= 0) {
			if (-NUMPI_PD / 2 <= beta) { res = 2 + sin(alpha) + sin(beta); }
			else { res = sin(alpha) - sin(beta); }
		}
		else {
			if (NUMPI_PD / 2 <= beta) { res = 2 - sin(alpha) - sin(beta); }
			else { res = sin(beta) - sin(alpha); }
		}
	}
	return res;
}

double sliced_wassasertein_distance::compute_int(
	const double& theta1,
	const double& theta2,
	const int& p,
	const int& q
) {
	double norm = std::sqrt(pow(PDi[p].first - PDj[q].first, 2) + pow(PDi[p].second - PDj[q].second, 2));
	double angle1;
	if (PDi[p].first > PDj[q].first)
		angle1 = theta1 - asin((PDi[p].second - PDj[q].second) / norm);
	else
		angle1 = theta1 - asin((PDj[q].second - PDi[p].second) / norm);
	double angle2 = angle1 + theta2 - theta1;
	double integral = compute_int_cos(angle1, angle2);
	return norm*integral;
}

double sliced_wassasertein_distance::compute_angle(
	const PD& PersDiag,
	const int& i,
	const int& j
) {
	std::pair<double, double> vect; double x1, y1, x2, y2;
	x1 = PersDiag[i].first; y1 = PersDiag[i].second;
	x2 = PersDiag[j].first; y2 = PersDiag[j].second;
	if (y1 - y2 > 0) {
		vect.first = y1 - y2;
		vect.second = x2 - x1;
	}
	else {
		if (y1 - y2 < 0) {
			vect.first = y2 - y1;
			vect.second = x1 - x2;
		}
		else {
			vect.first = 0;
			vect.second = abs(x1 - x2);
		}
	}
	double norm = std::sqrt(pow(vect.first, 2) + pow(vect.second, 2));
	return asin(vect.second / norm);
}

bool sliced_wassasertein_distance::sortAngle(
	const std::pair<double,
	std::pair<int, int> >& p1,
	const std::pair<double,
	std::pair<int, int> >& p2
) {
	return p1.first < p2.first;
}

double sliced_wassasertein_distance::compute_sw(
	const std::vector<std::vector<std::pair<int, double> > >& V1,
	const std::vector<std::vector<std::pair<int, double> > >& V2
) {
	int N = V1.size(); double sw = 0;
	for (int i = 0; i < N; i++) {
		std::vector<std::pair<int, double> > U, V; U = V1[i]; V = V2[i];
		double theta1, theta2; theta1 = -NUMPI_PD / 2;
		int ku, kv; ku = 0; kv = 0; theta2 = std::min(U[ku].second, V[kv].second);
		while (theta1 != NUMPI_PD / 2) {
			if (PDi[U[ku].first].first != PDj[V[kv].first].first || PDi[U[ku].first].second != PDj[V[kv].first].second)
				if (theta1 != theta2)
					sw += compute_int(theta1, theta2, U[ku].first, V[kv].first);
			theta1 = theta2;
			if ((theta2 == U[ku].second) && ku < U.size() - 1) { ku++; }
			if ((theta2 == V[kv].second) && kv < V.size() - 1) { kv++; }
			theta2 = std::min(U[ku].second, V[kv].second);
		}
	}
	return sw / NUMPI_PD;
}