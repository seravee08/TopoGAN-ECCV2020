#pragma once

#ifndef SW_DISTANCE_H
#define SW_DISTANCE_H

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

#define NUMPI_PD 3.14159265359

using PD = std::vector<std::pair<double, double>>;

class sliced_wassasertein_distance {
public:
	sliced_wassasertein_distance() {};

	~sliced_wassasertein_distance() {};

	static cv::Mat compute_SW_matrix(
		const std::vector<std::vector<std::vector<double>>>& BD_list,
		const bool approx,
		const int slices
	);

	static double compute_approximate_SW(
		const std::vector<double>& PD1_,
		const std::vector<double>& PD2_,
		int N = 100
	);

	static double compute_exact_SW(
		const std::vector<double>& PD1_,
		const std::vector<double>& PD2_
	);

private:
	static void clear_PDs();
	static PD convert_vec2PD(const std::vector<double>& t);
	static bool compOri(const int& p, const int& q);
	static bool compOrj(const int& p, const int& q);
	static bool myComp(const std::pair<int, double> & P1, const std::pair<int, double> & P2);

	static double compute_int_cos(const double& alpha, const double& beta);
	static double compute_int(const double& theta1, const double& theta2, const int& p, const int& q);
	static double compute_angle(const PD& PersDiag, const int& i, const int& j);

	static bool sortAngle(
		const std::pair<double,
		std::pair<int, int> >& p1,
		const std::pair<double,
		std::pair<int, int> >& p2
	);

	static double compute_sw(
		const std::vector<std::vector<std::pair<int, double> > >& V1,
		const std::vector<std::vector<std::pair<int, double> > >& V2
	);

private:
	static PD PDi, PDj;
};

#endif //!SW_DISTANCE_H