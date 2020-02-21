#pragma once

#ifndef VIEWER_H
#define VIEWER_H

#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>

class Viewer {
public:
	Viewer();

	~Viewer();

	void show_3dcloud(const std::vector<cv::Vec3f>& pt3d);

	cv::Mat show_2dmask(
		const std::vector<cv::Vec2i>& pt2d,
		const cv::Mat& img_,
		const bool display = true,
		const int resize_width = -1,
		const int resize_height = -1
	);

	static void random_sample_patch_dual(
		const cv::Mat& img_a,
		const cv::Mat& img_b,
		cv::Mat& patcha,
		cv::Mat& patchb,
		const int height,
		const int width
	);

	static void random_sample_patch_solo(
		const cv::Mat& img_a,
		cv::Mat& patcha,
		const int height,
		const int width
	);

	static void color_boundary_gray(
		cv::Mat& img_,
		const int thickness,
		const int gray_value
	);

	template<typename DATTYPE>
	static cv::Mat inv_intensity(
		const cv::Mat& img_,
		DATTYPE val
	);

	template<typename DATTYPE>
	static cv::Mat binarize_img(
		const cv::Mat& img_, 
		DATTYPE threshold,
		DATTYPE upper_val,
		DATTYPE lower_Val
	);

private:
	cv::viz::Viz3d viz;
};

#endif // !VIEWER_H