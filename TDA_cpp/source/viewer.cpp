#include "viewer.h"
#include "utility.h"

#include <time.h>

// Definations for Viewer Class
Viewer::Viewer()
{
	// Set the initial viewing point for the viewer
	cv::Matx<float, 4, 4> initial_pose(
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, -20,
		0, 0, 0, 1);
	cv::Affine3d initial_pose_aff(initial_pose);

	// Initialize the viewer
	viz = cv::viz::Viz3d("Model Viewer");
	viz.setBackgroundColor(cv::viz::Color::black());
	viz.showWidget("World Frame", cv::viz::WCoordinateSystem());
	viz.setWindowPosition(cv::Point(150, 150));
	viz.showWidget("text2d", cv::viz::WText("Model Viewer", cv::Point(20, 20), 20, cv::viz::Color::green()));
	viz.setViewerPose(initial_pose_aff);
}

Viewer::~Viewer()
{
	viz.close();
}

void Viewer::show_3dcloud(const std::vector<cv::Vec3f>& pt3d) {
	viz.showWidget("Model", cv::viz::WCloud(pt3d, cv::viz::Color::bluberry()));
	viz.spin();
}

cv::Mat Viewer::show_2dmask(
	const std::vector<cv::Vec2i>& pt2d,
	const cv::Mat& img_,
	const bool display,
	const int resize_width,
	const int resize_height
) {
	cv::Mat img = img_.clone();
	if (img_.channels() == 1) {
		cv::cvtColor(img_, img, CV_GRAY2BGR);
	}

	cv::Vec3b color = { 0, 0, 255 };
	for (int i = 0; i < pt2d.size(); i++) {
		img.at<cv::Vec3b>(cv::Point(pt2d[i][0], pt2d[i][1])) = color;
	}

	// ===== Draw grid on the images in red =====
	//cv::Mat mask = cv::Mat::zeros(img_.size(), img_.type());
	//const int width = mask.cols;
	//const int height = mask.rows;
	//int dist = 50;
	//for (int i = 0; i < height; i += dist)
	//	cv::line(mask, cv::Point(0, i), cv::Point(width, i), cv::Scalar(255));

	//for (int i = 10; i < width; i += dist)
	//	cv::line(mask, cv::Point(i, 0), cv::Point(i, height), cv::Scalar(255));

	//for (int i = 0; i < height; i++) {
	//	for (int j = 0; j < width; j++) {
	//		if (mask.at<uchar>(i, j) == 255) {
	//			uchar tmp = img_.at<uchar>(i, j);
	//			img.at<cv::Vec3b>(i+1, j+1) = cv::Vec3b(tmp, tmp, tmp);
	//		}
	//	}
	//}
	
	if (display) {
		if (resize_width > 0 && resize_height > 0) {
			cv::Mat resized;
			cv::resize(img, resized, cv::Size(resize_width, resize_height));
			cv::imshow("Result", resized);
			cv::waitKey(0);
		}
		else {
			cv::imshow("Result", img);
			cv::waitKey(0);
		}
	}
	return img;
}

// Sample a random patch from the image
void Viewer::random_sample_patch_dual(
	const cv::Mat& img_a,
	const cv::Mat& img_b,
	cv::Mat& patcha,
	cv::Mat& patchb,
	const int height,
	const int width
) {
	const int img_height = img_a.rows;
	const int img_width  = img_a.cols;
	assert(img_height == img_b.rows);
	assert(img_width  == img_b.cols);
	assert(height <= img_height);
	assert(width <= img_width);
	const int valid_height = img_height - height;
	const int valid_width  = img_width - width;

	int rand_x = rand() % (valid_width + 1);
	int rand_y = rand() % (valid_height + 1);
	cv::Rect roi(rand_x, rand_y, width, height);
	patcha = img_a(roi).clone();
	patchb = img_b(roi).clone();
}

// Sample a random patch from the image, solo version
void Viewer::random_sample_patch_solo(
	const cv::Mat& img_a,
	cv::Mat& patcha,
	const int height,
	const int width
) {
	const int img_height = img_a.rows;
	const int img_width = img_a.cols;
	assert(height <= img_height);
	assert(width <= img_width);
	const int valid_height = img_height - height;
	const int valid_width = img_width - width;

	int rand_x = rand() % (valid_width + 1);
	int rand_y = rand() % (valid_height + 1);
	cv::Rect roi(rand_x, rand_y, width, height);
	patcha = img_a(roi).clone();
}

// Draw a bounding box around image
void Viewer::color_boundary_gray(
	cv::Mat& img_,
	const int thickness,
	const int gray_value
) {
	const int img_height = img_.rows;
	const int img_width = img_.cols;
	cv::Point top_left(0, 0), bottom_right(img_width - 1, img_height - 1);
	cv::rectangle(img_, top_left, bottom_right, cv::Scalar(gray_value), thickness);
}

// Inverse a grayscale image
template<typename DATTYPE>
cv::Mat Viewer::inv_intensity(
	const cv::Mat& img_,
	DATTYPE val
) {
	return val - img_;
}

// Binarize image with two values by threshold
template<typename DATTYPE>
cv::Mat Viewer::binarize_img(
	const cv::Mat& img_,
	DATTYPE threshold,
	DATTYPE upper_val,
	DATTYPE lower_Val
) {
	const int rows = img_.rows;
	const int cols = img_.cols;
	cv::Mat res(img_.size(), img_.type(), cv::Scalar(0));
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			if (img_.at<DATTYPE>(i, j) > threshold)
				res.at<DATTYPE>(i, j) = upper_val;
			else
				res.at<DATTYPE>(i, j) = lower_Val;
	return res;
}

// ===== Instantiate templates =====
template cv::Mat Viewer::inv_intensity<uchar>(
	const cv::Mat& img_,
	uchar val
);

template cv::Mat Viewer::binarize_img<uchar>(
	const cv::Mat& img_,
	uchar threshold,
	uchar upper_val,
	uchar lower_Val
);