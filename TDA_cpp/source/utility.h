#pragma once

#ifndef UTILITY_H
#define UTILITY_H

#include <opencv2/viz.hpp>

class Utility {
public:
	Utility() {};

	~Utility() {};

	template <typename DATTYPE>
	static std::string tostr(const DATTYPE& t);

	template <typename DATTYPE>
	static std::vector<int> sort_indices(
		const std::vector<DATTYPE>& t,
		const bool ascend
	);

	template<typename DATTYPE>
	static void nested4vec_2_Vec3f(
		const std::vector<std::vector<std::vector<std::vector<DATTYPE>>>>& origin,
		std::vector<cv::Vec3f>& pt3d
	);

	template<typename DATTYPE>
	static void nested3vec_2_Vec3f(
		const std::vector<std::vector<std::vector<DATTYPE>>>& origin,
		std::vector<cv::Vec3f>& pt3d
	);

	template<typename DATTYPE>
	static void nested4vec_2_Vec2i(
		const std::vector<std::vector<std::vector<std::vector<DATTYPE>>>>& origin,
		std::vector<cv::Vec2i>& pt2d
	);

	template<typename DATTYPE>
	static void nested3vec_2_Vec2i(
		const std::vector<std::vector<std::vector<DATTYPE>>>& origin,
		std::vector<cv::Vec2i>& pt2d
	);

	template<typename DATTYPE>
	static void nested2vec_2_Vec2i(
		const std::vector<std::vector<DATTYPE>>& origin,
		std::vector<cv::Vec2i>& pt2d
	);

	template<typename DATTYPE1, typename DATTYPE2>
	static void extract_beyond_or_below_thresh(
		const std::vector<std::vector<std::vector<std::vector<DATTYPE1>>>>& data,
		const std::vector<std::vector<DATTYPE2>>& reference,
		const double th,
		const int mode,
		std::vector<std::vector<std::vector<std::vector<DATTYPE1>>>>& res
	);

	template<typename DATTYPE1, typename DATTYPE2>
	static void extract_beyond_or_below_thresh(
		const std::vector<std::vector<DATTYPE1>>& data,
		const std::vector<std::vector<DATTYPE2>>& reference,
		const double th,
		const int mode,
		std::vector<std::vector<DATTYPE1>>& res
	);

	template<typename DATTYPE>
	static void extract_by_num_range(
		const std::vector<std::vector<std::vector<std::vector<DATTYPE>>>>& data,
		const int lwr,
		const int upr,
		std::vector<std::vector<std::vector<std::vector<DATTYPE>>>>& res
	);

	template<typename DATTYPE>
	static void sort_by_elenum(
		const std::vector<std::vector<std::vector<std::vector<DATTYPE>>>>& data,
		const bool ascend,
		std::vector<std::vector<std::vector<std::vector<DATTYPE>>>>& res
	);

	template<typename DATTYPE>
	static std::vector<std::vector<DATTYPE>> cvt_BD2Pers(
		const std::vector<std::vector<DATTYPE>>& data
	);

	template<typename DATTYPE>
	static std::vector<std::vector<DATTYPE>> create_2D_array(
		const std::vector<int>& dims,
		const DATTYPE init_value
	);

	template<typename DATTYPE>
	static std::vector<std::vector<std::vector<DATTYPE>>> create_3D_array(
		const std::vector<int>& dims,
		const DATTYPE init_value
	);

	template<typename DATTYPE>
	static void shift_mask(
		cv::Mat&   mask,
		const int  target_label,
		const int  assign_label,
		const int  target_dim,
		const int  shift_pix,
		const bool increase
	);

	template<typename DATTYPE>
	static cv::Mat retrieve_mat_by_mask_2D3D(
		const cv::Mat& data,
		const cv::Mat& mask,
		DATTYPE threshold,
		DATTYPE background_val,
		bool select_beyond_threshold
	);

	template<typename DATTYPE>
	static cv::Mat find_content_bound_3D(
		const cv::Mat& data,
		DATTYPE background_val
	);

	// For checking opencv mat type
	static std::string type2str(int type);

	static bool hasEnding(const std::string& fullString, const std::string& ending);

	static void seg_str_by_char(
		const std::string& t,
		const std::string& token,
		std::string& pre,
		std::string& post
	);

	// ===== DMT code sections =====
	static std::vector<cv::Vec3f> retrieve_DMT_cycles_by_index(
		std::vector<cv::Point3i>& coord,
		std::vector<std::vector<int>>& cycs
	);
	// ==============================
};

#endif // !UTILITY_H