#include "utility.h"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <sstream>

template <typename DATTYPE>
static std::string Utility::tostr(const DATTYPE& t) {
	std::ostringstream os;
	os << t;
	return os.str();
}

template <typename DATTYPE>
static std::vector<int> Utility::sort_indices(const std::vector<DATTYPE>& t, const bool ascend) {

	// Note: this function will NOT sort t itself
	std::vector<int> idx(t.size());
	std::iota(idx.begin(), idx.end(), 0);
	if (ascend)
		sort(idx.begin(), idx.end(), [&t](int i1, int i2) {return t[i1] < t[i2]; });
	else
		sort(idx.begin(), idx.end(), [&t](int i1, int i2) {return t[i1] > t[i2]; });
	return idx;
}

template<typename DATTYPE>
static void Utility::nested4vec_2_Vec3f(
	const std::vector<std::vector<std::vector<std::vector<DATTYPE>>>>& origin,
	std::vector<cv::Vec3f>& pt3d
) {
	int index     = 0;
	int pts_count = 0;
	for (int i = 0; i < origin.size(); i++) {
		for (int j = 0; j < origin[i].size(); j++) {
			pts_count += origin[i][j].size();
		}
	}
	pt3d.resize(pts_count);
	for (int i = 0; i < origin.size(); i++) {
		for (int j = 0; j < origin[i].size(); j++) {
			for (int k = 0; k < origin[i][j].size(); k++) {
				pt3d[index++] = cv::Vec3f(
					(float)origin[i][j][k][0],
					(float)origin[i][j][k][1],
					(float)origin[i][j][k][2]
				);
			}
		}
	}
}

template<typename DATTYPE>
static void Utility::nested3vec_2_Vec3f(
	const std::vector<std::vector<std::vector<DATTYPE>>>& origin,
	std::vector<cv::Vec3f>& pt3d
) {
	int index = 0;
	int pts_count = 0;
	for (int i = 0; i < origin.size(); i++) {
		pts_count += origin[i].size();
	}
	pt3d.resize(pts_count);
	for (int i = 0; i < origin.size(); i++) {
		for (int j = 0; j < origin[i].size(); j++) {
			pt3d[index++] = cv::Vec3f(
				(float)origin[i][j][0],
				(float)origin[i][j][1],
				(float)origin[i][j][2]
			);
		}
	}
}

template<typename DATTYPE>
static void Utility::nested4vec_2_Vec2i(
	const std::vector<std::vector<std::vector<std::vector<DATTYPE>>>>& origin,
	std::vector<cv::Vec2i>& pt2d
) {
	pt2d.clear();

	int index = 0;
	int pts_count = 0;
	for (int i = 0; i < origin.size(); i++) {
		for (int j = 0; j < origin[i].size(); j++) {
			pts_count += origin[i][j].size();
		}
	}
	pt2d.resize(pts_count);
	for (int i = 0; i < origin.size(); i++) {
		for (int j = 0; j < origin[i].size(); j++) {
			for (int k = 0; k < origin[i][j].size(); k++) {
				pt2d[index++] = cv::Vec2i(
					(int)origin[i][j][k][0],
					(int)origin[i][j][k][1]
				);
			}
		}
	}
}

template<typename DATTYPE>
static void Utility::nested3vec_2_Vec2i(
	const std::vector<std::vector<std::vector<DATTYPE>>>& origin,
	std::vector<cv::Vec2i>& pt2d
) {
	int index = 0;
	int pts_count = 0;
	for (int i = 0; i < origin.size(); i++) {
		pts_count += origin[i].size();
	}
	pt2d.resize(pts_count);
	for (int i = 0; i < origin.size(); i++) {
		for (int j = 0; j < origin[i].size(); j++) {
			pt2d[index++] = cv::Vec2i(
				(int)origin[i][j][0],
				(int)origin[i][j][1]
			);
		}
	}
}

template<typename DATTYPE>
static void Utility::nested2vec_2_Vec2i(
	const std::vector<std::vector<DATTYPE>>& origin,
	std::vector<cv::Vec2i>& pt2d
) {
	pt2d.resize(origin.size());
	for (int i = 0; i < origin.size(); i++) {
		pt2d[i] = cv::Vec2i(
			(int)origin[i][0],
			(int)origin[i][1]
		);
	}
}

std::string Utility::type2str(int type) {
	std::string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');
	return r;
}

bool Utility::hasEnding(const std::string& fullString, const std::string& ending) {
	if (fullString.length() >= ending.length()) {
		return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
	}
	else {
		return false;
	}
}

void Utility::seg_str_by_char(
	const std::string& t,
	const std::string& token,
	std::string& pre,
	std::string& post
) {
	std::size_t found = t.find_last_of(token);
	pre  = t.substr(0, found);
	post = t.substr(found + 1);
}

template<typename DATTYPE1, typename DATTYPE2>
void Utility::extract_beyond_or_below_thresh(
	const std::vector<std::vector<std::vector<std::vector<DATTYPE1>>>>& data,
	const std::vector<std::vector<DATTYPE2>>& reference,
	const double th,
	const int mode,
	std::vector<std::vector<std::vector<std::vector<DATTYPE1>>>>& res
) {
	res.clear();

	// mode 1: select beyond threshold; 2: select below threshold
	assert(data.size() == reference.size());
	const int dim = data.size();
	res.resize(dim);
	for (size_t i = 0; i < dim; i++) {
		assert(data[i].size() == reference[i].size());
		for (int j = 0; j < data[i].size(); j++) {
			if (mode == 1 && reference[i][j] >= th) {
				res[i].push_back(data[i][j]);
			}
			else if (mode == 2 && reference[i][j] <= th) {
				res[i].push_back(data[i][j]);
			}
		}
	}
}

template<typename DATTYPE1, typename DATTYPE2>
void Utility::extract_beyond_or_below_thresh(
	const std::vector<std::vector<DATTYPE1>>& data,
	const std::vector<std::vector<DATTYPE2>>& reference,
	const double th,
	const int mode,
	std::vector<std::vector<DATTYPE1>>& res
) {
	res.clear();

	// mode 1: select beyond threshold; 2: select below threshold
	assert(data.size() == reference.size());
	const int dim = data.size();
	res.resize(dim);
	for (size_t i = 0; i < dim; i++) {
		assert(data[i].size() == 2 * reference[i].size());
		for (int j = 0; j < reference[i].size(); j++) {
			if (mode == 1 && reference[i][j] >= th) {
				res[i].push_back(data[i][j * 2]);
				res[i].push_back(data[i][j * 2 + 1]);
			}
			else if (mode == 2 && reference[i][j] <= th) {
				res[i].push_back(data[i][j * 2]);
				res[i].push_back(data[i][j * 2 + 1]);
			}
		}
	}
}

template<typename DATTYPE>
void Utility::extract_by_num_range(
	const std::vector<std::vector<std::vector<std::vector<DATTYPE>>>>& data,
	const int lwr,
	const int upr,
	std::vector<std::vector<std::vector<std::vector<DATTYPE>>>>& res
) {
	res.clear();

	const int dim = data.size();
	res.resize(dim);
	for (size_t i = 0; i < dim; i++) {
		for (int j = 0; j < data[i].size(); j++) {
			if (data[i][j].size() >= lwr && data[i][j].size() <= upr) {
				res[i].push_back(data[i][j]);
			}
		}
	}
}

template<typename DATTYPE>
void Utility::sort_by_elenum(
	const std::vector<std::vector<std::vector<std::vector<DATTYPE>>>>& data,
	const bool ascend,
	std::vector<std::vector<std::vector<std::vector<DATTYPE>>>>& res
) {
	res.clear();

	const int dim = data.size();
	res.resize(dim);
	std::vector<int> worker;
	std::vector<std::vector<int>> idx_array(dim);
	for (size_t i = 0; i < dim; i++) {
		worker.resize(data[i].size());
		res[i].resize(data[i].size());
		for (int j = 0; j < data[i].size(); j++) {
			worker[j] = data[i][j].size();
		}
		idx_array[i] = sort_indices(worker, ascend);
		worker.clear();
	}
	for (size_t i = 0; i < dim; i++) {
		for (int j = 0; j < data[i].size(); j++) {
			res[i][j] = data[i][idx_array[i][j]];
		}
	}
}

template<typename DATTYPE>
std::vector<std::vector<DATTYPE>> Utility::cvt_BD2Pers(
	const std::vector<std::vector<DATTYPE>>& data
) {
	std::vector<std::vector<DATTYPE>> res(data.size());
	for (size_t i = 0; i < data.size(); i++) {
		assert(data[i].size() % 2 == 0);
		res[i].resize(data[i].size() / 2);
		for (int j = 0; j < res[i].size(); j++) {
			res[i][j] = data[i][j * 2 + 1] - data[i][j * 2];
		}
	}
	return res;
}

template<typename DATTYPE>
std::vector<std::vector<DATTYPE>> Utility::create_2D_array(
	const std::vector<int>& dims,
	const DATTYPE init_value
) {
	assert(dims.size() == 2);
	const int dim1 = dims[0], dim2 = dims[1];
	std::vector<std::vector<DATTYPE>> res(dim1);
	for (int i = 0; i < dim1; i++) {
		res[i].resize(dim2);
	}
	for (int i = 0; i < dim1; i++) {
		for (int j = 0; j < dim2; j++) {
			res[i][j] = init_value;
		}
	}
	return res;
}

template<typename DATTYPE>
std::vector<std::vector<std::vector<DATTYPE>>> Utility::create_3D_array(
	const std::vector<int>& dims,
	const DATTYPE init_value
) {
	assert(dims.size() == 3);
	const int dim1 = dims[0], dim2 = dims[1], dim3 = dims[2];
	std::vector<std::vector<std::vector<DATTYPE>>> res(dim1);
	for (int i = 0; i < dim1; i++) {
		res[i].resize(dim2);
		for (int j = 0; j < dim2; j++) {
			res[i][j].resize(dim3);
		}
	}
	for (int i = 0; i < dim1; i++) {
		for (int j = 0; j < dim2; j++) {
			for (int k = 0; k < dim3; k++) {
				res[i][j][k] = init_value;
			}
		}
	}
	return res;
}

template<typename DATTYPE>
void Utility::shift_mask(
	cv::Mat&   mask,
	const int  target_label,
	const int  assign_label,
	const int  target_dim,
	const int  shift_pix,
	const bool increase
) {
	assert(target_dim < mask.dims);
	const cv::MatSize size = mask.size;
	for (int i = 0; i < size[0]; i++)
		for (int j = 0; j < size[1]; j++)
			for (int k = 0; k < size[2]; k++)
				if (mask.at<DATTYPE>(i, j, k) == target_label) {
					std::vector<int> pos{ i, j, k };
					if (increase) pos[target_dim] += shift_pix;
					else pos[target_dim] -= shift_pix;

					// Validity check
					if (pos[target_dim] >= 0 && pos[target_dim] < size[target_dim])
						mask.at<DATTYPE>(pos[0], pos[1], pos[2]) = assign_label;
				}
}

template<typename DATTYPE>
cv::Mat Utility::retrieve_mat_by_mask_2D3D(
	const cv::Mat& data,
	const cv::Mat& mask,
	DATTYPE threshold,
	DATTYPE background_val,
	bool select_beyond_threshold
) {
	assert(data.dims == mask.dims);
	assert(data.size == mask.size);
	const int dims = data.dims;
	const cv::MatSize size = data.size;
	int* dim_layout = new int[dims];
	for (int i = 0; i < dims; i++) dim_layout[i] = size[i];
	cv::Mat res(dims, dim_layout, data.type(), cv::Scalar(0));

	if (dims == 2) {
		for (int i = 0; i < size[0]; i++)
			for (int j = 0; j < size[1]; j++)
				if (select_beyond_threshold)
					res.at<DATTYPE>(i, j) = (mask.at<DATTYPE>(i, j) >= threshold) ?
					data.at<DATTYPE>(i, j) : background_val;
				else
					res.at<DATTYPE>(i, j) = (mask.at<DATTYPE>(i, j) <= threshold) ?
					data.at<DATTYPE>(i, j) : background_val;
	}
	else if (dims == 3) {
		for (int i = 0; i < size[0]; i++)
			for (int j = 0; j < size[1]; j++)
				for (int k = 0; k < size[2]; k++)
					if (select_beyond_threshold)
						res.at<DATTYPE>(i, j, k) = (mask.at<DATTYPE>(i, j, k) >= threshold) ?
						data.at<DATTYPE>(i, j, k) : background_val;
					else
						res.at<DATTYPE>(i, j, k) = (mask.at<DATTYPE>(i, j, k) <= threshold) ?
						data.at<DATTYPE>(i, j, k) : background_val;
	}

	return res;
}

template<typename DATTYPE>
cv::Mat Utility::find_content_bound_3D(const cv::Mat& data, DATTYPE background_val) {
	const int dims = data.dims;
	const cv::MatSize size = data.size;
	assert(dims == 3);
	
	int l1_ = -1, u1_ = -1;
	// Search along the first axis for lower bound
	for (int i = 0; i < size[0]; i++) {
		for (int j = 0; j < size[1]; j++) {
			for (int k = 0; k < size[2]; k++) {
				if (data.at<DATTYPE>(i, j, k) > background_val) {
					l1_ = i;
					break;
				}
			}
			if (l1_ != -1) break;
		}
		if (l1_ != -1) break;
	}
	// Search along the first axis in backwards direction for upper bound
	for (int i = size[0] - 1; i >= 0 ; i--) {
		for (int j = 0; j < size[1]; j++) {
			for (int k = 0; k < size[2]; k++) {
				if (data.at<DATTYPE>(i, j, k) > background_val) {
					u1_ = i;
					break;
				}
			}
			if (u1_ != -1) break;
		}
		if (u1_ != -1) break;
	}

	int l2_ = -1, u2_ = -1;
	// Search along the second axis for lower bound
	for (int i = 0; i < size[1]; i++) {
		for (int j = 0; j < size[0]; j++) {
			for (int k = 0; k < size[2]; k++) {
				if (data.at<DATTYPE>(j, i, k) > background_val) {
					l2_ = i;
					break;
				}
			}
			if (l2_ != -1) break;
		}
		if (l2_ != -1) break;
	}
	// Search along the second axis in backwards direction for upper bound
	for (int i = size[1] - 1; i >= 0; i--) {
		for (int j = 0; j < size[0]; j++) {
			for (int k = 0; k < size[2]; k++) {
				if (data.at<DATTYPE>(j, i, k) > background_val) {
					u2_ = i;
					break;
				}
			}
			if (u2_ != -1) break;
		}
		if (u2_ != -1) break;
	}

	int l3_ = -1, u3_ = -1;
	// Search along the third axis for lower bound
	for (int i = 0; i < size[2]; i++) {
		for (int j = 0; j < size[0]; j++) {
			for (int k = 0; k < size[1]; k++) {
				if (data.at<DATTYPE>(j, k, i) > background_val) {
					l3_ = i;
					break;
				}
			}
			if (l3_ != -1) break;
		}
		if (l3_ != -1) break;
	}
	// Search along the third axis in backwards direction for upper bound
	for (int i = size[2] - 1; i >= 0; i--) {
		for (int j = 0; j < size[0]; j++) {
			for (int k = 0; k < size[1]; k++) {
				if (data.at<DATTYPE>(j, k, i) > background_val) {
					u3_ = i;
					break;
				}
			}
			if (u3_ != -1) break;
		}
		if (u3_ != -1) break;
	}

	// Improved version, pad one pixel to each dimension from both directions
	int dim_layout[3] = { u1_ - l1_ + 3, u2_ - l2_ + 3, u3_ - l3_ + 3 };
	cv::Mat res(dims, dim_layout, data.type(), cv::Scalar(background_val));
	for (int i = 1; i < res.size[0]-1; i++) {
		for (int j = 1; j < res.size[1]-1; j++) {
			for (int k = 1; k < res.size[2]-1; k++) {
				res.at<DATTYPE>(i, j, k) = data.at<DATTYPE>(i - 1 + l1_, j - 1 + l2_, k - 1 + l3_);
			}
		}
	}
	return res;
}

// ===== DMT code sections =====
std::vector<cv::Vec3f> Utility::retrieve_DMT_cycles_by_index(
	std::vector<cv::Point3i>& coord,
	std::vector<std::vector<int>>& cycs
) {
	std::vector<cv::Vec3f> res;
	const int cyc_num = cycs.size();
	for (int i = 0; i < cyc_num; i++) {
		for (int j = 0; j < cycs[i].size(); j++) {
			const cv::Point3i& t = coord[cycs[i][j]];
			res.push_back(cv::Vec3f(t.x, t.y, t.z));
		}
	}

	return res;
}
// ==============================

// Template instantiation
template std::string Utility::tostr<int>(const int& t);
template std::string Utility::tostr<float>(const float& t);
template std::string Utility::tostr<double>(const double& t);

template std::vector<int> Utility::sort_indices<int>(
	const std::vector<int>&,
	const bool);
template std::vector<int> Utility::sort_indices<float>(
	const std::vector<float>&,
	const bool);
template std::vector<int> Utility::sort_indices<double>(
	const std::vector<double>&,
	const bool);

template void Utility::nested4vec_2_Vec3f<int>(
	const std::vector<std::vector<std::vector<std::vector<int>>>>&,
	std::vector<cv::Vec3f>&);
template void Utility::nested4vec_2_Vec3f<float>(
	const std::vector<std::vector<std::vector<std::vector<float>>>>&,
	std::vector<cv::Vec3f>&);
template void Utility::nested4vec_2_Vec3f<double>(
	const std::vector<std::vector<std::vector<std::vector<double>>>>&,
	std::vector<cv::Vec3f>&);

template void Utility::nested3vec_2_Vec3f<int>(
	const std::vector<std::vector<std::vector<int>>>&,
	std::vector<cv::Vec3f>&);
template void Utility::nested3vec_2_Vec3f<float>(
	const std::vector<std::vector<std::vector<float>>>&,
	std::vector<cv::Vec3f>&);
template void Utility::nested3vec_2_Vec3f<double>(
	const std::vector<std::vector<std::vector<double>>>&,
	std::vector<cv::Vec3f>&);

template void Utility::nested4vec_2_Vec2i<int>(
	const std::vector<std::vector<std::vector<std::vector<int>>>>&,
	std::vector<cv::Vec2i>&
	);

template void Utility::nested3vec_2_Vec2i<int>(
	const std::vector<std::vector<std::vector<int>>>&,
	std::vector<cv::Vec2i>&
	);

template void Utility::nested2vec_2_Vec2i<int>(
	const std::vector<std::vector<int>>&,
	std::vector<cv::Vec2i>&
	);

template void Utility::extract_beyond_or_below_thresh<int, double>(
	const std::vector<std::vector<std::vector<std::vector<int>>>>&,
	const std::vector<std::vector<double>>&,
	const double,
	const int,
	std::vector<std::vector<std::vector<std::vector<int>>>>&
	);

template void Utility::extract_beyond_or_below_thresh<double, double>(
	const std::vector<std::vector<double>>&,
	const std::vector<std::vector<double>>&,
	const double,
	const int,
	std::vector<std::vector<double>>&
	);

template void Utility::extract_by_num_range<int>(
	const std::vector<std::vector<std::vector<std::vector<int>>>>&,
	const int,
	const int,
	std::vector<std::vector<std::vector<std::vector<int>>>>&
	);

template void Utility::sort_by_elenum<int>(
	const std::vector<std::vector<std::vector<std::vector<int>>>>&,
	const bool,
	std::vector<std::vector<std::vector<std::vector<int>>>>&
	);

template std::vector<std::vector<double>>  Utility::cvt_BD2Pers<double>(
	const std::vector<std::vector<double>>& data
);

template std::vector<std::vector<double>> Utility::create_2D_array<double>(
	const std::vector<int>& dims,
	const double init_value
);

template std::vector<std::vector<std::vector<double>>> Utility::create_3D_array<double>(
	const std::vector<int>& dims,
	const double init_value
);

template void Utility::shift_mask<float>(
	cv::Mat&   mask,
	const int  target_label,
	const int  assign_label,
	const int  target_dim,
	const int  shift_pix,
	const bool increase
);

template void Utility::shift_mask<double>(
	cv::Mat&   mask,
	const int  target_label,
	const int  assign_label,
	const int  target_dim,
	const int  shift_pix,
	const bool increase
	);

template cv::Mat Utility::retrieve_mat_by_mask_2D3D<float>(
	const cv::Mat& data,
	const cv::Mat& mask,
	float threshold,
	float background_val,
	bool select_beyond_threshold
);

template cv::Mat Utility::retrieve_mat_by_mask_2D3D<double>(
	const cv::Mat& data,
	const cv::Mat& mask,
	double threshold,
	double background_val,
	bool select_beyond_threshold
);

template cv::Mat Utility::find_content_bound_3D<double>(const cv::Mat& data, double background_val);

template cv::Mat Utility::find_content_bound_3D<float>(const cv::Mat& data, float background_val);