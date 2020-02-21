#pragma once

#ifndef FILEIO_H
#define FILEIO_H

#include <opencv2/opencv.hpp>

class fileIO {
public:
	fileIO() {};

	~fileIO() {};

	static void Mat2Dat(const cv::Mat& t, const std::string& file_path);

	static cv::Mat read_DAT_PD(const std::string& input_name);

	static std::vector<std::string> read_filelist(const std::string& path);

	static void generate_filelist(
		const std::string& path,
		const std::string& post_fix
	);

	static std::vector<std::vector<std::vector<std::vector<int>>>> read_BNDorRED(
		const std::string& input_name,
		const std::string& file_format
	);
	
	static std::vector<std::vector<double>> read_pers_BD(const std::string& input_name);

	// DMT: Discrete morse theory
	static void read_DMT_vertices3D(
		const std::string& input_name,
		std::vector<cv::Point3i>& coord,
		std::vector<float>& f_vals
	);

	static void read_DMT_cycles(
		const std::string& input_name,
		std::vector<std::vector<int>>& cycs,
		std::vector<float>& pers
	);

	static void write_image(
		const cv::Mat& t,
		const std::string& name
	);

	static void write_PD(
		const cv::Mat& t,
		const std::string& path
	);

	template<typename DATTYPE>
	static void write_vec2txt(
		const std::vector<DATTYPE>& t,
		const std::string& out_path
	);
};

class mnist {
public: 
	mnist(const std::string& root_) { root = root_; }

	~mnist() {}

	void extract_generate_labs(const std::string& dest);

private:
	std::string root;
};

#endif //!FILEIO_H