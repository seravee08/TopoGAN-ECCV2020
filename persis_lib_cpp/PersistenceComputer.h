#ifndef PERSISTENCE_COMPUTER
#define PERSISTENCE_COMPUTER

#include <iostream>
#include <cstring>
#include <vector>
#include <opencv2/opencv.hpp>

#include "InputFileInfo.h"

class Persistence_Computer {
public:
	Persistence_Computer() { output_name = ""; debug_enabled = false; debug_path = "."; }
	~Persistence_Computer() {}

	double run();
	void source_from_file(const std::string& input_file, const std::string& output_file = "");

	void source_from_mat(const std::string& output_file, const cv::Mat& t);

	void source_from_mat_from_int(const std::string& output_file, std::vector<int>& t, const int height, const int width);

	void source_from_mat_from_double(const std::string& output_file, std::vector<double>& t, const int height, const int width);
	
	void set_output_file(const std::string& t);
	void set_pers_thd(double t);
	void set_algorithm(int t);
	void set_max_dim(int t);
	void set_num_threads(int t);
	void set_verbose(bool t);
	void set_debug(bool t, const std::string& debug_path_=".");

	void write_bnd(const std::vector<std::vector<std::vector<std::vector<int>>>>& t);
	void write_red(const std::vector<std::vector<std::vector<std::vector<int>>>>& t);
	void write_pers_V(const std::vector<std::vector<std::vector<int>>>& pers_V);
	void write_pers_BD(const std::vector<std::vector<std::vector<double>>>& pers_BD);
	void write_pers_BD(const std::vector<std::vector<double>>& pers_BD);
	void return_bnd(std::vector<std::vector<std::vector<std::vector<int>>>>& t);
	void return_red(std::vector<std::vector<std::vector<std::vector<int>>>>& t);
	void return_pers_V(std::vector<std::vector<std::vector<int>>>& t);
	void return_pers_BD(std::vector<std::vector<std::vector<double>>>& t);

	void write_output();
	void clear();
	static void debugStart(const std::string& debug_path);
	static void debugEnd();

private:

	bool debug_enabled;
	std::string debug_path;
	std::string output_name;
	InputFileInfo file_info;

	std::vector<std::vector<std::vector<std::vector<int>>>> final_red_list_grand;
	std::vector<std::vector<std::vector<std::vector<int>>>> final_boundary_list_grand;
	std::vector<std::vector<std::vector<int>>> pers_V;
	std::vector<std::vector<std::vector<double>>> pers_BD;
};

#endif // !PERSISTENCE_COMPUTER