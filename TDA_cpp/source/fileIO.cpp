#include "fileIO.h"
#include "utility.h"
#include <numeric>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

void fileIO::Mat2Dat(const cv::Mat& t, const std::string& file_path) {
	// This function is modified to include fileType <from pengxiang's optimal cycle>
	// The image data's fileType is 0

	if (t.channels() != 1) {
		std::cout << "1 dimension data required ..." << std::endl;
		exit(1);
	}
	cv::Mat t_double;
	t.convertTo(t_double, CV_64F);

	// Prepare header
	int fileType = 0;  // https://github.com/pxiangwu/Persistent-Homology-Localization-Algorithms
	int dims = t.dims;
	cv::MatSize size = t.size;
	std::vector<unsigned int> dim_layout(dims);
	for (int i = 0; i < dims; i++) dim_layout[i] = (unsigned int)size[i];
	int prod = std::accumulate(std::begin(dim_layout), std::end(dim_layout), 1, std::multiplies<int>());

	// Open file stream
	std::string output_path = file_path + ".dat";
	std::ofstream out(output_path.c_str(), std::ios::binary | std::ios::out);
	out.write(reinterpret_cast<char*>(&fileType), sizeof(int));
	out.write(reinterpret_cast<char*>(&dims), sizeof(int));
	out.write(reinterpret_cast<char*>(dim_layout.data()), sizeof(unsigned int) * dim_layout.size());
	out.write(reinterpret_cast<char*>(t_double.data), sizeof(double) * prod);
	out.close();
}

void fileIO::generate_filelist(
	const std::string& path,
	const std::string& post_fix
) {
	std::vector<std::string> res;
	for (auto& entry : boost::make_iterator_range(
		boost::filesystem::directory_iterator(path), {})) {
		if (Utility::hasEnding(entry.path().string(), post_fix)) {
			res.push_back(entry.path().string());
		}
	}
	// Write out the file list
	std::string output = (Utility::hasEnding(path, "/")) ?
		path + "file_list.txt" : path + "/file_list.txt";
	std::fstream out(output, std::fstream::out | std::fstream::trunc);
	for (int i = 0; i < res.size(); i++) {
		out << res[i] << std::endl;
	}
	out.close();
}

// ========================================== //
// ===== Read functions =====

std::vector<std::string> fileIO::read_filelist(const std::string& path) {
	std::string line;
	std::ifstream in(path);
	std::vector<std::string> res;
	while (getline(in, line))
	{
		res.push_back(line);
	}
	in.close();
	return res;
}

cv::Mat fileIO::read_DAT_PD(
	const std::string& input_name
) {
	// This function is modified to include fileType <from pengxiang's optimal cycle>
	// The image data's fileType is 0
	std::string input_path = (Utility::hasEnding(input_name, ".dat")) ? input_name : input_name + ".dat";
	std::ifstream in(input_path.c_str(), std::ios::binary | std::ios::in);
	if (!in.good()) { std::cout << "read_DAT_PD: open failure" << std::endl; exit(1); }

	int fileType, dim; // https://github.com/pxiangwu/Persistent-Homology-Localization-Algorithms
	in.read(reinterpret_cast<char*>(&fileType), sizeof(int));
	in.read(reinterpret_cast<char*>(&dim), sizeof(int));
	std::vector<unsigned int> dim_layout(dim);
	in.read(reinterpret_cast<char*>(dim_layout.data()), sizeof(unsigned int) * dim);
	int* dim_layout_arr = new int[dim];
	std::copy(dim_layout.begin(), dim_layout.end(), dim_layout_arr);
	cv::Mat res(dim, dim_layout_arr, CV_64F); // High dimensional mat, cannot use .rows/.cols, use .size instead
	in.read(reinterpret_cast<char*>(res.data), sizeof(double) * 
		std::accumulate(std::begin(dim_layout), std::end(dim_layout), 1, std::multiplies<int>()));
	in.close();
	delete dim_layout_arr;
	return res;
}

std::vector<std::vector<std::vector<std::vector<int>>>> fileIO::read_BNDorRED(
	const std::string& input_name,
	const std::string& file_format
) {
	std::string input_path;
	if (file_format == "bnd" || file_format == "red") {
		input_path = (Utility::hasEnding(input_name, ".dat")) ?
			input_name + "." + file_format : input_name + ".dat." + file_format;
	}
	else {
		std::cout << "Reqeusted file format invalid ..." << std::endl;
		exit(1);
	}

	FILE *infile;
	infile = fopen(input_path.c_str(), "rb");
	unsigned int dim;
	fread((void*)&dim, sizeof(unsigned int), 1, infile);
	std::vector<unsigned int> dim_layout(dim);
	fread((void*)&dim_layout[0], sizeof(unsigned int), dim, infile);

	std::vector<unsigned int> struct_header(dim);
	std::vector<std::vector<std::vector<std::vector<int>>>> res(dim);
	for (size_t i = 0; i < dim; i++) {
		res[i].resize(dim_layout[i]);
		for (int j = 0; j < dim_layout[i]; j++) {
			fread((void*)&struct_header[0], sizeof(unsigned int), dim, infile);
			assert(struct_header[0] > 0);
			res[i][j].resize(struct_header[0]);
			std::vector<unsigned int> tmp_vec(struct_header[0] * dim);
			fread((void*)&tmp_vec[0], sizeof(unsigned int), struct_header[0] * dim, infile);
			for (int k = 0; k < struct_header[0]; k++) {
				res[i][j][k].resize(dim);
				for (size_t l = 0; l < dim; l++) {
					res[i][j][k][l] = tmp_vec[k * dim + l];
				}
			}
		}
	}
	fclose(infile);
	return res;
}

std::vector<std::vector<double>> fileIO::read_pers_BD(const std::string& input_name) {

	std::vector<double> worker;
	std::vector<std::vector<double>> res;
	std::string path = (Utility::hasEnding(input_name, ".dat")) ? 
		input_name + ".pers.txt" : input_name + ".dat.pers.txt";
	std::string word;
	std::ifstream in(path);
	int ele_num, birth, death;
	std::string names[] = { "Vertex", "Edge", "Face", "Cube", "4D-Cell", "5D-Cell" };
	int counter = 0;

	while (!in.eof()) {
		in >> word;
		if (word != names[counter++]) break;
		for (size_t i = 0; i < 4; i++) in >> word;
		in >> word;
		ele_num = std::stoi(word);

		worker.resize(2 * ele_num);
		for (int i = 0; i < ele_num; i++) {
			in >> word;
			birth = std::stoi(word);
			in >> word;
			death = std::stoi(word);
			worker[i * 2]     = birth;
			worker[i * 2 + 1] = death;
		}
		res.push_back(worker);
	}

	in.close();
	return res;
}

void fileIO::read_DMT_vertices3D(
	const std::string& input_name,
	std::vector<cv::Point3i>& coord,
	std::vector<float>& f_vals
) {
	coord.clear();
	f_vals.clear();
	std::ifstream in(input_name);
	std::string word1, word2, word3, word4;
	while (!in.eof()) {
		in >> word1 >> word2 >> word3 >> word4;
		cv::Point3i t(std::stoi(word1), std::stoi(word2), std::stoi(word3));
		coord.push_back(t);
		f_vals.push_back(float(atof(word4.c_str())));
	}
	in.close();
}

void fileIO::read_DMT_cycles(
	const std::string& input_name,
	std::vector<std::vector<int>>& cycs,
	std::vector<float>& pers
) {
	cycs.clear();
	pers.clear();
	std::ifstream in(input_name);
	std::string line, word;
	while (std::getline(in, line))
	{
		std::stringstream linestream(line);
		std::getline(linestream, word, ' ');
		pers.push_back(float(atof(word.c_str())));
		std::vector<int> cur_coord;
		while (!linestream.eof()) {
			std::getline(linestream, word, ' ');
			cur_coord.push_back(std::stoi(word));
		}
		cycs.push_back(cur_coord);
	}
	in.close();
}

// ========================================== //
// ===== Write functions =====
void fileIO::write_image(
	const cv::Mat& t,
	const std::string& name
) {
	cv::imwrite(name, t);
}

void fileIO::write_PD(const cv::Mat& t, const std::string& path) {
	int dims = t.channels();
	int instance_num = t.rows;
	assert(instance_num == t.cols);

	std::vector<cv::Mat> singleton(dims);
	cv::split(t, singleton);

	// Open file stream
	std::string rec_path = (Utility::hasEnding(path, "/")) ?
		path + "PD.dat" : path + "/PD.dat";
	std::ofstream out(rec_path.c_str(), std::ios::binary | std::ios::out);
	out.write(reinterpret_cast<char*>(&dims), sizeof(int));
	out.write(reinterpret_cast<char*>(&instance_num), sizeof(int));

	for (size_t i = 0; i < dims; i++) {
		out.write(reinterpret_cast<char*>(singleton[i].data), sizeof(double) * instance_num * instance_num);

	}
	out.close();
}

template<typename DATTYPE>
void fileIO::write_vec2txt(
	const std::vector<DATTYPE>& t,
	const std::string& out_path
) {
	// Write out the vector
	std::fstream out(out_path, std::fstream::out | std::fstream::trunc);
	for (int i = 0; i < t.size(); i++) {
		out << t[i] << std::endl;
	}
	out.close();
}

// Template instantiation
template void fileIO::write_vec2txt<int>(
	const std::vector<int>& t,
	const std::string& out_path);
template void fileIO::write_vec2txt<float>(
	const std::vector<float>& t,
	const std::string& out_path);
template void fileIO::write_vec2txt<double>(
	const std::vector<double>& t,
	const std::string& out_path);

// ========================================== //
// ===== MNIST functions =====
void mnist::extract_generate_labs(const std::string& dest) {
	// Assuming folder structures: root/training and root/testing
	// under each foler, there are 0 - 9, 10 folders

	// Check if the two folders exist and create destination folder if not exsist
	std::string train = (Utility::hasEnding(root, "/")) ?
		root + "training" : root + "/training";
	std::string test  = (Utility::hasEnding(root, "/")) ?
		root + "testing" : root + "/testing";
	if (!boost::filesystem::is_directory(train) ||
		!boost::filesystem::is_directory(test)) {
		std::cout << "Invalid folder structures ..." << std::endl;
		exit(1);
	}
	if (!boost::filesystem::is_directory(dest)) {
		boost::filesystem::create_directory(dest);
	}

	long int counter = 0;
	std::vector<int> labels;
	for (int i = 0; i <= 9; i++) {
		std::string digit_folder_train = train + "/" + std::to_string(i);
		std::string digit_folder_test  = test + "/" + std::to_string(i);
		if (!boost::filesystem::is_directory(digit_folder_train) ||
			!boost::filesystem::is_directory(digit_folder_test)) {
			std::cout << "Digit folder missing ..." << std::endl;
			exit(1);
		}
		for (auto& entry : boost::make_iterator_range(
			boost::filesystem::directory_iterator(digit_folder_train), {})) {
			boost::filesystem::copy_file(entry.path().string(), dest+"/"+std::to_string(counter++)+".png",
				boost::filesystem::copy_option::fail_if_exists);
			labels.push_back(i);
		}
		for (auto& entry : boost::make_iterator_range(
			boost::filesystem::directory_iterator(digit_folder_test), {})) {
			boost::filesystem::copy_file(entry.path().string(), dest + "/"+std::to_string(counter++)+".png",
				boost::filesystem::copy_option::fail_if_exists);
			labels.push_back(i);
		}
	}

	// Create label.txt file
	std::fstream out(dest+"/labels.txt", std::fstream::out | std::fstream::trunc);
	for (long int i = 0; i < labels.size(); i++) {
		out << labels[i] << std::endl;
	}
	out.close();
}