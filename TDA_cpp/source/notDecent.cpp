#include "PersistenceComputer.h"
#include "notDecent.h"
#include "viewer.h"
#include "utility.h"
#include "fileIO.h"
#include "vtkIO.h"
#include "editor.h"
#include "sw_distance.h"

#include <ctime>
#include <cstdlib>

#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>

#include <boost/filesystem.hpp>

using namespace std;
using namespace cv;

void mass_Mat2Dat(const string& path, const string& post_fix) {

	std::string flist_path = (Utility::hasEnding(path, "/")) ?
		path + "file_list.txt" : path + "/file_list.txt";
	if (!boost::filesystem::exists(flist_path)) {
		fileIO::generate_filelist(path, post_fix);
	}
	// Read in file_list file
	vector<string> flist = fileIO::read_filelist(flist_path);
	for (int i = 0; i < flist.size(); i++) {
		Mat img = imread(flist[i], IMREAD_GRAYSCALE);
		fileIO::Mat2Dat(img, flist[i]);
	}
}

void mass_Dat2PD(const string& path, const string& post_fix, int ind_s, int ind_f) {
	/*post_fix: the post fix in the file_list!!! */

	std::string flist_path = (Utility::hasEnding(path, "/")) ?
		path + "file_list.txt" : path + "/file_list.txt";
	if (!boost::filesystem::exists(flist_path)) {
		fileIO::generate_filelist(path, post_fix);
	}
	// Read in file_list file
	vector<string> flist = fileIO::read_filelist(flist_path);
	int start_ind = std::max(ind_s, 0);
	int finish_ind = std::min(ind_f, int(flist.size()));

	Persistence_Computer pc;
	pc.set_pers_thd(0.0);
	for (int i = start_ind; i < finish_ind; i++) {
		(post_fix != ".dat") ? pc.source_from_file(flist[i] + ".dat") :
			pc.source_from_file(flist[i]);
		pc.run();
		pc.write_output();
		pc.clear();
	}
}

void single_visualize3D(const string& path, const string& file_format) {
	Viewer viewer;
	std::vector<cv::Vec3f> pt3d;
	std::vector<std::vector<std::vector<std::vector<int>>>> grand_list;
	grand_list = fileIO::read_BNDorRED(path, file_format);
	Utility::nested4vec_2_Vec3f(grand_list, pt3d);
	viewer.show_3dcloud(pt3d);
}

void mass_visualize2D(const string& path, const string& post_fix, const string& file_format) {

	std::string flist_path = (Utility::hasEnding(path, "/")) ?
		path + "file_list.txt" : path + "/file_list.txt";
	if (!boost::filesystem::exists(flist_path)) {
		fileIO::generate_filelist(path, post_fix);
	}
	// Read in file_list file
	vector<string> flist = fileIO::read_filelist(flist_path);
	vector<Vec2i> pt2d;
	Viewer viz;
	vector<vector<vector<vector<int>>>> grand_list, filtered_list1, filtered_list2;
	vector<vector<double>> pers_BD, pers_Time;
	for (int i = 0; i < 10; i++) {
		cout << flist[i] << endl;
		Mat img = imread(flist[i], IMREAD_GRAYSCALE);
		grand_list = fileIO::read_BNDorRED(flist[i], file_format);
		pers_BD = fileIO::read_pers_BD(flist[i]);
		pers_Time = Utility::cvt_BD2Pers(pers_BD);
		Utility::extract_beyond_or_below_thresh(grand_list, pers_Time, 10, 1, filtered_list1);
		//Utility::extract_by_num_range(filtered_list1, 0, 330000, filtered_list2);
		Utility::sort_by_elenum(filtered_list1, false, filtered_list2);
		Utility::nested2vec_2_Vec2i(filtered_list2[1][0], pt2d);
		Mat masked = viz.show_2dmask(pt2d, img, true, 640, 640);
		//masked = Editor::edit(masked, img);
		//std::string pre, post;
		//Utility::seg_str_by_char(flist[i], ".", pre, post);
		//fileIO::write_image(masked, pre + "_edited.png");
	}
}

void mass_compute_slicedWasserstein(
	const string& path,
	const string& post_fix,
	const bool approx,
	const int slices
) {

	std::string flist_path = (Utility::hasEnding(path, "/")) ?
		path + "file_list.txt" : path + "/file_list.txt";
	if (!boost::filesystem::exists(flist_path)) {
		fileIO::generate_filelist(path, post_fix);
	}
	// Read in file_list file
	vector<string> flist = fileIO::read_filelist(flist_path);
	const int instance_num = flist.size();
	vector<vector<vector<double>>> pers_BD_list(instance_num);
	for (int i = 0; i < instance_num; i++) {
		pers_BD_list[i] = fileIO::read_pers_BD(flist[i]);
		vector<double> tmp = pers_BD_list[i][0];
		for (size_t j = 1; j < pers_BD_list[i].size(); j++) {
			tmp.insert(tmp.end(), pers_BD_list[i][j].begin(), pers_BD_list[i][j].end());
		}
		pers_BD_list[i].push_back(tmp);
	}
	
	Mat dis_matrix = sliced_wassasertein_distance::compute_SW_matrix(pers_BD_list, approx, slices);
	fileIO::write_PD(dis_matrix, path);
}

void mass_extract_ABIDE_labels(
	const string& path,
	const string& post_fix,
	const string& out_path
) {
	std::string flist_path = (Utility::hasEnding(path, "/")) ?
		path + "file_list.txt" : path + "/file_list.txt";
	if (!boost::filesystem::exists(flist_path)) {
		fileIO::generate_filelist(path, post_fix);
	}
	// Read in file_list file
	vector<string> flist = fileIO::read_filelist(flist_path);
	const int instance_num = flist.size();
	string pre, post;
	vector<int> res(instance_num);
	for (int i = 0; i < instance_num; i++) {
		Utility::seg_str_by_char(flist[i], ".", pre, post);
		res[i] = pre[pre.length() - 1] - '0';
	}
	fileIO::write_vec2txt(res, out_path);
}

void mass_filter_PD(
	const string& in_folder,
	const string& out_folder,
	const double threshold,
	const string& post_fix
) {
	std::string flist_path = (Utility::hasEnding(in_folder, "/")) ?
		in_folder + "file_list.txt" : in_folder + "/file_list.txt";
	if (!boost::filesystem::exists(flist_path)) {
		fileIO::generate_filelist(in_folder, post_fix);
	}
	// Read in file_list file
	string pre, post;
	Persistence_Computer pc;
	vector<string> flist = fileIO::read_filelist(flist_path);
	vector<vector<double>> pers_BD, filtered_pers_BD, pers_Time;
	vector<vector<vector<vector<int>>>> red, bnd, filtered_red, filtered_bnd;

	for (int i = 0; i < flist.size(); i++) {
		Utility::seg_str_by_char(flist[i], "\\", pre, post);
		string out_path = (Utility::hasEnding(out_folder, "/")) ?
			out_folder + post + ".dat" : out_folder + "/" + post + ".dat";
		pc.set_output_file(out_path);
	
		pers_BD = fileIO::read_pers_BD(flist[i]);
		pers_Time = Utility::cvt_BD2Pers(pers_BD);
		bnd = fileIO::read_BNDorRED(flist[i], "bnd");
		red = fileIO::read_BNDorRED(flist[i], "red");
		Utility::extract_beyond_or_below_thresh(pers_BD, pers_Time, threshold, 1, filtered_pers_BD);
		Utility::extract_beyond_or_below_thresh(bnd, pers_Time, threshold, 1, filtered_bnd);
		Utility::extract_beyond_or_below_thresh(red, pers_Time, threshold, 1, filtered_red);

		pc.write_pers_BD(filtered_pers_BD);
		pc.write_bnd(filtered_bnd);
		pc.write_red(filtered_red);
	}
}

void mass_extract_patch_dual(
	const std::string& segmentation_folder,
	const std::string& segmentation_extension,
	const std::string& texture_folder,
	const std::string& texture_extension,
	const std::string& segres_folder,
	const std::string& segres_base_name,
	const std::string& segres_extension,
	const std::string& texres_folder,
	const std::string& texres_base_name,
	const std::string& texres_extension,
	const int patch_per_image,
	const int target_width,
	const int target_height,
	const int boundary_thickness,
	const int boundary_intensity,
	const int resize_width,
	const int resize_height,
	const int channels,
	const float pass_if_beyond,
	bool binarize
) {
	/* =====
	1. Assume segmentation_folder and texture_folder have files with the same names
	   but not necessarily the same extension.
	2. Segmentation should always be in grayscale while texture could be RGB OR GRAYSCALE
	3. Segmentation should have value range from 0 to 255
	  */
	std::string flist_path = (Utility::hasEnding(segmentation_folder, "/")) ?
		segmentation_folder + "file_list.txt" : segmentation_folder + "/file_list.txt";
	if (!boost::filesystem::exists(flist_path)) {
		fileIO::generate_filelist(segmentation_folder, segmentation_extension);
	}
	std::string segres_template = (Utility::hasEnding(segres_folder, "/")) ?
		segres_folder + segres_base_name + "_" : segres_folder + "/" + segres_base_name + "_";
	std::string texres_template = (Utility::hasEnding(texres_folder, "/")) ?
		texres_folder + texres_base_name + "_" : texres_folder + "/" + texres_base_name + "_";

	// Read in file_list file
	srand(time(NULL));
	vector<string> flist = fileIO::read_filelist(flist_path);
	const int instance_num = flist.size();
	int global_counter = 0;
	for (int i = 0; i < instance_num; i++) {
		string pre, post;
		Utility::seg_str_by_char(flist[i], "\\", pre, post);
		string name = post.substr(0, post.length() - 4);
		string composed_tx_name = (Utility::hasEnding(texture_folder, "/")) ?
			texture_folder + name + texture_extension : texture_folder + "/" + name + texture_extension;

		Mat segimg = imread(flist[i], IMREAD_GRAYSCALE);
		Mat teximg = (channels == 1) ? imread(composed_tx_name, IMREAD_GRAYSCALE) : imread(composed_tx_name, IMREAD_COLOR);
		for (int j = 0; j < patch_per_image; j++) {
			Mat patcha, patchb;
			Viewer::random_sample_patch_dual(segimg, teximg, patcha, patchb, target_height, target_width);
			if (pass_if_beyond > 0) {
				int lower_pix_count = 0;
				for (int i_ = 0; i_ < target_height; i_++) {
					for (int j_ = 0; j_ < target_width; j_++)
						if (int(patcha.at<uchar>(i_, j_)) < 127.5) lower_pix_count++;
				}
				if (lower_pix_count * 1.0 / (target_height * target_width) < pass_if_beyond) {
					--j;
					continue;
				}
			}
			if (resize_width > 0 && resize_height > 0) {
				resize(patcha, patcha, cv::Size(resize_width, resize_height));
				resize(patchb, patchb, cv::Size(resize_width, resize_height));
			}
			if (binarize) {
				assert(patchb.channels() == 1);
				patcha = Viewer::binarize_img(patcha, (uchar)250, (uchar)255, (uchar)0);
				patchb = Viewer::binarize_img(patchb, (uchar)250, (uchar)255, (uchar)0);
			}
			Viewer::color_boundary_gray(patcha, boundary_thickness, boundary_intensity);
			Viewer::color_boundary_gray(patchb, boundary_thickness, boundary_intensity);
			string instance_name = to_string(global_counter);
			while (instance_name.length() < 5) {
				instance_name = "0" + instance_name;
			}
			string segname = segres_template + instance_name + segres_extension;
			string texname = texres_template + instance_name + texres_extension;
			
			imwrite(segname, patcha);
			imwrite(texname, patchb);
			global_counter++;
		}
	}
}

void mass_extract_patch_solo(
	const std::string& source_folder,
	const std::string& source_extension,
	const std::string& target_folder,
	const std::string& target_base_name,
	const std::string& target_extension,
	const int patch_per_image,
	const int target_width,
	const int target_height,
	const int boundary_thickness,
	const int boundary_intensity,
	const int resize_width,
	const int resize_height,
	const int channels,
	const float pass_if_beyond,
	bool binarize
) {
	std::string flist_path = (Utility::hasEnding(source_folder, "/")) ?
		source_folder + "file_list.txt" : source_folder + "/file_list.txt";
	if (!boost::filesystem::exists(flist_path)) {
		fileIO::generate_filelist(source_folder, source_extension);
	}
	std::string output_template = (Utility::hasEnding(target_folder, "/")) ?
		target_folder + target_base_name + "_" : target_folder + "/" + target_base_name + "_";

	// Read in file_list file
	srand(time(NULL));
	vector<string> flist = fileIO::read_filelist(flist_path);
	const int instance_num = flist.size();
	int global_counter = 0;
	for (int i = 0; i < instance_num; i++) {
		Mat img = (channels == 1) ? imread(flist[i], IMREAD_GRAYSCALE) : imread(flist[i], IMREAD_COLOR);
		string pre, post;
		Utility::seg_str_by_char(flist[i], "\\", pre, post);
		for (int j = 0; j < patch_per_image; j++) {
			Mat patch;
			Viewer::random_sample_patch_solo(img, patch, target_height, target_width);
			if (pass_if_beyond > 0) {
				Mat gray_tmp;
				if (channels != 1) cvtColor(patch, gray_tmp, cv::COLOR_BGR2GRAY);
				else gray_tmp = patch;
				int lower_pix_count = 0;
				for (int i_ = 0; i_ < target_height; i_++) {
					for (int j_ = 0; j_ < target_width; j_++)
						if (int(gray_tmp.at<uchar>(i_, j_)) < 127.5) lower_pix_count++;
				}
				if (lower_pix_count * 1.0 / (target_height * target_width) < pass_if_beyond) {
					--j;
					continue;
				}
			}
			if (resize_width > 0 && resize_height > 0)
				resize(patch, patch, cv::Size(resize_width, resize_height));
			if (binarize) {
				assert(patch.channels() == 1);
				patch = Viewer::binarize_img(patch, (uchar)250, (uchar)255, (uchar)0);
			}
			Viewer::color_boundary_gray(patch, boundary_thickness, boundary_intensity);
			string instance_name = to_string(global_counter);
			while (instance_name.length() < 5) {
				instance_name = "0" + instance_name;
			}
			instance_name = output_template + instance_name + target_extension;

			imwrite(instance_name, patch);
			global_counter++;
		}
	}
}

void mass_binarize_grayimg(
	const std::string& source_folder,
	const std::string& source_extension,
	const std::string& target_folder
) {
	std::string flist_path = (Utility::hasEnding(source_folder, "/")) ?
		source_folder + "file_list.txt" : source_folder + "/file_list.txt";
	if (!boost::filesystem::exists(flist_path)) {
		fileIO::generate_filelist(source_folder, source_extension);
	}
	std::string out_folder = (Utility::hasEnding(target_folder, "/")) ?
		target_folder : target_folder + "/";

	vector<string> flist = fileIO::read_filelist(flist_path);
	const int instance_num = flist.size();
	string seg_pre, seg_post;
	for (int i = 0; i < instance_num; i++) {
		Utility::seg_str_by_char(flist[i], "\\", seg_pre, seg_post);
		string instance_name = out_folder + seg_post;
		Mat img = imread(flist[i], IMREAD_GRAYSCALE);
		img = Viewer::inv_intensity(img, (uchar)255);
		img = Viewer::binarize_img(img, (uchar)125, (uchar)255, (uchar)0);
		imwrite(instance_name, img);
	}
}

void mass_extract_segmentation_vertebral_bone(
	const std::string& mask_folder,
	const std::string& data_folder,
	const std::string& out_folder,
	const std::string& mode,
	const double bacgnd_value
) {
	// mode: "sub" or "sup"
	vector<int>   phase1_density = { 54, 60, 65, 85, 100, 140, 200 };
	vector<float> phase2_density = { 43.8f, 46.7f, 64.9f, 75.8f, 76.8f, 89.5f, 94.4f, 99.7f, 106.5f, 145.4f, 148.4f, 169.6f, 265.8f };
	vector<string> filter_function = { "SHARPA", "SHARPB", "SHARPC", "VSHARPB", "VSHARPC", "VSHARPD" };
	vector<int> fov = { 120, 150 };

	string fname, mask_path, data_path, out_path;
	Mat mask, data, res, res_bnd;
	for (int i = 0; i < phase2_density.size(); i++)
		for (int j = 0; j < filter_function.size(); j++)
			for (int k = 0; k < fov.size(); k++) {
				fname = Utility::tostr(phase2_density[i]) + "_" + filter_function[j] + "_" + to_string(fov[k]);
				mask_path = mask_folder + "/" + fname + "_mask.dat";
				data_path = data_folder + "/" + fname + "_" + mode + ".dat";
				out_path  = out_folder + "/" + fname + "_" + mode + "_dat";
				mask = fileIO::read_DAT_PD(mask_path);
				data = fileIO::read_DAT_PD(data_path);
				res = Utility::retrieve_mat_by_mask_2D3D(data, mask, (double)1.0, bacgnd_value, true);
				res_bnd = Utility::find_content_bound_3D<double>(res, bacgnd_value);
				fileIO::Mat2Dat(res_bnd, out_path);
			}
}

void mass_extract_segmentation_leg_bone(
	const std::string& mask_folder,
	const std::string& data_folder,
	const std::string& out_folder,
	const std::string& mode,
	const double bacgnd_value
) {
	vector<int> seq = { 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
					   26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40 };
	string fname, mask_path, data_path, out_path;
	Mat mask, data, res, res_bnd;
	for (int i = 0; i < seq.size(); i++) {
		fname = "s" + Utility::tostr(seq[i]);
		mask_path = mask_folder + "/" + fname + "_mask.dat";
		data_path = data_folder + "/" + fname + "_" + mode + ".dat";
		out_path = out_folder + "/" + fname + "_" + mode + "_dat";
		mask = fileIO::read_DAT_PD(mask_path);
		data = fileIO::read_DAT_PD(data_path);
		Utility::shift_mask<double>(mask, 5, 7, 0, 1, true);
		Utility::shift_mask<double>(mask, 6, 8, 0, 1, false);
		res = Utility::retrieve_mat_by_mask_2D3D(data, mask, (double)7.0, bacgnd_value, true);
		res_bnd = Utility::find_content_bound_3D<double>(res, bacgnd_value);
		fileIO::Mat2Dat(res_bnd, out_path);

		//vector<Vec3f> pt3d;
		//for (int i = 0; i < res_bnd.size[0]; i++)
		//	for (int j = 0; j < res_bnd.size[1]; j++)
		//		for (int k = 0; k < res_bnd.size[2]; k++)
		//			if (res_bnd.at<double>(i, j, k) > -1) {
		//				pt3d.push_back(Vec3f(i, j, k));
		//			}
		//Viewer viewer;
		//viewer.show_3dcloud(pt3d);
	}
}

void test() {
	//string path = "D:/Data/MIAS/mdb001.pgm";
	//Mat img = cv::imread(path, IMREAD_GRAYSCALE);
	//vector<vector<vector<vector<int>>>> red_list;
	//red_list = fileIO::read_BNDorRED(path, "red");

	//std::vector<cv::Vec2i> pt2d;
	//Utility::nested4vec_2_Vec2i(red_list, pt2d);

	//string path = "D:/Data/cremi_exp/gen_test.png";
	//Mat img = imread(path, IMREAD_GRAYSCALE);
	//Persistence_Computer pc;
	//pc.set_pers_thd(0.0);
	//pc.set_algorithm(1);
	//pc.source_from_mat(path + ".dat", img);
	////pc.source_from_file(path);
	//pc.run();
	//pc.write_output();
	//pc.clear();

	//std::vector<cv::Vec2i> pt2d;
	//Utility::nested4vec_2_Vec2i(final_red_list_grand, pt2d);

	//Viewer viz;
	//viz.show_2dmask(pt2d, img);


	//string path = "D:/Data/test.png";
	//Mat img = imread(path, IMREAD_GRAYSCALE);
	////Mat cur = Mat::ones(img.size(), img.type()) * 255;
	////Mat inv = cur - img;
	//fileIO::Mat2Dat(img, "D:/Data/test.png");

	//const int width  = img.cols;
	//const int height = img.rows;
	//int dist = 50;
	//for (int i = 0; i<height; i += dist)
	//	cv::line(img, Point(0, i), Point(width, i), cv::Scalar(0));

	//for (int i = 0; i<width; i += dist)
	//	cv::line(img, Point(i, 0), Point(i, height), cv::Scalar(0));

	//fileIO::write_image(img, "D:/Data/test/mdb001_edited_inv_grid.png");
	//fileIO::Mat2Dat(img, "D:/Data/test/mdb001_edited_inv_grid.png");

	//imshow("Window", img);
	//waitKey();


	//string path = "D:/Data/test/mdb001_edited_inv.png";
	//Mat img = imread(path, IMREAD_GRAYSCALE);
	//const int width = img.cols;
	//const int height = img.rows;
	//for (int i = 0; i < height; i++) {
	//	for (int j = 0; j < width; j++) {
	//			img.at<uchar>(i, j) *= 5;
	//	}
	//}

	//imshow("W", img);
	//waitKey();

	//string path = "D:/Data/mnist/mnist_png/training/8/8618.png";
	//Mat img = imread(path, IMREAD_GRAYSCALE);
	//img = Mat(img.size(), img.type(), Scalar(255)) - img;
	//resize(img, img, Size(), 4, 4);
	//imwrite("D:/Data/mnist/mnist_png/training/show.png", img);
	//dilate(img, img, Mat(), Point(-1, -1), 6);
	//erode(img, img, Mat(), Point(-1, -1), 1);
	//imshow("Windows", img);
	//waitKey();

	//std::vector<cv::Point3i> coord;
	//std::vector<std::vector<int>> cycs;
	//std::vector<float> f_vals, pers;
	//fileIO::read_DMT_vertices3D("D:/Data/DMT/output3-ABIDE2Partial/IU/controls/smwp1sub-0029538.nii/tri_verts.txt",
	//	coord, f_vals);
	//fileIO::read_DMT_cycles("D:/Data/DMT/output3-ABIDE2Partial/IU/controls/smwp1sub-0029538.nii/cycle.txt",
	//	cycs, pers);

	//std::vector<cv::Vec3f> pt3d = Utility::retrieve_DMT_cycles_by_index(coord, cycs);

	//Viewer viewer;
	//viewer.show_3dcloud(pt3d);

	//string root = "D:/Data/mnist/mnist_png";
	//string dest = "D:/Data/mnist/mnist_curated";
	//mnist m(root);
	//m.extract_generate_labs(dest);

	//vector<int> bone_array = { 54, 60, 65, 85, 100, 140, 200 };
	//for (int i = 0; i < bone_array.size(); i++) {
	//	string ori_path = "D:/Data/Bones/superlevel-set-nopad/" + to_string(bone_array[i]) + "_sup_ori.dat";
	//	string mask_path = "D:/Data/Bones/superlevel-set-nopad/" + to_string(bone_array[i]) + "_mask.dat";
	//	Mat ori = fileIO::read_DAT_PD("D:/Data/Bones-P2/11965917_BMD=145.4/SHARP A 120_307/145.4_SHARPA_120.dat");
	//	Mat mask = fileIO::read_DAT_PD("D:/Data/Bones-P2/11965917_BMD=145.4/SHARP A 120_307/145.4_SHARPA_120_mask.dat");
	//	Mat res = Utility::retrieve_mat_by_mask_2D3D(ori, mask, (double)1.0, (double)-1023, true);
	//	Mat res_bnd = Utility::find_content_bound_3D<double>(res);
	//	fileIO::Mat2Dat(res_bnd, "D:/Data/Bones/superlevel-set-nopad/" + to_string(bone_array[i]) + "_sup.dat");

	//	vector<Vec3f> pt3d;
	//	for (int i = 0; i < res_bnd.size[0]; i++)
	//		for (int j = 0; j < res_bnd.size[1]; j++)
	//			for (int k = 0; k < res_bnd.size[2]; k++)
	//				if (res_bnd.at<double>(i, j, k) > -1023) {
	//					pt3d.push_back(Vec3f(i, j, k));
	//				}
	//	Viewer viewer;
	//	viewer.show_3dcloud(pt3d);
	//}

	//vector<string> names = {"D:/Data/Bones/superlevel-set-nopad/54_sup.dat.dat", "D:/Data/Bones/superlevel-set-nopad/60_sup.dat.dat",
	//	"D:/Data/Bones/superlevel-set-nopad/65_sup.dat.dat", "D:/Data/Bones/superlevel-set-nopad/85_sup.dat.dat", "D:/Data/Bones/superlevel-set-nopad/100_sup.dat.dat",
	//	"D:/Data/Bones/superlevel-set-nopad/140_sup.dat.dat", "D:/Data/Bones/superlevel-set-nopad/200_sup.dat.dat" };
	//vector<vector<vector<vector<vector<int>>>>> grand_list(7), filtered_list1(7);
	//vector<vector<vector<double>>> pers_BD(7), pers_Time(7);
	//for (int i = 0; i < 7; i++) {
	//	grand_list[i] = fileIO::read_BNDorRED(names[i], "bnd");
	//	pers_BD[i]    = fileIO::read_pers_BD(names[i]);
	//	pers_Time[i]  = Utility::cvt_BD2Pers(pers_BD[i]);
	//	Utility::extract_beyond_or_below_thresh(grand_list[i], pers_Time[i], 20, 1, filtered_list1[i]);
	//}

	//for (int i = 0; i < 7; i++) {
	//	cout << filtered_list1[i][0].size() << " ";
	//}
	//cout << endl;
	//for (int i = 0; i < 7; i++) {
	//	cout << filtered_list1[i][1].size() << " ";
	//}
	//cout << endl;
	//for (int i = 0; i < 7; i++) {
	//	cout << filtered_list1[i][2].size() << " ";
	//}
	//cout << endl;

	//string path = "D:/Data/cremi/cremi_0001.png";
	//Mat img = imread(path, IMREAD_GRAYSCALE);
	//Mat patch = Viewer::random_sample_patch(img, 250, 250);
	//Viewer::color_boundary_gray(patch, 3, 0);
	//imshow("1", patch);
	//waitKey();

	//testVTK();

	//string path = "D:/Data/mnist/mnist_curated/0.png";
	//Mat img = imread(path, IMREAD_GRAYSCALE);
	//img = Viewer::inv_intensity(img, (uchar)255);
	//img = Viewer::binarize_img(img, (uchar)125, (uchar)255, (uchar)0);

	//resize(img, img, Size(128, 128));
	//imshow("test", img);
	//waitKey();

	//Mat ori = fileIO::read_DAT_PD("D:/Data/Bones/sublevel-set-nopad/54_sub.dat.dat");
	//MatSize size = ori.size;
	//int maxv = INT_MIN;
	//int minv = INT_MAX;

	//for (int i = 0; i < size[0]; i++) {
	//	for (int j = 0; j < size[1]; j++) {
	//		for (int k = 0; k < size[2]; k++) {
	//			if (ori.at<double>(i, j, k) > maxv) maxv = ori.at<double>(i, j, k);
	//			if (ori.at<double>(i, j, k) < minv) minv = ori.at<double>(i, j, k);
	//		}
	//	}
	//}
	//cout << maxv << " " << minv << endl;
}