#include "notDecent.h"
#include <string>

int main() {

	//// ===== Command line inputs =====
	//// first arg: path to the file
	//// second arg: output folder
	//if (argc < 2)
	//{
	//	std::cout << "usage: " << argv[0] << " input_file.ext"
	//		<< " [output_path]" << std::endl;
	//	return 1;
	//}
	//string input_file = argv[1];
	//string output_file = (argc == 2) ? "" : argv[2];
	//// ==============================

	//string input_file = "D:/Data/MIAS/mdb001.pgm.dat";
	//string output_file = "";

	//vector<vector<vector<vector<int>>>> final_red_list_grand;
	//vector<vector<vector<vector<int>>>> final_boundary_list_grand;
	//vector<vector<vector<int>>> pers_V;
	//vector<vector<vector<double>>> pers_BD;

	//PersistenceCubic pc;
	//pc.set_input_file(input_file);
	//pc.set_output_file(output_file);
	//pc.set_pers_thd(15.0);
	//pc.run(final_red_list_grand, final_boundary_list_grand, pers_V, pers_BD);

	//// ===== Write out resutls =====
	//pc.write_bnd(final_boundary_list_grand);
	//pc.write_red(final_red_list_grand);
	//pc.write_pers(pers_V, pers_BD);
	//// =============================

	std::string path_in  = "D:/Data/isbi/tt";
	std::string path_out = "D:/Data/isbi/test";
	//mass_Mat2Dat(path_in, ".png");
	//mass_Dat2PD(path_in, ".png", -1, 10000);
	//mass_visualize2D(path_in, ".png", "red");
	//mass_compute_slicedWasserstein(path_in, ".dat");
	//mass_filter_PD(path, "D:/Data/MIAS_curated_th5", 5, ".png");
	//mass_extract_ABIDE_labels(path, ".dat", "D:/Data/Brain/DAT_INV_FANLIU_ABIDE2/ABIDE_labels.txt");
	//mass_extract_patch_solo(path_in, ".tiff", path_out, "isbi", ".png", 2, 128, 128, 1, 0, 64, 64, 1, -1, false);
	mass_extract_patch_dual("D:/Data/isbi/segmentation", ".png", "D:/Data/isbi/texture", ".png", "D:/Data/isbi/isbi64_mask", "isbi", ".jpg",
		"D:/Data/isbi/isbi64_tx", "isbi", ".jpg", 50, 128, 128, 1, 0, 64, 64, 1, -1, false);
	//mass_binarize_grayimg(path_in, ".png", path_out);
	//mass_extract_segmentation_vertebral_bone("D:/Data/leg_bone_complete/dat_mask", "D:/Data/leg_bone_complete/dat_sublevel", "D:/Data/leg_bone_complete/res_sublevel", "sub", -1);
	//mass_extract_segmentation_leg_bone("D:/Data/leg_bone_complete/dat_mask", "D:/Data/leg_bone_complete/dat_sublevel", "D:/Data/leg_bone_complete/res_sublevel", "sub", -1);

	//test();

	system("pause");

	return 0;
}

