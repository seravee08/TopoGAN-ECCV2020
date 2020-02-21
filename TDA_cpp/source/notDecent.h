#pragma once

#ifndef NOTDECENT_H
#define NOTDECENT_H

#include <string>

void mass_Mat2Dat(const std::string& path, const std::string& post_fix);

void mass_Dat2PD(const std::string& path, const std::string& post_fix, int ind_s=-1, int ind_f=INT_MAX);

void mass_compute_slicedWasserstein(
	const std::string& path,
	const std::string& post_fix,
	const bool approx = true,
	const int slices = 100
);

void mass_extract_ABIDE_labels(
	const std::string& path,
	const std::string& post_fix,
	const std::string& out_path
);

void single_visualize3D(
	const std::string& path,
	const std::string& file_format
);

void mass_visualize2D(
	const std::string& path,
	const std::string& post_fix,
	const std::string& file_format
);

void mass_filter_PD(
	const std::string& in_folder,
	const std::string& out_folder,
	const double threshold,
	const std::string& post_fix
);

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
	const int resize_width  = -1,
	const int resize_height = -1,
	const int channels = 1,
	const float pass_if_beyond = 0.05,
	bool binarize = false
);

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
	const int resize_width  = -1,
	const int resize_height = -1,
	const int channels = 1,
	const float pass_if_beyond = 0.05,
	bool binarize = false
);

void mass_binarize_grayimg(
	const std::string& source_folder,
	const std::string& source_extension,
	const std::string& target_folder
);

void mass_extract_segmentation_vertebral_bone(
	const std::string& mask_folder,
	const std::string& data_folder,
	const std::string& out_folder,
	const std::string& mode,
	const double bacgnd_value
);

void mass_extract_segmentation_leg_bone(
	const std::string& mask_folder,
	const std::string& data_folder,
	const std::string& out_folder,
	const std::string& mode,
	const double bacgnd_value
);

void test();

#endif //!NOTDECENT_H