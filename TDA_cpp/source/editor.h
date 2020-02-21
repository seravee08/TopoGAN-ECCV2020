#pragma once

#ifndef EDITOR_H
#define EDITOR_H

#include <opencv2/opencv.hpp>

class Editor {
public:
	Editor() {};

	~Editor() {};

	static cv::Mat edit(const cv::Mat& surface, const cv::Mat& real);

private:
	static void checkBoundary();

	static void showImage();

	static void onMouse(int event, int x, int y, int f, void*);
};

#endif //!EDITOR_H