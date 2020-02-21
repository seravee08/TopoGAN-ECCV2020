#include "editor.h"
#include "utility.h"

cv::Mat editor_surface_src, editor_surface_edited, editor_surface_forshow;
cv::Rect editor_cropRect(0, 0, 0, 0);
cv::Point editor_P1(0, 0);
cv::Point editor_P2(0, 0);
bool editor_clicked = false;

void Editor::checkBoundary() {
	//check croping rectangle exceed image boundary
	if (editor_cropRect.width>editor_surface_src.cols - editor_cropRect.x)
		editor_cropRect.width = editor_surface_src.cols - editor_cropRect.x;

	if (editor_cropRect.height>editor_surface_src.rows - editor_cropRect.y)
		editor_cropRect.height = editor_surface_src.rows - editor_cropRect.y;

	if (editor_cropRect.x<0)
		editor_cropRect.x = 0;

	if (editor_cropRect.y<0)
		editor_cropRect.height = 0;
}

void Editor::showImage() {
	editor_surface_forshow = editor_surface_edited.clone();
	checkBoundary();
	rectangle(editor_surface_forshow, editor_cropRect, cv::Scalar(0, 255, 0), 1, 8, 0);
	imshow("Editor", editor_surface_forshow);
}

void Editor::onMouse(int event, int x, int y, int f, void*) {

	switch (event) {
	case  CV_EVENT_LBUTTONDOWN:
		editor_clicked = true;
		editor_P1.x = x;
		editor_P1.y = y;
		editor_P2.x = x;
		editor_P2.y = y;
		break;
	case  CV_EVENT_LBUTTONUP:
		editor_P2.x = x;
		editor_P2.y = y;
		editor_clicked = false;
		break;
	case  CV_EVENT_MOUSEMOVE:
		if (editor_clicked) {
			editor_P2.x = x;
			editor_P2.y = y;
		}
		break;
	default:   break;
	}

	if (editor_clicked) {
		if (editor_P1.x>editor_P2.x) {
			editor_cropRect.x = editor_P2.x;
			editor_cropRect.width = editor_P1.x - editor_P2.x;
		}
		else {
			editor_cropRect.x = editor_P1.x;
			editor_cropRect.width = editor_P2.x - editor_P1.x;
		}

		if (editor_P1.y>editor_P2.y) {
			editor_cropRect.y = editor_P2.y;
			editor_cropRect.height = editor_P1.y - editor_P2.y;
		}
		else {
			editor_cropRect.y = editor_P1.y;
			editor_cropRect.height = editor_P2.y - editor_P1.y;
		}
	}

	showImage();
}

cv::Mat Editor::edit(const cv::Mat& surface, const cv::Mat& real) {
	std::cout << "Click and drag for Selection" << std::endl;
	std::cout << "------> Press 'r' to reset" << std::endl;
	std::cout << "------> Press 'Enter' to mask the ROI" << std::endl;
	std::cout << "------> Press 'Esc' to quit" << std::endl;

	assert(surface.channels() == 3);
	assert(real.channels() == 1);

	cv::Mat real_edited   = real.clone();
	editor_surface_src    = surface.clone();
	editor_surface_edited = editor_surface_src.clone();

	cv::namedWindow("Editor");
	cv::setMouseCallback("Editor", onMouse, NULL);
	cv::imshow("Editor", editor_surface_src);

	while (1) {
		char c = cv::waitKey();
		if (c == 13) {      // Enter
			editor_surface_edited(editor_cropRect).setTo(cv::Scalar(0, 0, 0));
			real_edited(editor_cropRect).setTo(cv::Scalar(0));
		}
		if (c == 27) break; // ESC
		if (c == 'r') { 
			editor_cropRect.x = 0;
			editor_cropRect.y = 0;
			editor_cropRect.width = 0;
			editor_cropRect.height = 0;
			editor_surface_edited = editor_surface_src.clone();
			real_edited = real.clone();
		}
		showImage();
	}
	cv::destroyWindow("Editor");
	return real_edited;
}