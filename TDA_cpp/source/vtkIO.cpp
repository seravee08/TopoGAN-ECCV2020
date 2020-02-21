#include "vtkIO.h"
#include "fileIO.h"
#include "vtkVersion.h"
#include "vtkCellArray.h"
#include "vtkPoints.h"
#include "vtkXMLPolyDataWriter.h"
#include "vtkPolyData.h"
#include "vtkSmartPointer.h"
#include "vtkPointData.h"
#include "vtkDoubleArray.h"
#include "vtkFloatArray.h"

void testVTK() {
	vtkSmartPointer<vtkPoints> points =
		vtkSmartPointer<vtkPoints>::New();
	cv::Mat ori = fileIO::read_DAT_PD("D:/Data/Bones/sublevel-set-nopad/54_sub.dat.dat");

	int cnt = 0;
	for (int i = 0; i < ori.size[0]; i++)
		for (int j = 0; j < ori.size[1]; j++)
			for (int k = 0; k < ori.size[2]; k++)
				if (ori.at<double>(i, j, k) > -100 && ori.at<double>(i, j, k) < 100) {
					points->InsertNextPoint(i, j, k);
					cnt++;
				}

	// Create a polydata object and add the points to it.
	vtkSmartPointer<vtkPolyData> polydata =
		vtkSmartPointer<vtkPolyData>::New();
	polydata->SetPoints(points);

	vtkSmartPointer<vtkDoubleArray> weights =
		vtkSmartPointer<vtkDoubleArray>::New();
	weights->SetNumberOfValues(cnt);
	cnt = 0;
	for (int i = 0; i < ori.size[0]; i++)
		for (int j = 0; j < ori.size[1]; j++)
			for (int k = 0; k < ori.size[2]; k++)
				if (ori.at<double>(i, j, k) > -100 && ori.at<double>(i, j, k) < 100) {
					weights->SetValue(cnt, ori.at<double>(i, j, k));
					cnt++;
				}
	polydata->GetPointData()->SetScalars(weights);

	// Write the file
	vtkSmartPointer<vtkXMLPolyDataWriter> writer =
		vtkSmartPointer<vtkXMLPolyDataWriter>::New();
	writer->SetFileName("D:/WorkBench/CPP/VTK/VTK-install/test.vtp");
	writer->SetInputData(polydata);

	// Optional - set the mode. The default is binary.
	//writer->SetDataModeToBinary();
	//writer->SetDataModeToAscii();

	writer->Write();

	return;
}