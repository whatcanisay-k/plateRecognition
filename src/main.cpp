#include "loadModel.h"
#include "plateSegmentation.h"
#include "recognition.h"
#include <iostream>
using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
	string name = argv[1];
	dnn::Net net = initModel(argv[2], argv[3]);
	Mat roiImg = plateDetect(imread(name));
	Mat plateImgGray = plateDetectAcc(roiImg);
	vector<Mat> plateNumber = plateNumberSeg(plateImgGray);
	plateRecognition(plateNumber, net);
	imshow("car", imread(name));
	waitKey(-1);
	return 0;
}
