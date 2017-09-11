#ifndef _PLATESEGMENTATION_H_
#define _PLATESEGMENTATION_H_
#include <opencv2/highgui.hpp>
#include <vector>
using namespace std;
using namespace cv;

// locate the plate from picture
Mat plateDetect(const Mat &tmpImg);

// plate detection accuracy
Mat plateDetectAcc(Mat& roiImg);

// plate numnber segmentation
vector<Mat> plateNumberSeg(Mat& rowGrayImg);

// get plate number
int getPlateNumber(int* shadow, int& start, int& tz, Mat& cutgrayImg, Mat& rowGrayImg, Mat& plateNumber);

#endif