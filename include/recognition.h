#ifndef _RECOGNITION_H_
#define _RECOGNITION_H_
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
using namespace std;
using namespace cv;

// predict the plate number by dnn based on caffemodel
int myCaffePredict(Mat& inputImage, dnn::Net& net);

// get the max prob of the plate numbers
vector<int> getMaxClass(dnn::Blob &probBlob);

// predict all plate numbet and print the predict results
void plateRecognition(vector<Mat>& plateNumber, dnn::Net& net);

#endif