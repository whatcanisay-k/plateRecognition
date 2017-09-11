#include "plateSegmentation.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
using namespace std;
using namespace cv;

Mat plateDetect(const Mat &tmpImg){

	Mat hsvImg;
	cvtColor(tmpImg, hsvImg, COLOR_BGR2HSV);
	vector<Mat> hsvSplits;
	split(hsvImg, hsvSplits);
	equalizeHist(hsvSplits[2], hsvSplits[2]);
	Mat mergeImg;
	merge(hsvSplits, mergeImg);

	//detect blue area
	Mat blueImgThresh;
	inRange(mergeImg, Scalar(60, 60, 60), Scalar(130, 255, 255), blueImgThresh);


	Mat dilateImg;
	dilate(blueImgThresh, dilateImg, Mat(2.5, 2.5, CV_8U), Point(-1, -1), 1);
	Mat openImg;
	morphologyEx(dilateImg, openImg, MORPH_OPEN, Mat(3, 3, CV_8U), Point(-1, -1), 1);
	//imshow("split", openImg);
	//waitKey(-1);

	Mat grayImg;
	cvtColor(tmpImg, grayImg, CV_BGR2GRAY);
	Mat equalImg;
	equalizeHist(grayImg, equalImg);
	Mat blurImg;
	medianBlur(equalImg, blurImg, 3);

	//detect plate edge
	Mat cannyImg;
	Canny(blurImg, cannyImg, 120, 360);

	Mat closeImg;
	morphologyEx(cannyImg, closeImg, MORPH_CLOSE, Mat(2, 11, CV_8U), Point(-1, -1), 1);
	Mat openImgEdge;
	morphologyEx(closeImg, openImgEdge, MORPH_OPEN, Mat(1, 7, CV_8U), Point(-1, -1), 1);
	
	// and operation
	Mat andImg;
	bitwise_and(openImg, openImgEdge, andImg);

	Mat dilateImgEdge;
	dilate(andImg, dilateImgEdge, Mat(7, 7, CV_8U), Point(-1, -1), 1);

	// detect contours
	vector<vector<Point> > contours;
	findContours(dilateImgEdge, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	vector<Point> contmax;
	int area, maxarea = 0;
	for (int i = 0; i<contours.size(); i++){
		area = contourArea(contours[i]);
		if (area>maxarea){
			maxarea = area;
			contmax = contours[i];
		}
	}
	Rect arect = boundingRect(contmax);
	Mat roiImg;
	roiImg.create(arect.height, arect.width, CV_8U);
	tmpImg(arect).copyTo(roiImg);
	//imshow("ROI image", roiImg);
	//waitKey(-1);
	return roiImg;
}

Mat plateDetectAcc(Mat& roiImg){

	Mat HSVimg;
	cvtColor(roiImg, HSVimg, COLOR_BGR2HSV);
	vector<Mat> HSVsplit;
	split(HSVimg, HSVsplit);
	equalizeHist(HSVsplit[2], HSVsplit[2]);
	Mat MERGEimg;
	merge(HSVsplit, MERGEimg);

	Mat blueImg;
	inRange(MERGEimg, Scalar(60, 60, 60), Scalar(140, 255, 255), blueImg);

	Mat nonZero;
	findNonZero(blueImg, nonZero);
	Rect arect = boundingRect(nonZero);

	Mat plateImg;
	roiImg(arect).copyTo(plateImg);
	Mat plateImgGray;
	cvtColor(plateImg, plateImgGray, CV_BGR2GRAY);

	Mat plateImgEdge;
	Canny(plateImgGray, plateImgEdge, 50, 200, 3);
	int minvote = 46;
	double minlength = 25;
	double maxgap = 4;
	double rho = 1;
	double theta = CV_PI / 180;
	vector<Vec4i> lines;
	HoughLinesP(plateImgEdge, lines, rho, theta, minvote, minlength, maxgap);

	int n = lines.size();
	vector<double> a(n, 0);
	int k = 0;
	Mat houghImg;
	if (n != 0){
		for (int i = 0; i < n; i++){
			Vec4i L = lines[i];
			a[i] = sqrtf((L[0] - L[2])*(L[0] - L[2]) + (L[1] - L[3])*(L[1] - L[3]));
		}

		double temp = a[0];
		for (int i = 0; i<n; i++){
			if (a[i]>temp){
				temp = a[i];
				k = i;
			}
		}

		Vec4i maxline = lines[k];
		Point p1(maxline[0], maxline[1]);
		Point p2(maxline[2], maxline[3]);
		Mat houghLine;;
		plateImg.copyTo(houghLine);
		line(houghLine, p1, p2, Scalar(177, 156, 242));

		double angle = fastAtan2(maxline[3] - maxline[1], maxline[2] - maxline[0]);
		Point2f centerpoint = (cv::Point_<float>(plateImgGray.cols / 2), cv::Point_<float>(plateImgGray.rows / 2));
		Mat rotateMat = getRotationMatrix2D(centerpoint, angle, 1.0);
		warpAffine(plateImg, houghImg, rotateMat, plateImgGray.size(), 1, 0, 0);
	}

	else{ plateImg.copyTo(houghImg); }

	Mat roiImgHoughGray;
	cvtColor(houghImg, roiImgHoughGray, CV_BGR2GRAY);

	Mat threshimg;
	Mat mean;
	Mat stddev;
	meanStdDev(roiImgHoughGray, mean, stddev);
	int T1;
	double t1 = sqrt(0.318 / 0.682);
	T1 = mean.at<double>(0, 0) + t1*stddev.at<double>(0, 0);
	threshold(roiImgHoughGray, threshimg, T1, 255, CV_THRESH_BINARY);

	Mat_<uchar> row;
	row.create(threshimg.size().height, threshimg.size().width);
	int rows = threshimg.size().height;
	int cols = threshimg.size().width;
	Mat sumrow = Mat::zeros(threshimg.size().height, 1, CV_8U);
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			threshimg.at<uchar>(i, j) = threshimg.at<uchar>(i, j) / 255;
		}
	}

	for (int i = 0; i < rows; i++){
		for (int j = 0; j<cols - 1; j++){
			row(i, j) = abs(threshimg.at<uchar>(i, j) - threshimg.at<uchar>(i, j + 1));
			sumrow.at<uchar>(i, 0) += row(i, j);
		}
	}

	int bottom = rows;
	int h1 = bottom / 2, h2 = bottom / 2;
	while (sumrow.at<uchar>(h1, 0)>13)
	{
		h1 = h1 - 1;
	}
	while (sumrow.at<uchar>(h2, 0) > 13)
	{
		h2 = h2 + 1;
	}
	int h = h2 - h1 + 1;
	int w = cols;
	houghImg.copyTo(plateImg);
	rectangle(plateImg, Rect(0, h1, w, h), Scalar(177, 156, 242));

	Mat rowImg;
	houghImg(Rect(0, h1, w, h)).copyTo(rowImg);

	Mat rowGrayImg;
	cvtColor(rowImg, rowGrayImg, CV_BGR2GRAY);
	return rowGrayImg;
}

vector<Mat> plateNumberSeg(Mat& rowGrayImg){
	vector<Mat>plateNumbers(7);
	Mat rowThreshImg;
	Mat mean;
	Mat stddev;
	meanStdDev(rowGrayImg, mean, stddev);
	int T;
	double t = sqrt(0.318 / 0.682);
	T = mean.at<double>(0, 0) + t*stddev.at<double>(0, 0);
	threshold(rowGrayImg, rowThreshImg, T, 255, CV_THRESH_BINARY);

	int *shadow = new int[rowThreshImg.cols];
	for (int i = 0; i < rowThreshImg.cols; i++){
		shadow[i] = 0;
	}

	for (int i = 0; i < rowThreshImg.cols; i++){
		for (int j = 0; j < rowThreshImg.rows; j++){
			shadow[i] += rowThreshImg.at<uchar>(j, i);
		}
	}

	Mat cutgrayImg;
	rowThreshImg.copyTo(cutgrayImg);

	int tz = 0;
	for (int i = 0; i < rowThreshImg.cols; i++){
		tz += shadow[i];
	}
	tz = 0.63*(tz / rowThreshImg.cols);

	double maxproj = 0;
	for (int i = 0; i<rowThreshImg.cols; i++){
		if (shadow[i]>maxproj)
			maxproj = shadow[i];
	}

	Mat projimg(maxproj, rowThreshImg.cols, CV_8U, Scalar(255));
	for (int i = 0; i < rowThreshImg.cols; i++){
		line(projimg, Point(i, maxproj - shadow[i]), Point(i, maxproj - 1), Scalar::all(0));
	}
	line(projimg, Point(0, maxproj - tz), Point(rowThreshImg.cols - 1, maxproj - tz), Scalar::all(0), 15);

	resize(projimg, projimg, Size(480, 320));

	int start = 0.3*cutgrayImg.cols;
	while (shadow[start] > tz){
		start = start + 1;
	}
	int meanWidth = 0;;
	for (int i = 2; i < 7; i++){
		int end = getPlateNumber(shadow, start, tz, cutgrayImg, rowGrayImg, plateNumbers[i]);
		meanWidth = end - start;
		start = end;
	}
	meanWidth = meanWidth / 5;

	// second number city code segmentation
	start = 0.35*cutgrayImg.cols;
	int end;
	while (shadow[start] > tz){
		start = start - 1;
	}
	for (; start > 0; start--){
		if (shadow[start] > tz)
			break;
	}
	for (end = start; end > 0; end--){
		if (shadow[end] == 0)
			break;
	}

	for (; start < cutgrayImg.cols; start++){
		if (shadow[start] == 0)
			break;
	}
	int addlength;
	if (start - end + 1 - 0.5*cutgrayImg.rows < 0){
		addlength = 0.5*cutgrayImg.rows - (start - end + 1);
	}
	else{ addlength = 0; }
	Mat numberImg;
	rowGrayImg(Rect(end - addlength / 2, 0, start - end + 1 + addlength, cutgrayImg.rows)).copyTo(numberImg);
	resize(numberImg, numberImg, Size(28, 28));
	numberImg.convertTo(numberImg, CV_32F, 0.003906, 0);
	plateNumbers[1] = numberImg;

	// ProvinceCode segmentation
	start = end;
	for (; start > 0; start--){
		if (shadow[start] > tz)
			break;
	}
	for (end = start; end > 0; end--){
		if (shadow[end] == 0)
			break;
	}
	for (; start<cutgrayImg.cols; start++){
		if (shadow[start] == 0)
			break;
	}
	int wordwidth;
	wordwidth = start - end;
	while (wordwidth<0.8*meanWidth){
		for (; end>0; end--){
			if (shadow[end]>0)
				break;
		}
		for (end = end - 1; end > 0; end--){
			if (shadow[end] == 0)
				break;
		}
		wordwidth = start - end;
	}
	if (start - end + 1 - 0.5*cutgrayImg.rows < 0){
		addlength = 0.5*cutgrayImg.rows - (start - end + 1);
	}
	else{ addlength = 0;}
	rowGrayImg(Rect(end - addlength / 2, 0, start - end + 1 + addlength, cutgrayImg.rows)).copyTo(numberImg);
	resize(numberImg, numberImg, Size(28, 28));
	numberImg.convertTo(numberImg, CV_32F, 0.003906, 0);
	plateNumbers[0] = numberImg;
	return plateNumbers; 
}

int getPlateNumber(int* shadow, int& start, int& tz, Mat& cutgrayImg, Mat& rowGrayImg, Mat& plateNumber){
	int end;
	for (; start<cutgrayImg.cols; start++){
		if (shadow[start]>tz)
			break;
	}
	for (end = start; end<cutgrayImg.cols; end++){
		if (shadow[end] == 0)
			break;
	}
	for (; start>0; start--){
		if (shadow[start] == 0)
			break;
	}
	int addlength;
	if (end - start + 1 - 0.5*cutgrayImg.rows < 0){
		addlength = 0.5*cutgrayImg.rows - (end - start + 1);
	}
	else{ addlength = 0; }
	Mat numberImg;
	rowGrayImg(Rect(start - addlength / 2, 0, end - start + 1 + addlength, cutgrayImg.rows)).copyTo(numberImg);
	resize(numberImg, numberImg, Size(28, 28));
	numberImg.convertTo(numberImg, CV_32F, 0.003906, 0);
	plateNumber = numberImg;
	return end;
}
