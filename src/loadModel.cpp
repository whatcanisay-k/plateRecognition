#include "loadModel.h"         
#include <iostream>  
using namespace std;
using namespace cv;

dnn::Net initModel(char* model, char* weights){
	string modelTxt = model;
	string modelBin = weights;
	Ptr<dnn::Importer> importer;
	try
	{
		importer = dnn::createCaffeImporter(modelTxt, modelBin);
	}
	catch (const Exception &err)
	{
		std::cerr << err.msg << std::endl;
	}
	if (!importer)
	{
		std::cerr << "cant load network!" << std::endl;
		exit(-1);
	}
	dnn::Net net;
	importer->populateNet(net);
	importer.release();
	return net;
}
