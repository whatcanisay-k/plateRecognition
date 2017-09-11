#ifndef _LOADMODEL_H_
#define _LOADMODEL_H_

#include <opencv2/dnn.hpp>        
using namespace cv;

// load the caffemodel by the opencv dnn toolboxes
dnn::Net initModel(char* model, char* weights);

#endif
