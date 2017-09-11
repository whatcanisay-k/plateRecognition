# plateRecognition

This program used opencv3.1 to load the caffemodel for plate numbers recognition,

Specify the opencv include and library path in the Makefile.

compile: make all 

run：./PR ./car.jpg ./model/lenet.prototxt ./model/plate_iter_90000.caffemodel

