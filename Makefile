#

DIR_INC = -I./include/ -I/data/zhangminwen/my_lib/opencv3/include/

DIR_LIB = -L/data/zhangminwen/my_lib/opencv3/lib/

LIBS = -lopencv_imgproc -lopencv_highgui -lopencv_core -lopencv_dnn -lopencv_imgcodecs

DIR_SRC = ./src/

all:PR

PR: main.o loadModel.o segPlate.o regPlate.o
	g++ main.o loadModel.o segPlate.o regPlate.o -o PR $(DIR_INC) $(DIR_LIB) $(LIBS)

main.o: $(DIR_SRC)main.cpp
	g++ -c -g $(DIR_SRC)main.cpp -o main.o $(DIR_INC) $(DIR_LIB) $(LIBS)

loadModel.o: $(DIR_SRC)loadModel.cpp
	g++ -c -g $(DIR_SRC)loadModel.cpp -o loadModel.o $(DIR_INC) $(DIR_LIB) $(LIBS)

segPlate.o: $(DIR_SRC)plateSegmentation.cpp
	g++ -c -g $(DIR_SRC)plateSegmentation.cpp -o segPlate.o $(DIR_INC) $(DIR_LIB) $(LIBS)

regPlate.o: $(DIR_SRC)recognition.cpp
	g++ -c -g $(DIR_SRC)recognition.cpp -o regPlate.o $(DIR_INC) $(DIR_LIB) $(LIBS)

clean:
	rm -rf ./*.o PR
