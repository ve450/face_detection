all:
	g++ ocl_facedetection.cpp `pkg-config --cflags --libs opencv` -lcl -I$(BEIGNET_PATH)/include -L$(BEIGNET_PATH)/build/src -o face -g
#all:
	#g++ test.cpp `pkg-config --cflags --libs opencv` -lOpenCL -o face
#all:
	#g++ test.cpp -l$(LD_LIBRARY_PATH) -I$(BEIGNET_PATH)/include -L$(BEIGNET_PATH)/build/src -o face
clean:
	rm face
