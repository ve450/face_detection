all:
	g++ ocl_facedetection.cpp `pkg-config --cflags --libs opencv` -lcl -I$(BEIGNET_PATH)/include -L$(BEIGNET_PATH)/build/src -o face
clean:
	rm face
