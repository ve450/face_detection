#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cmath>
#include <CL/cl.h>

using namespace std;
using namespace cv;

//cl source path
const char *PROG_SOURCE = "cl/lib.cl";
cl_int err;

cl_uint work_dim = 2;
const int LOCAL_GROUP_WIDTH = 16;
const int LOCAL_GROUP_HEIGHT = 16;

//cl parameters
cl_platform_id 		x_platform;
cl_device_id 		x_device;
cl_context 		x_context;
cl_command_queue 	x_cmd_q;
cl_program 		x_prog;
size_t 			prog_length;

#define check(err) assert(err == CL_SUCCESS)
typedef unsigned int uint;

/* To initialize OpenCL relevant parameters */
void setupCL() {
	err = clGetPlatformIDs(1, &x_platform, NULL);
	check(err);
	err = clGetDeviceIDs(x_platform,CL_DEVICE_TYPE_GPU, 1, &x_device, NULL);
	check(err);
	x_context = clCreateContext(NULL, 1, &x_device, NULL,NULL,&err);

	// ???
	if(err != 0) {
		err = clGetDeviceIDs(x_platform, CL_DEVICE_TYPE_GPU, 1, &x_device, NULL);
		check(err);
		x_context = clCreateContext(NULL,1, &x_device, NULL,NULL,&err);
		check(err);
	}

	x_cmd_q = clCreateCommandQueue(x_context, x_device, 0, &err);
	check(err);
	char *source = oclLoadProgSource(PROG_SOURCE,"",&prog_length);
	x_prog = clCreateProgramWithSource(x_context, 1, (const char**)&source, &prog_length, &err);
	check(err);
	err = clBuildProgram(x_prog, 1, &x_device, "-I../cl",NULL,NULL);
	//log
	if(err == CL_BUILD_PROGRAM_FAILURE) {
		size_t log_size;
		clGetProgramBuildInfo(x_prog,x_device,CL_PROGRAM_BUILD_LOG,0,NULL,&log_size);
    char *log = (char*) malloc(log_size);
    clGetProgramBuildInfo(x_prog,x_device,CL_PROGRAM_BUILD_LOG,log_size,log,NULL);
    cout<<log<<endl;
	}
	//log end
	check(err);
}

int main(int argc, char** argv) {

}
