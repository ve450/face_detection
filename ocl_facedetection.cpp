#include <iostream>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <cmath>
#include <CL/cl.h>

using namespace std;
using namespace cv;

//cl source path
const char *PROG_SOURCE = "kernels.cl";
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

// TODO:test
typedef struct testT {
  int left;
  int right;
  int res;
} testT;

char* loadProgramSource(const char* filename, size_t &length) {
  FILE* programHandle;
  char* programBuffer;
  
  programHandle = fopen(filename, "r");
  fseek(programHandle, 0, SEEK_END);
  length = ftell(programHandle);
  rewind(programHandle);

  programBuffer = (char*) malloc(length + 1);
  programBuffer[length] ='\0';
  fread(programBuffer, sizeof(char), length, programHandle);
  fclose(programHandle);
  return programBuffer;
}

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
	char *source = loadProgramSource(PROG_SOURCE, prog_length);
	x_prog = clCreateProgramWithSource(x_context, 1, (const char**)&source, &prog_length, &err);
	free(source);

	check(err);
	err = clBuildProgram(x_prog, 1, &x_device, NULL,NULL,NULL);
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
  //Mat image = imread("",CV_LOAD_IMAGE_ANYCOLOR);
  //CascadeClassifier cascade;
  //cascade.load("/home/guoxing/opencv/data/haarcascades/haarcascade_frontalface_alt.xml");
  setupCL();
  //uint a[10], b[10];
  //for (int i = 0; i < 10; ++i) {
    //a[i] = i;
    //b[i] = 10 - i;
  //}
  //for (int i = 0; i < 10; ++i) {
    //cout << a[i] << ' ';
  //}
  //cout << endl;
  //cl_mem aBuffer = clCreateBuffer(
      //x_context,
      //CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      //10 * sizeof(uint), a, &err);
  //check(err);
  //cl_mem bBuffer = clCreateBuffer(
      //x_context, 
      //CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      //10 * sizeof(uint), b, &err);
  //check(err);

  //cl_kernel saxpy = clCreateKernel(x_prog, "SAXPY", &err);
  //check(err);

  //err = clSetKernelArg(saxpy, 0, sizeof(aBuffer), &aBuffer);
  //check(err);
  //err = clSetKernelArg(saxpy, 1, sizeof(bBuffer), &bBuffer);
  //check(err);
  //uint two = 2;
  //err = clSetKernelArg(saxpy, 2, sizeof(uint), &two);
  //check(err);
	//cout << "aBuffer size: " << sizeof(aBuffer) << endl;
	//cout << "bBuffer size: " << sizeof(bBuffer) << endl;
	//cout << "generic size: " << sizeof(cl_mem) << endl;
	// -----------------------------------------------------------
	testT a[10];
	for (int i = 0; i < 10; ++i) {
    a[i].left = i;
    a[i].right = 10 - i;
    a[i].res = 0;
  }
  cl_mem aBuffer = clCreateBuffer(
      x_context,
      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      10 * sizeof(testT), a, &err);
  check(err);
  cl_kernel saxpy = clCreateKernel(x_prog, "SAXPY", &err);
  check(err);
  err = clSetKernelArg(saxpy, 0, sizeof(aBuffer), &aBuffer);
  check(err);
  uint two = 2;
  err = clSetKernelArg(saxpy, 1, sizeof(uint), &two);
  check(err);

	// -----------------------------------------------------------

  const size_t global_size[] = {320};
  check(
    clEnqueueNDRangeKernel(
      x_cmd_q,
      saxpy,
      1,
      NULL,
      global_size,
      NULL,
      0,
      NULL,
      NULL
    )
  );

  //uint b_out[10];
  //clEnqueueReadBuffer(x_cmd_q, bBuffer, CL_TRUE, 0, 10*sizeof(uint),
    //b_out, 0, NULL, NULL);
	testT *a_out = (testT *) clEnqueueMapBuffer(x_cmd_q,
					aBuffer,
					CL_TRUE,
					0,
					0,
					10 * sizeof(testT),
					0,
					NULL,
					NULL,
					&err);
	check(err);
  check(clFinish(x_cmd_q));
  
  for (int i = 0; i < 10; ++i) {
    cout << a_out[i].res << ' ';
  }
  cout << endl;

  clReleaseKernel(saxpy);
  clReleaseMemObject(aBuffer);
}
