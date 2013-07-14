#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "ocl_facedetection.hpp"
#include "face_usage.hpp"
#include <cctype>
#include <iostream>
#include <iterator>
#include <vector>
#include <stdio.h>
#include "mat_convert.h"
#include <iostream>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cmath>
#include <CL/cl.h>

using namespace std;
using namespace cv;

static void help()
{
    cout << "\nThis program demonstrates the cascade recognizer. Now you can use Haar or LBP features.\n"
            "This classifier can recognize many kinds of rigid objects, once the appropriate classifier is trained.\n"
            "It's most known use is for faces.\n"
            "Usage:\n"
            "./facedetect [--cascade=<cascade_path> this is the primary trained classifier such as frontal face]\n"
               "   [--nested-cascade[=nested_cascade_path this an optional secondary classifier such as eyes]]\n"
               "   [--scale=<image scale greater or equal to 1, try 1.3 for example>]\n"
               "   [--try-flip]\n"
               "   [filename|camera_index]\n\n"
            "see facedetect.cmd for one call:\n"
            "./facedetect --cascade=\"../../data/haarcascades/haarcascade_frontalface_alt.xml\" --nested-cascade=\"../../data/haarcascades/haarcascade_eye.xml\" --scale=1.3\n\n"
            "During execution:\n\tHit any key to quit.\n"
            "\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
}

const char *PROG_SOURCE = "kernels.cl";
cl_int err;

cl_uint work_dim = 1;
const float kScaleFactor = 1.3f;

//cl parameters
cl_platform_id 		x_platform;
cl_device_id 		x_device;
cl_context 		x_context;
cl_command_queue 	x_cmd_q;
cl_program 		x_prog;
size_t 			prog_length;

#define check(err) assert(err == CL_SUCCESS)
typedef unsigned int uint;

typedef struct NewHidHaarFeature
{
    int tilted;
    struct
    {
        int p0[2], p1[2], p2[2], p3[2];
        float weight;
    }
    rect[CV_HAAR_FEATURE_MAX];
} NewHidHaarFeature;


typedef struct NewHidHaarTreeNode
{
    NewHidHaarFeature feature;
    float threshold;
    //int left;
    //int right;
} NewHidHaarTreeNode;


typedef struct NewHidHaarClassifier
{
    //int count;
    //CvHaarFeature* orig_feature;
    NewHidHaarTreeNode node;
    float alpha[2];
} NewHidHaarClassifier;


typedef struct NewHidHaarStageClassifier
{
    int  count;
    float threshold;
    NewHidHaarClassifier classifier[213];
    int two_rects;

} NewHidHaarStageClassifier;


typedef struct NewHidHaarClassifierCascade
{
    int  count;
    int  isStumpBased;
    int  has_tilted_features;
    int  is_tree;
    float inv_window_area;
    NewHidHaarStageClassifier stage_classifier[22];
    int p0[2], p1[2], p2[2], p3[2];

} NewHidHaarClassifierCascade;

typedef struct numbers_t
{
  float factor;
  int scale_num;
  int y;
  int x;
  float variance_norm_factor;
} numbers_t;

NewHidHaarClassifierCascade* convertPointerToNonPointer(
  const CvHidHaarClassifierCascade *old_cascade
) {
  NewHidHaarClassifierCascade* new_cascade = new NewHidHaarClassifierCascade();
  new_cascade->count = old_cascade->count;
  new_cascade->isStumpBased = old_cascade->isStumpBased;
  new_cascade->has_tilted_features = old_cascade->has_tilted_features;
  new_cascade->is_tree = old_cascade->is_tree;
  new_cascade->inv_window_area = old_cascade->inv_window_area;
  new_cascade->p0[0] = old_cascade->p0_loc[0];
  new_cascade->p0[1] = old_cascade->p0_loc[1];
  new_cascade->p1[0] = old_cascade->p1_loc[0];
  new_cascade->p1[1] = old_cascade->p1_loc[1];
  new_cascade->p2[0] = old_cascade->p2_loc[0];
  new_cascade->p2[1] = old_cascade->p2_loc[1];
  new_cascade->p3[0] = old_cascade->p3_loc[0];
  new_cascade->p3[1] = old_cascade->p3_loc[1];
  
  for (int i = 0; i < old_cascade->count; ++i) {
    CvHidHaarStageClassifier* hid_stage_classifier = old_cascade->stage_classifier + i;
    NewHidHaarStageClassifier new_stage_classifier;
    new_stage_classifier.count = hid_stage_classifier->count;
    new_stage_classifier.threshold = hid_stage_classifier->threshold;
    new_stage_classifier.two_rects = hid_stage_classifier->two_rects;
    for (int j = 0; j < hid_stage_classifier->count; ++j) {
      CvHidHaarClassifier* hid_classifier = hid_stage_classifier->classifier + j;
      NewHidHaarClassifier new_classifier;
      new_classifier.alpha[0] = hid_classifier->alpha[0];
      new_classifier.alpha[1] = hid_classifier->alpha[1];

      // node
      CvHidHaarTreeNode* node = hid_classifier->node;
      NewHidHaarTreeNode new_node;
      new_node.threshold = node->threshold;

      // feature
      CvHidHaarFeature hid_feature = node->feature;
      NewHidHaarFeature new_feature;
      new_feature.tilted = hid_feature.tilted;
      for (int k = 0; k < CV_HAAR_FEATURE_MAX; ++k) {
        new_feature.rect[k].p0[0] = hid_feature.rect[k].p0_loc[0];
        new_feature.rect[k].p0[1] = hid_feature.rect[k].p0_loc[1];
        new_feature.rect[k].p1[0] = hid_feature.rect[k].p1_loc[0];
        new_feature.rect[k].p1[1] = hid_feature.rect[k].p1_loc[1];
        new_feature.rect[k].p2[0] = hid_feature.rect[k].p2_loc[0];
        new_feature.rect[k].p2[1] = hid_feature.rect[k].p2_loc[1];
        new_feature.rect[k].p3[0] = hid_feature.rect[k].p3_loc[0];
        new_feature.rect[k].p3[1] = hid_feature.rect[k].p3_loc[1];
        new_feature.rect[k].weight = hid_feature.rect[k].weight;
      }
      new_node.feature = new_feature;
      new_classifier.node = new_node;
      new_stage_classifier.classifier[j] = new_classifier;
    }
    new_cascade->stage_classifier[i] = new_stage_classifier;
  }
  return new_cascade;
}

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


const string cascadeName = "haarcascade_frontalface_alt.xml";
const string nestedCascadeName = "/home/zichao/opencv-2.4.5/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

int main( int argc, const char** argv )
{
  setupCL();
    CvCapture* capture = 0;
    Mat frame, frameCopy, image;
    const string scaleOpt = "--scale=";
    size_t scaleOptLen = scaleOpt.length();
    const string cascadeOpt = "--cascade=";
    size_t cascadeOptLen = cascadeOpt.length();
    const string nestedCascadeOpt = "--nested-cascade";
    size_t nestedCascadeOptLen = nestedCascadeOpt.length();
    const string tryFlipOpt = "--try-flip";
    size_t tryFlipOptLen = tryFlipOpt.length();
    string inputName;
    bool tryflip = false;
    help();

    OCL_CascadeClassifier cascade, nestedCascade;
    double scale = 1;

    for( int i = 1; i < argc; i++ )
    {
        cout << "Processing " << i << " " <<  argv[i] << endl;
        if( cascadeOpt.compare( 0, cascadeOptLen, argv[i], cascadeOptLen ) == 0 )
        {
            //cascadeName.assign( argv[i] + cascadeOptLen );
            cout << "  from which we have cascadeName= " << cascadeName << endl;
        }
        else if( nestedCascadeOpt.compare( 0, nestedCascadeOptLen, argv[i], nestedCascadeOptLen ) == 0 )
        {
            //if( argv[i][nestedCascadeOpt.length()] == '=' )
                //nestedCascadeName.assign( argv[i] + nestedCascadeOpt.length() + 1 );
            if( !nestedCascade.load( nestedCascadeName ) )
                cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
        }
        else if( scaleOpt.compare( 0, scaleOptLen, argv[i], scaleOptLen ) == 0 )
        {
            if( !sscanf( argv[i] + scaleOpt.length(), "%lf", &scale ) || scale < 1 )
                scale = 1;
            cout << " from which we read scale = " << scale << endl;
        }
        else if( tryFlipOpt.compare( 0, tryFlipOptLen, argv[i], tryFlipOptLen ) == 0 )
        {
            tryflip = true;
            cout << " will try to flip image horizontally to detect assymetric objects\n";
        }
        else if( argv[i][0] == '-' )
        {
            cerr << "WARNING: Unknown option %s" << argv[i] << endl;
        }
        else
            inputName.assign( argv[i] );
    }

    if( !cascade.load( cascadeName ) )
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        help();
        return -1;
    }

    if( inputName.empty() || (isdigit(inputName.c_str()[0]) && inputName.c_str()[1] == '\0') )
    {
        capture = cvCaptureFromCAM( inputName.empty() ? 0 : inputName.c_str()[0] - '0' );
        int c = inputName.empty() ? 0 : inputName.c_str()[0] - '0' ;
        if(!capture) cout << "Capture from CAM " <<  c << " didn't work" << endl;
    }
    else if( inputName.size() )
    {
        image = imread( inputName, 1 );
        if( image.empty() )
        {
            capture = cvCaptureFromAVI( inputName.c_str() );
            if(!capture) cout << "Capture from AVI didn't work" << endl;
        }
    }
    else
    {
        image = imread( "lena.jpg", 1 );
        if(image.empty()) cout << "Couldn't read lena.jpg" << endl;
    }

    cvNamedWindow( "result", 1 );

    if( capture )
    {
        cout << "In capture ..." << endl;
        for(;;)
        {
            IplImage* iplImg = cvQueryFrame( capture );
            frame = iplImg;
            if( frame.empty() )
                break;
            if( iplImg->origin == IPL_ORIGIN_TL )
                frame.copyTo( frameCopy );
            else
                flip( frame, frameCopy, 0 );

            detectAndDraw( frameCopy, cascade, nestedCascade, scale, tryflip );

            if( waitKey( 10 ) >= 0 )
                goto _cleanup_;
        }

        waitKey(0);

_cleanup_:
        cvReleaseCapture( &capture );
    }
    else
    {
        cout << "In image read" << endl;
        if( !image.empty() )
        {
            detectAndDraw( image, cascade, nestedCascade, scale, tryflip );
            waitKey(0);
        }
        else if( !inputName.empty() )
        {
            /* assume it is a text file containing the
            list of the image filenames to be processed - one per line */
            FILE* f = fopen( inputName.c_str(), "rt" );
            if( f )
            {
                char buf[1000+1];
                while( fgets( buf, 1000, f ) )
                {
                    int len = (int)strlen(buf), c;
                    while( len > 0 && isspace(buf[len-1]) )
                        len--;
                    buf[len] = '\0';
                    cout << "file " << buf << endl;
                    image = imread( buf, 1 );
                    if( !image.empty() )
                    {
                        detectAndDraw( image, cascade, nestedCascade, scale, tryflip );
                        c = waitKey(0);
                        if( c == 27 || c == 'q' || c == 'Q' )
                            break;
                    }
                    else
                    {
                        cerr << "Aw snap, couldn't read image " << buf << endl;
                    }
                }
                fclose(f);
            }
        }
    }

    cvDestroyWindow("result");

    return 0;
}




void detectAndDraw( Mat& img, OCL_CascadeClassifier& cascade,
                    OCL_CascadeClassifier& nestedCascade,
                    double scale, bool tryflip )
{
    int i = 0;
    double t = 0;
    vector<Rect> faces, faces2;
    const static Scalar colors[] =  { CV_RGB(0,0,255),
        CV_RGB(0,128,255),
        CV_RGB(0,255,255),
        CV_RGB(0,255,0),
        CV_RGB(255,128,0),
        CV_RGB(255,255,0),
        CV_RGB(255,0,0),
        CV_RGB(255,0,255)} ;
    Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );

    cvtColor( img, gray, CV_BGR2GRAY );
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );

    /* matrix encode test
    int size;
    int *header;
    uchar *buf = encodeMatrix(size, header, 1, &img);
    printf("%s\n", header);
    //decodeMatrix(size, header, 1, buf);
    */

    //detect
    t = (double)cvGetTickCount();
    cascade.CL_detectMultiScale( smallImg, faces,
        kScaleFactor, 2, 0
        //|CV_HAAR_FIND_BIGGEST_OBJECT
        //|CV_HAAR_DO_ROUGH_SEARCH
        |CV_HAAR_SCALE_IMAGE
        ,
        Size(30, 30) );
    t = (double)cvGetTickCount() - t;
    printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
    printf( "--------------------------------------\n");
    //draw
    for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
    {
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = colors[i%8];
        int radius;
//cout << "x: " << r->x << "y: " << r->y << endl;
        double aspect_ratio = (double)r->width/r->height;
        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
        {
            center.x = cvRound((r->x + r->width*0.5)*scale);
            center.y = cvRound((r->y + r->height*0.5)*scale);
            radius = cvRound((r->width + r->height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }
        else
            rectangle( img, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
                       cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
                       color, 3, 8, 0);
        if( nestedCascade.empty() )
            continue;
        smallImgROI = smallImg(*r);
        nestedCascade.CL_detectMultiScale( smallImgROI, nestedObjects,
            kScaleFactor, 2, 0
            //|CV_HAAR_FIND_BIGGEST_OBJECT
            //|CV_HAAR_DO_ROUGH_SEARCH
            //|CV_HAAR_DO_CANNY_PRUNING
            |CV_HAAR_SCALE_IMAGE
            ,
            Size(30, 30) );
        for( vector<Rect>::const_iterator nr = nestedObjects.begin(); nr != nestedObjects.end(); nr++ )
        {
            center.x = cvRound((r->x + nr->x + nr->width*0.5)*scale);
            center.y = cvRound((r->y + nr->y + nr->height*0.5)*scale);
            radius = cvRound((nr->width + nr->height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }
    }
    cv::imshow( "result", img );
}

//body begins

void OCL_CascadeClassifier::CL_detectMultiScale ( const Mat& image,
				   CV_OUT vector<Rect>& objects,
				   float scaleFactor,
				   int minNeighbors, int flags,
				   Size minSize,
				   Size maxSize) 
	{
		const double GROUP_EPS = 0.2;

		CV_Assert( scaleFactor > 1 && image.depth() == CV_8U );
		if( empty() )
		  return;
		MemStorage storage(cvCreateMemStorage(0));
		CvMat _image = image;
		CvSeq* _objects = OCL_cvHaarDetectObjectsForROC( &_image, oldCascade, storage, scaleFactor,
					      minNeighbors, flags, minSize, maxSize);
		vector<CvAvgComp> vecAvgComp;
		cv::Seq<CvAvgComp>(_objects).copyTo(vecAvgComp);
		//cv::Seq<CvAvgComp>(result_seq).copyTo(vecAvgComp);
		objects.resize(vecAvgComp.size());
		std::transform(vecAvgComp.begin(), vecAvgComp.end(), objects.begin(), getRect());
		return;
	}

CvSeq*
OCL_cvHaarDetectObjectsForROC( const CvArr* _img,
                     CvHaarClassifierCascade* cascade, CvMemStorage* storage,
                     float scaleFactor, int minNeighbors, int flags,
                     CvSize minSize, CvSize maxSize)
{
    const double GROUP_EPS = 0.2;
    CvMat stub, *img = (CvMat*)_img;
    cv::Ptr<CvMat> temp, sum, tilted, sqsum, normImg, imgSmall;
    CvSeq* result_seq = 0;
    cv::Ptr<CvMemStorage> temp_storage;

    std::vector<cv::Rect> allCandidates;
    std::vector<cv::Rect> rectList;
    std::vector<int> rweights;
    int coi;
    bool doCannyPruning = (flags & CV_HAAR_DO_CANNY_PRUNING) != 0;
    bool findBiggestObject = (flags & CV_HAAR_FIND_BIGGEST_OBJECT) != 0;
    bool roughSearch = (flags & CV_HAAR_DO_ROUGH_SEARCH) != 0;
    testing::internal::Mutex mtx;


    img = cvGetMat( img, &stub, &coi );
    if( maxSize.height == 0 || maxSize.width == 0 )
    {
        maxSize.height = img->rows;
        maxSize.width = img->cols;
    }

    temp = cvCreateMat( img->rows, img->cols, CV_8UC1 );
    sum = cvCreateMat( img->rows + 1, img->cols + 1, CV_32SC1 );
    sqsum = cvCreateMat( img->rows + 1, img->cols + 1, CV_64FC1 );
    if( !cascade->hid_cascade )
        icvCreateHidHaarClassifierCascade(cascade);


    if( cascade->hid_cascade->has_tilted_features )
        tilted = cvCreateMat( img->rows + 1, img->cols + 1, CV_32SC1 );

    result_seq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvAvgComp), storage );

    if( CV_MAT_CN(img->type) > 1 )
    {
        cvCvtColor( img, temp, CV_BGR2GRAY );
        img = temp;
    }

    if( findBiggestObject )
        flags &= ~(CV_HAAR_SCALE_IMAGE|CV_HAAR_DO_CANNY_PRUNING);

    vector<numbers_t> rects;
    vector<NewHidHaarClassifierCascade*> new_cascades;
    vector<Mat*> sums, tilteds;
    CvSize winSize0 = cascade->orig_window_size;
    imgSmall = cvCreateMat( img->rows + 1, img->cols + 1, CV_8UC1 );
    //different scales
    int num_scales = 0;
    for(float factor = 1; ; factor *= scaleFactor )
    {
        //winSize0: original window size in xml
        //winSize: window size
        //sz: scaledImageSize
        //sz1: processingRectSize
        CvSize winSize = { cvRound(winSize0.width*factor),
                            cvRound(winSize0.height*factor) };
        CvSize sz = { cvRound( img->cols/factor ), cvRound( img->rows/factor ) };
        CvSize sz1 = { sz.width - winSize0.width + 1, sz.height - winSize0.height + 1 };

        CvRect equRect = { icv_object_win_border, icv_object_win_border,
            winSize0.width - icv_object_win_border*2,
            winSize0.height - icv_object_win_border*2 };

        CvMat img1, sum1, sqsum1, norm1, tilted1, mask1;
        CvMat* _tilted = 0;

        if( sz1.width <= 0 || sz1.height <= 0 ) 
            break;
        if( winSize.width > maxSize.width || winSize.height > maxSize.height ) {  

            break;}
        if( winSize.width < minSize.width || winSize.height < minSize.height ) 
            continue;
        img1 = cvMat( sz.height, sz.width, CV_8UC1, imgSmall->data.ptr );
        sum1 = cvMat( sz.height+1, sz.width+1, CV_32SC1, sum->data.ptr );
        sqsum1 = cvMat( sz.height+1, sz.width+1, CV_64FC1, sqsum->data.ptr );
        if( tilted )
        {
            tilted1 = cvMat( sz.height+1, sz.width+1, CV_32SC1, tilted->data.ptr );
            _tilted = &tilted1;
        }
        cvResize( img, &img1, CV_INTER_LINEAR );
        cvIntegral( &img1, &sum1, &sqsum1, _tilted );

        int ystep = factor > 2 ? 1 : 2;
        const int LOCS_PER_THREAD = 1000;
        int stripCount = ((sz1.width/ystep)*(sz1.height + ystep-1)/ystep + LOCS_PER_THREAD/2)/LOCS_PER_THREAD;
        stripCount = std::min(std::max(stripCount, 1), 100);
        CL_cvSetImagesForHaarClassifierCascade( cascade, &sum1, &sqsum1, _tilted, 1. );
        CvHidHaarClassifierCascade* hid_cascade = cascade->hid_cascade;
        new_cascades.push_back(convertPointerToNonPointer(hid_cascade));
        //cout << hid_cascade->sum.step << endl;
        Mat* tmp_mat = new Mat(&hid_cascade->sum, 1);
        sums.push_back(tmp_mat);
        tmp_mat = new Mat(&hid_cascade->tilted, 1);
        tilteds.push_back(tmp_mat);

        // invoker
        int stripSize = (((sz1.height + stripCount - 1)/stripCount + ystep-1)/ystep)*ystep;
        int y1 = 0*stripSize, y2 = min(stripCount*stripSize, sum1.rows - 1 - winSize0.height);

        if (y2 <= y1 || sum1.cols <= 1 + winSize0.width)
          continue;

        Size ssz(sum1.cols - 1 - winSize0.width, y2 - y1);
        int x, y;

        for( y = y1; y < y2; y += ystep ) {
          for( x = 0; x < ssz.width; x += ystep ) {
            double real_wsize_width = cascade->real_window_size.width;
            double real_wsize_height = cascade->real_window_size.height;

            if( x < 0 || y < 0 ||
                x + real_wsize_width >= hid_cascade->sum.width ||
                y + real_wsize_height >= hid_cascade->sum.height )
                continue;
            
            numbers_t tmp;
	    tmp.factor = factor;
            tmp.scale_num = num_scales;//cout<<"num_scales "<<num_scales<<endl;
            tmp.y = y;
            tmp.x = x;
//cout<<"x: "<<x<<" y: "<<y<<" facotr: "<<(float)pow((float)scaleFactor, (float)num_scales)<<endl;
            int p_offset, pq_offset;
            p_offset = y * (hid_cascade->sum.step/sizeof(sumtype)) + x;
            pq_offset = y * (hid_cascade->sqsum.step/sizeof(sqsumtype)) + x;
            float mean = calc_sum(*hid_cascade,p_offset)*hid_cascade->inv_window_area;
            float variance_norm_factor = hid_cascade->pq0[pq_offset] - hid_cascade->pq1[pq_offset] - 
                                       hid_cascade->pq2[pq_offset] + hid_cascade->pq3[pq_offset];
            variance_norm_factor = variance_norm_factor*hid_cascade->inv_window_area - mean*mean;
            if( variance_norm_factor >= 0. )
                variance_norm_factor = sqrt(variance_norm_factor);
            else
                variance_norm_factor = 1.;

            tmp.variance_norm_factor = variance_norm_factor;
//cout<<"vnf: "<<variance_norm_factor<<endl;

            rects.push_back(tmp);
          }
        }
        num_scales++;
    }
    int num_rects = rects.size();
    int rects_arr[num_rects][3];
    float vnf[num_rects];
    for (int i = 0; i < rects.size(); ++i) {
      rects_arr[i][0] = rects[i].scale_num;
      rects_arr[i][1] = rects[i].y;
      rects_arr[i][2] = rects[i].x;
      vnf[i] = rects[i].variance_norm_factor;
//if((rects[i].scale_num==3)&&(rects[i].x==34)&&(rects[i].y==63)) cout<<"tag "<<i<<endl;//assert(0);
    }//assert(0);
    NewHidHaarClassifierCascade* new_cascade_list = new NewHidHaarClassifierCascade[num_scales];
    for (int i = 0; i < num_scales; ++i) {
      new_cascade_list[i] = *new_cascades[i];
      delete new_cascades[i];
    }
  // assert(0); 
    bool result_list[num_rects];
    int sum_mat_size, tilted_mat_size;
    int *sum_header, *tilted_mat_header;
    uchar* sum_mat_list = encodeMatrix(sum_mat_size, sum_header, sums);
    uchar* tilted_mat_list = encodeMatrix(tilted_mat_size, tilted_mat_header, tilteds);
    delete[] tilted_mat_header;
    for ( int k = 0; k < sums.size(); k++) {
	if( tilteds.size() > 0) delete tilteds[k];
   	delete sums[k];
    }
    //cli

    double cl_t = (double)cvGetTickCount();


    cl_kernel cascadesum = clCreateKernel(x_prog, "cascadesum", &err);
    check(err);
    cl_mem rects_buffer = clCreateBuffer(
        x_context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        3 * num_rects * sizeof(int), rects_arr, &err);
    check(err);
    cl_mem vnf_buffer = clCreateBuffer(
        x_context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        num_rects * sizeof(float), vnf, &err);
    check(err);
    cl_mem classifier_buffer = clCreateBuffer(
        x_context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        num_scales * sizeof(NewHidHaarClassifierCascade), new_cascade_list, &err);
    delete[] new_cascade_list;
    check(err);
    cl_mem sum_buffer = clCreateBuffer(
        x_context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sum_mat_size * sizeof(uchar), sum_mat_list, &err);
    check(err);
    delete[] sum_mat_list;
/*    cl_mem tilted_buffer = clCreateBuffer(
        x_context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        0 * sizeof(uchar), NULL, &err);//printf("tilted size is : %d\n", tilted_mat_size);
check(err);*/
    delete[] tilted_mat_list;
    cl_mem mat_len_buffer = clCreateBuffer(
        x_context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(int), &sum_mat_size, &err);
    check(err);
    cl_mem mat_header_buffer = clCreateBuffer(
        x_context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (sum_header[0] * 3 + 1) * sizeof(int), sum_header, &err);
    check(err);
    delete[] sum_header;
    cl_mem result_buffer = clCreateBuffer(
        x_context,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        num_rects * sizeof(bool), result_list, &err);
    check(err);
    cl_mem actual_buf = clCreateBuffer(
	x_context,
	CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	sizeof(int), &num_rects, &err);

    err = clSetKernelArg(cascadesum, 0, sizeof(cl_mem), &rects_buffer);
    err = clSetKernelArg(cascadesum, 1, sizeof(cl_mem), &vnf_buffer);
    err = clSetKernelArg(cascadesum, 2, sizeof(cl_mem), &classifier_buffer);
    err = clSetKernelArg(cascadesum, 3, sizeof(cl_mem), &sum_buffer);
//    cout << "here1\n";
    err = clSetKernelArg(cascadesum, 4, sizeof(cl_mem), 0);
//    cout << "here2\n";
    err = clSetKernelArg(cascadesum, 5, sizeof(cl_mem), &mat_len_buffer);
    err = clSetKernelArg(cascadesum, 6, sizeof(cl_mem), &mat_header_buffer);
    err = clSetKernelArg(cascadesum, 7, sizeof(cl_mem), &result_buffer);
    err = clSetKernelArg(cascadesum, 8, sizeof(cl_mem), &actual_buf);
    
    double cl_t1 = (double)cvGetTickCount() - cl_t;
    printf( "buf transfer time = %g ms\n", cl_t1/((double)cvGetTickFrequency()*1000.) );
    const size_t global_size[] = {num_rects-num_rects%16};
    const size_t local_size[] = {16};
    check(
      clEnqueueNDRangeKernel(
        x_cmd_q,
        cascadesum,
        1,
        NULL,
        global_size,
        local_size,
        0,
        NULL,
        NULL
      )
    );
    check(clFinish(x_cmd_q));
    check( clEnqueueReadBuffer(x_cmd_q, result_buffer, CL_TRUE, 0, num_rects * sizeof(bool),
        result_list, 0, NULL, NULL));
    //release CL objects
    clReleaseMemObject(rects_buffer);
    clReleaseMemObject(vnf_buffer);
    clReleaseMemObject(classifier_buffer);
    clReleaseMemObject(sum_buffer);
    clReleaseMemObject(mat_len_buffer);
    clReleaseMemObject(mat_header_buffer);
    clReleaseMemObject(result_buffer);
    clReleaseMemObject(actual_buf);
    clReleaseKernel(cascadesum);
   
    cl_t = (double)cvGetTickCount() - cl_t;
    printf( "OpenCL time = %g ms\n", cl_t/((double)cvGetTickFrequency()*1000.) );
    //??
    for (int i = 0; i < num_rects; ++i) {
      if (result_list[i]) {
//cout<<"x: "<<rects[i].x<<" y: "<<rects[i].y<<" factor: " << rects[i].factor<<endl;
        float factor = rects[i].factor;
        Size winSize(cvRound(cascade->orig_window_size.width*factor), cvRound(cascade->orig_window_size.height*factor));
        allCandidates.push_back(Rect(cvRound(rects[i].x*factor), cvRound(rects[i].y*factor),
                                winSize.width, winSize.height));
      }
    }

    rectList.resize(allCandidates.size());
    if(!allCandidates.empty())
        std::copy(allCandidates.begin(), allCandidates.end(), rectList.begin());

    //??

    if( minNeighbors != 0 || findBiggestObject )
    {
            groupRectangles(rectList, rweights, std::max(minNeighbors, 1), GROUP_EPS);
    }

    for( size_t i = 0; i < rectList.size(); i++ )
    {
        CvAvgComp c;
        c.rect = rectList[i];
        c.neighbors = !rweights.empty() ? rweights[i] : 0;
        cvSeqPush( result_seq, &c );
    }

    return result_seq;
}





CvHidHaarClassifierCascade*
icvCreateHidHaarClassifierCascade( CvHaarClassifierCascade* cascade )
{
    CvRect* ipp_features = 0;
    float *ipp_weights = 0, *ipp_thresholds = 0, *ipp_val1 = 0, *ipp_val2 = 0;
    int* ipp_counts = 0;

    CvHidHaarClassifierCascade* out = 0;

    int i, j, k, l;
    int datasize;
    int total_classifiers = 0;
    int total_nodes = 0;
    char errorstr[1000];
    CvHidHaarClassifier* haar_classifier_ptr;
    CvHidHaarTreeNode* haar_node_ptr;
    CvSize orig_window_size;
    int has_tilted_features = 0;
    int max_count = 0;

    if( !CV_IS_HAAR_CLASSIFIER(cascade) )
        CV_Error( !cascade ? CV_StsNullPtr : CV_StsBadArg, "Invalid classifier pointer" );

    if( cascade->hid_cascade )
        CV_Error( CV_StsError, "hid_cascade has been already created" );

    if( !cascade->stage_classifier )
        CV_Error( CV_StsNullPtr, "" );

    if( cascade->count <= 0 )
        CV_Error( CV_StsOutOfRange, "Negative number of cascade stages" );

    orig_window_size = cascade->orig_window_size;

    /* check input structure correctness and calculate total memory size needed for
       internal representation of the classifier cascade */
    for( i = 0; i < cascade->count; i++ )
    {
        CvHaarStageClassifier* stage_classifier = cascade->stage_classifier + i;

        if( !stage_classifier->classifier ||
            stage_classifier->count <= 0 )
        {
            sprintf( errorstr, "header of the stage classifier #%d is invalid "
                     "(has null pointers or non-positive classfier count)", i );
            CV_Error( CV_StsError, errorstr );
        }

        max_count = MAX( max_count, stage_classifier->count );
        total_classifiers += stage_classifier->count;

        for( j = 0; j < stage_classifier->count; j++ )
        {
            CvHaarClassifier* classifier = stage_classifier->classifier + j;

            total_nodes += classifier->count;
            for( l = 0; l < classifier->count; l++ )
            {
                for( k = 0; k < CV_HAAR_FEATURE_MAX; k++ )
                {
                    if( classifier->haar_feature[l].rect[k].r.width )
                    {
                        CvRect r = classifier->haar_feature[l].rect[k].r;
                        int tilted = classifier->haar_feature[l].tilted;
                        has_tilted_features |= tilted != 0;
                        if( r.width < 0 || r.height < 0 || r.y < 0 ||
                            r.x + r.width > orig_window_size.width
                            ||
                            (!tilted &&
                            (r.x < 0 || r.y + r.height > orig_window_size.height))
                            ||
                            (tilted && (r.x - r.height < 0 ||
                            r.y + r.width + r.height > orig_window_size.height)))
                        {
                            sprintf( errorstr, "rectangle #%d of the classifier #%d of "
                                     "the stage classifier #%d is not inside "
                                     "the reference (original) cascade window", k, j, i );
                            CV_Error( CV_StsNullPtr, errorstr );
                        }
                    }
                }
            }
        }
    }

    // this is an upper boundary for the whole hidden cascade size
    datasize = sizeof(CvHidHaarClassifierCascade) +
               sizeof(CvHidHaarStageClassifier)*cascade->count +
               sizeof(CvHidHaarClassifier) * total_classifiers +
               sizeof(CvHidHaarTreeNode) * total_nodes +
               sizeof(void*)*(total_nodes + total_classifiers);

    out = (CvHidHaarClassifierCascade*)cvAlloc( datasize );
    memset( out, 0, sizeof(*out) );

    /* init header */
    out->count = cascade->count;
    out->stage_classifier = (CvHidHaarStageClassifier*)(out + 1);
    haar_classifier_ptr = (CvHidHaarClassifier*)(out->stage_classifier + cascade->count);
    haar_node_ptr = (CvHidHaarTreeNode*)(haar_classifier_ptr + total_classifiers);

    out->isStumpBased = 1;
    out->has_tilted_features = has_tilted_features;
    out->is_tree = 0;

    /* initialize internal representation */
    for( i = 0; i < cascade->count; i++ )
    {
        CvHaarStageClassifier* stage_classifier = cascade->stage_classifier + i;
        CvHidHaarStageClassifier* hid_stage_classifier = out->stage_classifier + i;

        hid_stage_classifier->count = stage_classifier->count;
        hid_stage_classifier->threshold = stage_classifier->threshold - icv_stage_threshold_bias;
        hid_stage_classifier->classifier = haar_classifier_ptr;
        hid_stage_classifier->two_rects = 1;
        haar_classifier_ptr += stage_classifier->count;

        hid_stage_classifier->parent = (stage_classifier->parent == -1)
            ? NULL : out->stage_classifier + stage_classifier->parent;
        hid_stage_classifier->next = (stage_classifier->next == -1)
            ? NULL : out->stage_classifier + stage_classifier->next;
        hid_stage_classifier->child = (stage_classifier->child == -1)
            ? NULL : out->stage_classifier + stage_classifier->child;

        out->is_tree |= hid_stage_classifier->next != NULL;

        for( j = 0; j < stage_classifier->count; j++ )
        {
            CvHaarClassifier* classifier = stage_classifier->classifier + j;
            CvHidHaarClassifier* hid_classifier = hid_stage_classifier->classifier + j;
            int node_count = classifier->count;
            float* alpha_ptr = (float*)(haar_node_ptr + node_count);

            hid_classifier->count = node_count;
            hid_classifier->node = haar_node_ptr;
            hid_classifier->alpha = alpha_ptr;

            for( l = 0; l < node_count; l++ )
            {
                CvHidHaarTreeNode* node = hid_classifier->node + l;
                CvHaarFeature* feature = classifier->haar_feature + l;
                memset( node, -1, sizeof(*node) );
                node->threshold = classifier->threshold[l];
                node->left = classifier->left[l];
                node->right = classifier->right[l];

                if( fabs(feature->rect[2].weight) < DBL_EPSILON ||
                    feature->rect[2].r.width == 0 ||
                    feature->rect[2].r.height == 0 )
                    memset( &(node->feature.rect[2]), 0, sizeof(node->feature.rect[2]) );
                else
                    hid_stage_classifier->two_rects = 0;
            }

            memcpy( alpha_ptr, classifier->alpha, (node_count+1)*sizeof(alpha_ptr[0]));
            haar_node_ptr =
                (CvHidHaarTreeNode*)cvAlignPtr(alpha_ptr+node_count+1, sizeof(void*));

            out->isStumpBased &= node_count == 1;
        }
    }


    cascade->hid_cascade = out;
    assert( (char*)haar_node_ptr - (char*)out <= datasize );

    cvFree( &ipp_features );
    cvFree( &ipp_weights );
    cvFree( &ipp_thresholds );
    cvFree( &ipp_val1 );
    cvFree( &ipp_val2 );
    cvFree( &ipp_counts );

    return out;
}

void CL_cvSetImagesForHaarClassifierCascade( CvHaarClassifierCascade* _cascade,
                                     const CvArr* _sum,
                                     const CvArr* _sqsum,
                                     const CvArr* _tilted_sum,
                                     double scale )
{
    CvMat sum_stub, *sum = (CvMat*)_sum;
    CvMat sqsum_stub, *sqsum = (CvMat*)_sqsum;
    CvMat tilted_stub, *tilted = (CvMat*)_tilted_sum;
    CvHidHaarClassifierCascade* cascade;
    int coi0 = 0, coi1 = 0;
    int i;
    CvRect equRect;
    double weight_scale;

    if( !CV_IS_HAAR_CLASSIFIER(_cascade) )
        CV_Error( !_cascade ? CV_StsNullPtr : CV_StsBadArg, "Invalid classifier pointer" );

    if( scale <= 0 )
        CV_Error( CV_StsOutOfRange, "Scale must be positive" );

    sum = cvGetMat( sum, &sum_stub, &coi0 );
    sqsum = cvGetMat( sqsum, &sqsum_stub, &coi1 );

    if( coi0 || coi1 )
        CV_Error( CV_BadCOI, "COI is not supported" );

    if( !CV_ARE_SIZES_EQ( sum, sqsum ))
        CV_Error( CV_StsUnmatchedSizes, "All integral images must have the same size" );

    if( CV_MAT_TYPE(sqsum->type) != CV_64FC1 ||
        CV_MAT_TYPE(sum->type) != CV_32SC1 )
        CV_Error( CV_StsUnsupportedFormat,
        "Only (32s, 64f, 32s) combination of (sum,sqsum,tilted_sum) formats is allowed" );

    if( !_cascade->hid_cascade )
        icvCreateHidHaarClassifierCascade(_cascade);

    cascade = _cascade->hid_cascade;

    if( cascade->has_tilted_features )
    {
        tilted = cvGetMat( tilted, &tilted_stub, &coi1 );

        if( CV_MAT_TYPE(tilted->type) != CV_32SC1 )
            CV_Error( CV_StsUnsupportedFormat,
            "Only (32s, 64f, 32s) combination of (sum,sqsum,tilted_sum) formats is allowed" );

        if( sum->step != tilted->step )
            CV_Error( CV_StsUnmatchedSizes,
            "Sum and tilted_sum must have the same stride (step, widthStep)" );

        if( !CV_ARE_SIZES_EQ( sum, tilted ))
            CV_Error( CV_StsUnmatchedSizes, "All integral images must have the same size" );
        cascade->tilted = *tilted;
    }

    _cascade->scale = scale;
    _cascade->real_window_size.width = cvRound( _cascade->orig_window_size.width * scale );
    _cascade->real_window_size.height = cvRound( _cascade->orig_window_size.height * scale );

    cascade->sum = *sum;
    cascade->sqsum = *sqsum;

    equRect.x = equRect.y = cvRound(scale);
    equRect.width = cvRound((_cascade->orig_window_size.width-2)*scale);
    equRect.height = cvRound((_cascade->orig_window_size.height-2)*scale);
    weight_scale = 1./(equRect.width*equRect.height);
    cascade->inv_window_area = weight_scale;

    cascade->p0 = sum_elem_ptr(*sum, equRect.y, equRect.x);
    cascade->p1 = sum_elem_ptr(*sum, equRect.y, equRect.x + equRect.width );
    cascade->p2 = sum_elem_ptr(*sum, equRect.y + equRect.height, equRect.x );
    cascade->p3 = sum_elem_ptr(*sum, equRect.y + equRect.height,
                                     equRect.x + equRect.width );
    cascade->p0_loc[0] = equRect.y;
    cascade->p0_loc[1] = equRect.x;
    cascade->p1_loc[0] = equRect.y;
    cascade->p1_loc[1] = equRect.x + equRect.width;
    cascade->p2_loc[0] = equRect.y + equRect.height;
    cascade->p2_loc[1] = equRect.x;
    cascade->p3_loc[0] = equRect.y + equRect.height;
    cascade->p3_loc[1] = equRect.x + equRect.width;

    cascade->pq0 = sqsum_elem_ptr(*sqsum, equRect.y, equRect.x);
    cascade->pq1 = sqsum_elem_ptr(*sqsum, equRect.y, equRect.x + equRect.width );
    cascade->pq2 = sqsum_elem_ptr(*sqsum, equRect.y + equRect.height, equRect.x );
    cascade->pq3 = sqsum_elem_ptr(*sqsum, equRect.y + equRect.height,
                                          equRect.x + equRect.width );

    /* init pointers in haar features according to real window size and
       given image pointers */
    for( i = 0; i < _cascade->count; i++ )
    {
        int j, k, l;
        for( j = 0; j < cascade->stage_classifier[i].count; j++ )
        {
            for( l = 0; l < cascade->stage_classifier[i].classifier[j].count; l++ )
            {
                CvHaarFeature* feature =
                    &_cascade->stage_classifier[i].classifier[j].haar_feature[l];
                /* CvHidHaarClassifier* classifier =
                    cascade->stage_classifier[i].classifier + j; */
                CvHidHaarFeature* hidfeature =
                    &cascade->stage_classifier[i].classifier[j].node[l].feature;
                double sum0 = 0, area0 = 0;
                CvRect r[3];

                int base_w = -1, base_h = -1;
                int new_base_w = 0, new_base_h = 0;
                int kx, ky;
                int flagx = 0, flagy = 0;
                int x0 = 0, y0 = 0;
                int nr;

                /* align blocks */
                for( k = 0; k < CV_HAAR_FEATURE_MAX; k++ )
                {
                    if( !hidfeature->rect[k].p0 )
                        break;
                    r[k] = feature->rect[k].r;
                    base_w = (int)CV_IMIN( (unsigned)base_w, (unsigned)(r[k].width-1) );
                    base_w = (int)CV_IMIN( (unsigned)base_w, (unsigned)(r[k].x - r[0].x-1) );
                    base_h = (int)CV_IMIN( (unsigned)base_h, (unsigned)(r[k].height-1) );
                    base_h = (int)CV_IMIN( (unsigned)base_h, (unsigned)(r[k].y - r[0].y-1) );
                }

                nr = k;

                base_w += 1;
                base_h += 1;
                kx = r[0].width / base_w;
                ky = r[0].height / base_h;

                if( kx <= 0 )
                {
                    flagx = 1;
                    new_base_w = cvRound( r[0].width * scale ) / kx;
                    x0 = cvRound( r[0].x * scale );
                }

                if( ky <= 0 )
                {
                    flagy = 1;
                    new_base_h = cvRound( r[0].height * scale ) / ky;
                    y0 = cvRound( r[0].y * scale );
                }

                for( k = 0; k < nr; k++ )
                {
                    CvRect tr;
                    double correction_ratio;

                    if( flagx )
                    {
                        tr.x = (r[k].x - r[0].x) * new_base_w / base_w + x0;
                        tr.width = r[k].width * new_base_w / base_w;
                    }
                    else
                    {
                        tr.x = cvRound( r[k].x * scale );
                        tr.width = cvRound( r[k].width * scale );
                    }

                    if( flagy )
                    {
                        tr.y = (r[k].y - r[0].y) * new_base_h / base_h + y0;
                        tr.height = r[k].height * new_base_h / base_h;
                    }
                    else
                    {
                        tr.y = cvRound( r[k].y * scale );
                        tr.height = cvRound( r[k].height * scale );
                    }

#if CV_ADJUST_WEIGHTS
                    {
                    // RAINER START
                    const float orig_feature_size =  (float)(feature->rect[k].r.width)*feature->rect[k].r.height;
                    const float orig_norm_size = (float)(_cascade->orig_window_size.width)*(_cascade->orig_window_size.height);
                    const float feature_size = float(tr.width*tr.height);
                    //const float normSize    = float(equRect.width*equRect.height);
                    float target_ratio = orig_feature_size / orig_norm_size;
                    //float isRatio = featureSize / normSize;
                    //correctionRatio = targetRatio / isRatio / normSize;
                    correction_ratio = target_ratio / feature_size;
                    // RAINER END
                    }
#else
                    correction_ratio = weight_scale * (!feature->tilted ? 1 : 0.5);
#endif

                    hidfeature->tilted = feature->tilted;
                    if( !feature->tilted )
                    {
                        hidfeature->rect[k].p0 = sum_elem_ptr(*sum, tr.y, tr.x);
                        hidfeature->rect[k].p1 = sum_elem_ptr(*sum, tr.y, tr.x + tr.width);
                        hidfeature->rect[k].p2 = sum_elem_ptr(*sum, tr.y + tr.height, tr.x);
                        hidfeature->rect[k].p3 = sum_elem_ptr(*sum, tr.y + tr.height, tr.x + tr.width);
                        hidfeature->rect[k].p0_loc[0] = tr.y;
                        hidfeature->rect[k].p0_loc[1] = tr.x;
                        hidfeature->rect[k].p1_loc[0] = tr.y;
                        hidfeature->rect[k].p1_loc[1] = tr.x + tr.width;
                        hidfeature->rect[k].p2_loc[0] = tr.y + tr.height;
                        hidfeature->rect[k].p2_loc[1] = tr.x;
                        hidfeature->rect[k].p3_loc[0] = tr.y + tr.height;
                        hidfeature->rect[k].p3_loc[1] = tr.x + tr.width;
                    }
                    else
                    {
                        hidfeature->rect[k].p2 = sum_elem_ptr(*tilted, tr.y + tr.width, tr.x + tr.width);
                        hidfeature->rect[k].p3 = sum_elem_ptr(*tilted, tr.y + tr.width + tr.height,
                                                              tr.x + tr.width - tr.height);
                        hidfeature->rect[k].p0 = sum_elem_ptr(*tilted, tr.y, tr.x);
                        hidfeature->rect[k].p1 = sum_elem_ptr(*tilted, tr.y + tr.height, tr.x - tr.height);
                        hidfeature->rect[k].p0_loc[0] = tr.y;
                        hidfeature->rect[k].p0_loc[1] = tr.x;
                        hidfeature->rect[k].p1_loc[0] = tr.y + tr.height;
                        hidfeature->rect[k].p1_loc[1] = tr.x - tr.height;
                        hidfeature->rect[k].p2_loc[0] = tr.y + tr.width;
                        hidfeature->rect[k].p2_loc[1] = tr.x + tr.width;
                        hidfeature->rect[k].p3_loc[0] = tr.y + tr.height;
                        hidfeature->rect[k].p3_loc[1] = tr.x + tr.width;
                    }

                    hidfeature->rect[k].weight = (float)(feature->rect[k].weight * correction_ratio);

                    if( k == 0 )
                        area0 = tr.width * tr.height;
                    else
                        sum0 += hidfeature->rect[k].weight * tr.width * tr.height;
                }

                hidfeature->rect[0].weight = (float)(-sum0/area0);
            } /* l */
        } /* j */
    }
}
