#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/ts/ts_gtest.h"
//#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc_c.h"
//#include "/home/yang/opencvMethod/opencv-2.4.5/modules/core/include/opencv2/core/core.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/internal.hpp"
#include <cstdio>
#include <string>
#include <stdio.h>
const float icv_stage_threshold_bias = 0.0001f;
const int icv_object_win_border = 1;
using namespace cv;
struct getRect { Rect operator ()(const CvAvgComp& e) const { return e.rect; } };

#define calc_sum(rect,offset) \
    ((rect).p0[offset] - (rect).p1[offset] - (rect).p2[offset] + (rect).p3[offset])

#define calc_sumf(rect,offset) \
    static_cast<float>((rect).p0[offset] - (rect).p1[offset] - (rect).p2[offset] + (rect).p3[offset])

typedef int sumtype;
typedef double sqsumtype;

typedef struct CvHidHaarFeature
{
    int tilted;
    struct
    {
        sumtype *p0, *p1, *p2, *p3;
        int p0_loc[2];
        int p1_loc[2];
        int p2_loc[2];
        int p3_loc[2];
        float weight;
    }
    rect[CV_HAAR_FEATURE_MAX];
} CvHidHaarFeature;


typedef struct CvHidHaarTreeNode
{
    CvHidHaarFeature feature;
    float threshold;
    int left;
    int right;
} CvHidHaarTreeNode;


typedef struct CvHidHaarClassifier
{
    int count;
    //CvHaarFeature* orig_feature;
    CvHidHaarTreeNode* node;
    float* alpha;
} CvHidHaarClassifier;


typedef struct CvHidHaarStageClassifier
{
    int  count;
    float threshold;
    CvHidHaarClassifier* classifier;
    int two_rects;

    struct CvHidHaarStageClassifier* next;
    struct CvHidHaarStageClassifier* child;
    struct CvHidHaarStageClassifier* parent;
} CvHidHaarStageClassifier;


typedef struct CvHidHaarClassifierCascade
{
    int  count;
    int  isStumpBased;
    int  has_tilted_features;
    int  is_tree;
    double inv_window_area;
    CvMat sum, sqsum, tilted;
    CvHidHaarStageClassifier* stage_classifier;
    sqsumtype *pq0, *pq1, *pq2, *pq3;
    sumtype *p0, *p1, *p2, *p3;
    int p0_loc[2];
    int p1_loc[2];
    int p2_loc[2];
    int p3_loc[2];

    void** ipp_stages;
} CvHidHaarClassifierCascade;
/*
void
CL_cvSetImagesForHaarClassifierCascade( CvHaarClassifierCascade* _cascade,
                                     const CvArr* _sum,
                                     const CvArr* _sqsum,
                                     const CvArr* _tilted_sum,
                                     double scale );
*/
/*
typedef struct CvHidHaarFeature
{
    struct
    {
        sumtype *p0, *p1, *p2, *p3;
        float weight;
    }
    rect[CV_HAAR_FEATURE_MAX];
} CvHidHaarFeature;


typedef struct CvHidHaarTreeNode
{
    CvHidHaarFeature feature;
    float threshold;
    int left;
    int right;
} CvHidHaarTreeNode;


typedef struct CvHidHaarClassifier
{
    int count;
    //CvHaarFeature* orig_feature;
    CvHidHaarTreeNode* node;
    float* alpha;
} CvHidHaarClassifier;

typedef struct CvHidHaarStageClassifier
{
    int  count;
    float threshold;
    CvHidHaarClassifier* classifier;
    int two_rects;

    struct CvHidHaarStageClassifier* next;
    struct CvHidHaarStageClassifier* child;
    struct CvHidHaarStageClassifier* parent;
} CvHidHaarStageClassifier;


typedef struct CvHidHaarClassifierCascade
{
    int  count;
    int  isStumpBased;
    int  has_tilted_features;
    int  is_tree;
    double inv_window_area;
    CvMat sum, sqsum, tilted;
    CvHidHaarStageClassifier* stage_classifier;
    sqsumtype *pq0, *pq1, *pq2, *pq3;
    sumtype *p0, *p1, *p2, *p3;

    void** ipp_stages;
} CvHidHaarClassifierCascade;
*/

class OCL_CascadeClassifier;

void detectAndDraw( Mat& img, OCL_CascadeClassifier& cascade,
                    OCL_CascadeClassifier& nestedCascade,
                    double scale, bool tryflip );

CvHidHaarClassifierCascade* 
icvCreateHidHaarClassifierCascade( CvHaarClassifierCascade* cascade );

CvSeq*
OCL_cvHaarDetectObjectsForROC( const CvArr* _img,
                     CvHaarClassifierCascade* cascade, CvMemStorage* storage,
                     float scaleFactor, int minNeighbors, int flags,
                     CvSize minSize, CvSize maxSize);

void CL_Classifier_Invoker(int stripCount,
                          const CvHaarClassifierCascade* cascade,
                          int stripSize, double factor,
                          const Mat& sum1, const Mat& sqsum1,
                          std::vector<Rect>& _vec,
                          testing::internal::Mutex *mtx,
                          int **sum_ptr = 0,
                          int **tilted_ptr = 0);

static int
CL_cvRunHaarClassifierCascadeSum( const CvHaarClassifierCascade* _cascade,
                                CvPoint pt, double& stage_sum, int start_stage,
                                std::vector<Rect>* vec, double factor, Size winSize,
                                int **sum_ptr = 0,
                                int **tilted_ptr = 0);

static int CascadeSum_ParallelBody(//int x, int y, 
                                      int p_offset, 
                                      int variance_norm_factor,
                                      const CvHidHaarClassifierCascade* cascade,
                                      double& stage_sum, 
                                      int start_stage);

class OCL_CascadeClassifier : public cv::CascadeClassifier 
{
public:
	OCL_CascadeClassifier() { CascadeClassifier(); }
	virtual void CL_detectMultiScale( const Mat& image,
				   CV_OUT vector<Rect>& objects,
				   float scaleFactor=1.1,
				   int minNeighbors=3, int flags=0,
				   Size minSize=Size(),
				   Size maxSize=Size());
};

double getVal(int *list, int step, int x, int y, int p_offset) {
  int origin_offset = y * (step/sizeof(sumtype)) + x; //??? sumtype or tiltedtype the same?
  int offset = origin_offset + p_offset;
  return list[offset];
}

double calcSum(int *sum_list, int sum_step,  //step: row size
            int *tilted_list, int tilted_step,
            int p_offset, CvHidHaarFeature feature, int featureNo) {
  double result;
  if (!feature.tilted) {
    //use sum
    result = getVal(sum_list, sum_step, 
                      feature.rect[featureNo].p0_loc[1],
                      feature.rect[featureNo].p0_loc[0],
                      p_offset) -
            getVal(sum_list, sum_step, 
                      feature.rect[featureNo].p1_loc[1],
                      feature.rect[featureNo].p1_loc[0],
                      p_offset) -
            getVal(sum_list, sum_step, 
                      feature.rect[featureNo].p2_loc[1],
                      feature.rect[featureNo].p2_loc[0],
                      p_offset) +
            getVal(sum_list, sum_step, 
                      feature.rect[featureNo].p3_loc[1],
                      feature.rect[featureNo].p3_loc[0],
                      p_offset);
  } else {
    //use tilted
    result = getVal(tilted_list, tilted_step, 
                      feature.rect[featureNo].p0_loc[1],
                      feature.rect[featureNo].p0_loc[0],
                      p_offset) -
            getVal(tilted_list, tilted_step, 
                      feature.rect[featureNo].p1_loc[1],
                      feature.rect[featureNo].p1_loc[0],
                      p_offset) -
            getVal(tilted_list, tilted_step, 
                      feature.rect[featureNo].p2_loc[1],
                      feature.rect[featureNo].p2_loc[0],
                      p_offset) +
            getVal(tilted_list, tilted_step, 
                      feature.rect[featureNo].p3_loc[1],
                      feature.rect[featureNo].p3_loc[0],
                      p_offset);
  }
  //CV_Assert(0);
  return result;
}
