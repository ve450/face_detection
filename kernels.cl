#pragma OPENCL EXTENSION cl_khr_fp64: enable
typedef int sumtype;
typedef float sqsumtype;

typedef struct rect_t {
  int p0[2], p1[2], p2[2], p3[2];
  float weight;
} rect_t;

typedef struct CvHidHaarFeature
{
    int tilted;
    rect_t rect[3];
} CvHidHaarFeature;


typedef struct CvHidHaarTreeNode
{
    CvHidHaarFeature feature;
    float threshold;
    //int left;
    //int right;
} CvHidHaarTreeNode;


typedef struct CvHidHaarClassifier
{
    //int count;
    //CvHaarFeature* orig_feature;
    CvHidHaarTreeNode node;
    float alpha[2];
} CvHidHaarClassifier;


typedef struct CvHidHaarStageClassifier
{
    int  count;
    float threshold;
    CvHidHaarClassifier classifier[213];
    int two_rects;

} CvHidHaarStageClassifier;


typedef struct CvHidHaarClassifierCascade
{
    int  count;
    int  isStumpBased;
    int  has_tilted_features;
    int  is_tree;
    float inv_window_area;
    CvHidHaarStageClassifier stage_classifier[22];
    int p0[2], p1[2], p2[2], p3[2];

} CvHidHaarClassifierCascade;
/*
int** decodeMatrixInt(uchar* data, int* header) {
	int n = header[0];
	int** result = (int**)malloc(n * sizeof(int*));
	int count = 0;
	for( int i = 0; i < n; i++) 
	{
		int col = header[1 + i * 3];
		int row = header[1 + i * 3 + 1];
		result[i] = (int*)(data + count);
		count += col * row * 4;
	}
	return result;
}*/

int calc_sum(rect_t rect, int col, int offset, int* mat) {
  int p0_off, p1_off, p2_off, p3_off;
  int4 v0 = (int4)(rect.p0[0] * col + rect.p0[1], rect.p1[0] * col + rect.p1[1], rect.p2[0] * col + rect.p2[1], rect.p3[0] * col + rect.p3[1]) + (int4)offset;
  int4 v1 = (int4)(mat[v0.s0],mat[v0.s1],mat[v0.s2],mat[v0.s3]);
  return v1.s0-v1.s1-v1.s2+v1.s3;

  /*
  p0_off = rect.p0[0] * col + rect.p0[1] + offset;
  p1_off = rect.p1[0] * col + rect.p1[1] + offset;
  p2_off = rect.p2[0] * col + rect.p2[1] + offset;
  p3_off = rect.p3[0] * col + rect.p3[1] + offset;
  return mat[p0_off] - mat[p1_off] - mat[p2_off] + mat[p3_off];
  */
}

/*
  rects: rectangles, rects[x][0] -- scale, rects[x][1] -- y, rects[x][2] -- x
  classifiers: classifier list that contains classifiers of all scales.
  sum_list: list of sum matrix that contains sum mat of all scales.
  sqsum_list: list of sum matrix that contains sum mat of all scales.
  tilted_list: list of sum matrix that contains sum mat of all scales.
  result: bool list of whether is face.
*/
__kernel void cascadesum(__global int *rects, __global float *vnf,
                         __global CvHidHaarClassifierCascade *classifiers,
                         __global uchar *sum_list, __global uchar *tilted_list,
                         int mat_len, __global int *mat_header,
                         __global bool *result, __global int* actual_rects) {
  const int id = get_global_id(0);
  if(id >= *actual_rects) return;
  float variance_norm_factor = vnf[id];
  int scale_num = rects[id * 3];
  int y = rects[id * 3 + 1];
  int x = rects[id * 3 + 2];
  __global CvHidHaarClassifierCascade* cascade = &classifiers[scale_num];
  int col = mat_header[1 + scale_num * 3];
  int row = mat_header[1 + scale_num * 3 + 1];
  int* sum_list_int = (int*)sum_list;
  int* tilted_list_int = (int*)tilted_list;

  bool isStumpBased = true;
  float stage_sum;
  int start_stage = 0;
  int p_offset;
  int i, j;
  int mat_list_offset = 0;
  for (i = 0; i < scale_num; ++i) {
      int col_tmp = mat_header[1 + i * 3];
      int row_tmp = mat_header[1 + i * 3 + 1];
      mat_list_offset += col_tmp * row_tmp;
  }
  //official start
  p_offset = y * (col * sizeof(int) /sizeof(sumtype)) + x;

  if( isStumpBased ) {
      for( i = start_stage; i < cascade->count; i++ ) {
          float sumvec[cascade->stage_classifier[i].count];
          stage_sum = 0;
          if( cascade->stage_classifier[i].two_rects ) {
              for( j = 0; j < cascade->stage_classifier[i].count; j++ ) {
                  __global CvHidHaarClassifier* classifier = cascade->stage_classifier[i].classifier + j; //#j classifier in the #i stage
                  CvHidHaarTreeNode node = classifier->node;
                  float t = node.threshold*variance_norm_factor; //true threshold
                  int* mat;
                  if (node.feature.tilted)
                    mat = &tilted_list_int[mat_list_offset];
                  else
                    mat = &sum_list_int[mat_list_offset];
                  float sum = (float)calc_sum(node.feature.rect[0],col,p_offset,mat) * node.feature.rect[0].weight;
                  sum += calc_sum(node.feature.rect[1],col,p_offset,mat) * node.feature.rect[1].weight;
                  stage_sum += classifier->alpha[sum >= t];
                  /* no effect on performance
                  if( stage_sum > cascade->stage_classifier[i].threshold )
                    continue;
                  */
              }
          } else {
              for( j = 0; j < cascade->stage_classifier[i].count; j++ ) {
                  __global CvHidHaarClassifier* classifier = cascade->stage_classifier[i].classifier + j;
                  CvHidHaarTreeNode node = classifier->node;
                  float t = node.threshold*variance_norm_factor;
                  int* mat;
                  if (node.feature.tilted)
                    mat = &tilted_list_int[mat_list_offset];
                  else
                    mat = &sum_list_int[mat_list_offset];
                  float sum = calc_sum(node.feature.rect[0],col,p_offset,mat) * node.feature.rect[0].weight;
                  sum += calc_sum(node.feature.rect[1],col,p_offset,mat) * node.feature.rect[1].weight;
                  if( node.feature.rect[2].p0 )
                      sum += calc_sum(node.feature.rect[2],col,p_offset,mat) * node.feature.rect[2].weight;
                  stage_sum += classifier->alpha[sum >= t];
                  /* no effect on performance
                  if( stage_sum > cascade->stage_classifier[i].threshold )
                    continue;
                  */
              }
          }
          if( stage_sum < cascade->stage_classifier[i].threshold ) {
              result[id] = false;
              return;
          }
      }
  } else {
    result[id] = false;
    return;
  }

  result[id] = true;
  return;
}
