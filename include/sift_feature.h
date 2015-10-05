#ifndef SIFT_FEATURE_H
#define SIFT_FEATURE_H

#include <math.h>
#include <vl/covdet.h>
#include <vl/mathop.h>
#include <vl/sift.h>
#include <vl/liop.h>
#include <assert.h>
#include "parallel.h"

#define BOUND_BOX_DIM 200

#define MAX_NUM_SIFT_FEAT 1000

typedef struct{
	int first; //[first,last)
	int last;  //[first,last)
    float *patch;
    float *patchXY;
    float* tempDesc;
    float *features;
    VlCovDet * covdet;
    vl_index patchResolution;
	vl_size patchSide;
	double patchStep;
    double patchRelativeExtent;
    double patchRelativeSmoothing;
    int dim;
    VlCovDetFeature* covdet_feat;
    VlSiftFilt * sift_filter;
}DESC_THREAD_PARAM;

int sift(unsigned char* img, const int& widthStep, const int& H, const int&W, void* buffer, float* features);
class DESC{
    int type;  // 0 for sift.
    int dim; //dimension of feature.
    float* fdata; //float image.
    int i,j,k;
    int nthreads; //number of parallel threads.
    vl_bool doubleImage;
    vl_index patchResolution;
	vl_size patchSide;
	double patchStep;
    double patchRelativeExtent;
    double patchRelativeSmoothing;
    float *patch;
    float *patchXY;
    float* tempDesc;
    double boundaryMargin; //interesting points on the margin will be dropped.
    VlCovDet * covdet; //covdet object.
    int numFeatures; //number of features detected.
    double PeakThreshold; //threshold to control the number of interesting points.
    VlCovDetFeature* covdet_feat;
    VlSiftFilt * sift_filter; //sift filter.
    Parallel PL;
    DESC_THREAD_PARAM* thread_params;
	vector<void*> vec_params;
public:
    /* DESC: Initializes the feature descriptor DESC object.
     * t[In]: the type of descriptor. 0 for sift features.
     */
    DESC(const int& t);
    ~DESC();
    
    /* EXTRACT_FEAT
     * img[In]: input image provided by opencv.
     * widthStep[In]: widthStep of the img.
     * H[In]: Height of image.
     * W[In]: Width of image.
     * features[Out]: The buffer for features. Each row
     * max_num_features[In]: the size of 'features' buffer. The maximum number of local features it can accomodate.
     * Return value: the number of extracted local features if success. Return -1 if the buffer size is not enough, return -2 for unknown reasons.
     */
    int extract_feat(unsigned char* img, const int& widthStep, const int& H, const int&W, float* features, const int max_num_features);
    
    int get_dim(); //get the dimension of features.
    
    int get_type(); //get the type code of features.
};
#endif
