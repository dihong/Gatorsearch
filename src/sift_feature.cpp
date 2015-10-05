#include "sift_feature.h"
#include "common.h"

using namespace std;

static void flip_descriptor (float *dst, float const *src){
    int const BO = 8 ;  /* number of orientation bins */
    int const BP = 4 ;  /* number of spatial bins     */
    int i, j, t ;
    
    for (j = 0 ; j < BP ; ++j) {
        int jp = BP - 1 - j ;
        for (i = 0 ; i < BP ; ++i) {
            int o  = BO * i + BP*BO * j  ;
            int op = BO * i + BP*BO * jp ;
            dst [op] = src[o] ;
            for (t = 1 ; t < BO ; ++t)
                dst [BO - t + op] = src [t + o] ;
        }
    }
}

void Thread_Calculate_Sift(void* par){
    DESC_THREAD_PARAM* param = (DESC_THREAD_PARAM*) par;
    float *patch = param->patch;
    float *patchXY = param->patchXY;
    float* tempDesc = param->tempDesc;
    float* features = param->features + param->first*param->dim;
    VlCovDet * covdet = param->covdet;
    VlCovDetFeature * covdet_feat = param->covdet_feat;
    vl_index patchResolution = param->patchResolution;
	vl_size patchSide = param->patchSide;
	double patchStep = param->patchStep;
    double patchRelativeExtent = param->patchRelativeExtent;
    double patchRelativeSmoothing = param->patchRelativeSmoothing;
    int dim = param->dim;
    VlSiftFilt * sift_filter = param->sift_filter;
    int i,j;
    for (i = param->first ; i < param->last; ++i) {
        vl_covdet_extract_patch_for_frame(covdet,
                patch,
                patchResolution,
                patchRelativeExtent,
                patchRelativeSmoothing,
                covdet_feat[i].frame) ;
        vl_imgradient_polar_f (patchXY, patchXY +1,
                2, 2 * patchSide,
                patch, patchSide, patchSide, patchSide) ;
        vl_sift_calc_raw_descriptor (sift_filter,
                patchXY,
                tempDesc,
                (int)patchSide, (int)patchSide,
                (double)(patchSide-1) / 2, (double)(patchSide-1) / 2,
                (double)patchRelativeExtent / (3.0 * (4 + 1) / 2) /
                patchStep,
                VL_PI / 2);
        flip_descriptor (features, tempDesc);
        for(j=0;j<dim;j++)
        	features[j] *= 255.0;
        features += dim;
    }
}

DESC::DESC(const int& t){
    type = t;
    nthreads = 12;
    fdata = (float*)malloc(sizeof(float)*BOUND_BOX_DIM*BOUND_BOX_DIM);
    DESC_THREAD_PARAM* thread_params = new DESC_THREAD_PARAM [nthreads];
    if(type==0){ //sift
        dim = 128;
        doubleImage = VL_TRUE ;
        patchResolution = 15;
        patchRelativeExtent = 7.5 ;
        patchRelativeSmoothing = 1 ;
        boundaryMargin = 2.0 ;
        patch = (float*)malloc(nthreads*sizeof(float) * (2*patchResolution + 1) * (2*patchResolution + 1));
        patchXY = (float*)malloc(nthreads* sizeof(float) * 2 * (2*patchResolution + 1) * (2*patchResolution + 1));
        covdet = vl_covdet_new(VL_COVDET_METHOD_DOG);
        vl_covdet_set_transposed(covdet, VL_TRUE);
        vl_covdet_set_first_octave(covdet, doubleImage ? -1 : 0);
        sift_filter = vl_sift_new(16, 16, 1, 3, 0);
        vl_sift_set_magnif(sift_filter, 3.0);
        patchSide = 2 * patchResolution + 1;
        patchStep = (double)patchRelativeExtent / patchResolution;
        tempDesc = new float [dim*nthreads];
        for (k = 0; k < nthreads; k++) {
            thread_params[k].patch = patch + k * (2*patchResolution + 1) * (2*patchResolution + 1);
            thread_params[k].patchXY = patchXY + k* 2 * (2*patchResolution + 1) * (2*patchResolution + 1);
            thread_params[k].tempDesc = tempDesc + k * dim;
			thread_params[k].patchResolution = patchResolution;
			thread_params[k].patchSide = patchSide;
			thread_params[k].patchStep = patchStep;
			thread_params[k].patchRelativeExtent = patchRelativeExtent;
			thread_params[k].patchRelativeSmoothing = patchRelativeSmoothing;
			thread_params[k].dim = dim;
			thread_params[k].sift_filter = sift_filter;
            vec_params.push_back(thread_params + k);
        }
    }
}

DESC::~DESC(){
    if(fdata) free(fdata);
    if(patch) free(patch);
    if(patchXY) free(patchXY);
    if(tempDesc) free(tempDesc);
    if(covdet) vl_covdet_delete (covdet);
    vl_sift_delete(sift_filter);
}

/* EXTRACT_FEAT
 * img[In]: input image provided by opencv.
 * widthStep[In]: widthStep of the img.
 * H[In]: Height of image.
 * W[In]: Width of image.
 * features[Out]: The buffer for features. Each row
 * max_num_features[In]: the size of 'features' buffer. The maximum number of local features it can accomodate.
 * Return value: the number of extracted local features if success. Return -1 if the buffer size is not enough, return -2 for unknown reasons.
 */
int DESC::extract_feat(unsigned char* img, const int& widthStep, const int& H, const int&W, float* features, const int max_num_features){
    /*Check inputs*/
    if(H>BOUND_BOX_DIM || W>BOUND_BOX_DIM){
        printf("Image size too large.\n");
        return -2;
    }
    
    /*Convert image to single*/
    for(j=0;j<H;j++){
        for(k=0;k<W;k++){
            fdata[k*H+j] = img[j*widthStep+k];
        }
    }
    
    /*Determine the best PeakThreshold*/
    {
        i = 0;
        PeakThreshold = 4.0;  ///initial value: 4.0
        vl_covdet_put_image(covdet, fdata, H, W);
        while(i < 5){
            vl_covdet_set_peak_threshold(covdet, PeakThreshold); //set PeakThreshold parameter.
            vl_covdet_detect(covdet);
            vl_covdet_drop_features_outside(covdet, boundaryMargin);
            numFeatures = vl_covdet_get_num_features(covdet);
            if (numFeatures<100)
                PeakThreshold = PeakThreshold*0.2;
            else if (numFeatures>500)
                PeakThreshold = PeakThreshold*1.5;
            else
                break;
            i = i + 1;
        }
    }
    /*Check whether numFeatures is valid.*/
    if(numFeatures<5)
        return -2;
    else if(numFeatures>max_num_features)
        return -1;
    
    /*extract features on detected interesting points in parallel.*/
    if(type==0){
        i = ceil((double)numFeatures / nthreads);
        covdet_feat = (VlCovDetFeature*)vl_covdet_get_features(covdet);
        for (k = 0; k < nthreads; k++) {
            ((DESC_THREAD_PARAM*)vec_params[k])->first = k * i; //offset within this machine.
            if (k < nthreads - 1)
                ((DESC_THREAD_PARAM*)vec_params[k])->last = (k + 1) * i;  //offset within this machine.
            else
                ((DESC_THREAD_PARAM*)vec_params[k])->last = (signed)numFeatures;
            ((DESC_THREAD_PARAM*)vec_params[k])->features = features;
			((DESC_THREAD_PARAM*)vec_params[k])->covdet = covdet;
			((DESC_THREAD_PARAM*)vec_params[k])->covdet_feat = covdet_feat;
        }
        PL.Run(Thread_Calculate_Sift, vec_params);
    }
    
    /*if(type==0){
     * covdet_feat = (VlCovDetFeature*)vl_covdet_get_features(covdet);
     * vl_size patchSide = 2 * patchResolution + 1;
     * double patchStep = (double)patchRelativeExtent / patchResolution ;
     * for (i = 0 ; i < (signed)numFeatures ; ++i) {
     * vl_covdet_extract_patch_for_frame(covdet,
     * patch,
     * patchResolution,
     * patchRelativeExtent,
     * patchRelativeSmoothing,
     * covdet_feat[i].frame) ;
     *
     * vl_imgradient_polar_f (patchXY, patchXY +1,
     * 2, 2 * patchSide,
     * patch, patchSide, patchSide, patchSide) ;
     *
     * vl_sift_calc_raw_descriptor (sift,
     * patchXY,
     * tempDesc,
     * (int)patchSide, (int)patchSide,
     * (double)(patchSide-1) / 2, (double)(patchSide-1) / 2,
     * (double)patchRelativeExtent / (3.0 * (4 + 1) / 2) /
     * patchStep,
     * VL_PI / 2);
     * flip_descriptor (features, tempDesc);
     * features += dim ;
     * }
     * }*/
    
    /*Clean up and return.*/
    vl_covdet_reset(covdet);
    //if (covdet->patch) vl_free (covdet->patch);
    return numFeatures;
}

int DESC::get_dim(){ //get the dimension of features.
    return dim;
}
int DESC::get_type(){ //get the type code of features.
    return type;
}



void vl_covdet_sift(float* desc, vl_size& numFeatures, const float*image, vl_size numRows, vl_size numCols, double PeakThreshold){
    
    vl_bool doubleImage = VL_TRUE ;
    
    vl_index patchResolution = 15;
    double patchRelativeExtent = 7.5 ;
    double patchRelativeSmoothing = 1 ;
    float *patch = NULL ;
    float *patchXY = NULL ;
    
    
    
    double boundaryMargin = 2.0 ;
    
    VlCovDet * covdet = vl_covdet_new(VL_COVDET_METHOD_DOG) ;
    
    /* set covdet parameters */
    vl_covdet_set_transposed(covdet, VL_TRUE);  //images are not transposed in C.
    vl_covdet_set_first_octave(covdet, doubleImage ? -1 : 0) ;
    vl_covdet_set_peak_threshold(covdet, PeakThreshold); //set PeakThreshold parameter.
    
    /* process the image */
    vl_covdet_put_image(covdet, image, numRows, numCols);
    
    /* detect features */
    vl_covdet_detect(covdet) ;
    vl_covdet_drop_features_outside (covdet, boundaryMargin);
    /* store results back */
    numFeatures = vl_covdet_get_num_features(covdet);
    if(MAX_NUM_SIFT_FEAT<numFeatures){
        numFeatures = 0;
        return;
    }
    
    vl_size w = 2*patchResolution + 1 ;
    patch = (float*)malloc(sizeof(float) * w * w);
    patchXY = (float*)malloc(2 * sizeof(float) * w * w);
    
    VlCovDetFeature const * feature = (VlCovDetFeature const *)vl_covdet_get_features(covdet);
    VlSiftFilt * sift = vl_sift_new(16, 16, 1, 3, 0);
    vl_index i ;
    vl_size dimension = 128 ;
    vl_size patchSide = 2 * patchResolution + 1 ;
    double patchStep = (double)patchRelativeExtent / patchResolution ;
    vl_sift_set_magnif(sift, 3.0) ;
    float tempDesc [128] ;
    for (i = 0 ; i < (signed)numFeatures ; ++i) {
        vl_covdet_extract_patch_for_frame(covdet,
                patch,
                patchResolution,
                patchRelativeExtent,
                patchRelativeSmoothing,
                feature[i].frame) ;
        
        vl_imgradient_polar_f (patchXY, patchXY +1,
                2, 2 * patchSide,
                patch, patchSide, patchSide, patchSide) ;
        
        vl_sift_calc_raw_descriptor (sift,
                patchXY,
                tempDesc,
                (int)patchSide, (int)patchSide,
                (double)(patchSide-1) / 2, (double)(patchSide-1) / 2,
                (double)patchRelativeExtent / (3.0 * (4 + 1) / 2) /
                patchStep,
                VL_PI / 2);
        flip_descriptor (desc, tempDesc);
        desc += dimension ;
    }
    vl_sift_delete(sift) ;
    vl_covdet_delete (covdet) ;
    free(patchXY) ;
    free(patch) ;
}




int sift(unsigned char* img, const int& widthStep, const int& H, const int&W, void* buffer, float* features){
    /*Convert image to single*/
    float* fdata = (float*)buffer;
    int i,j,k;
    for(j=0;j<H;j++){
        for(k=0;k<W;k++){
            fdata[k*H+j] = img[j*widthStep+k];
        }
    }
    /*extract features*/
    vl_size numFeatures;
    int cnt = 0;
    float PeakThreshold = 4.0;  ///4.0
    while(cnt < 5){
        vl_covdet_sift(features, numFeatures, fdata, H, W, PeakThreshold);
        if (numFeatures<100)
            PeakThreshold = PeakThreshold*0.2;
        else if (numFeatures>500)
            PeakThreshold = PeakThreshold*1.5;
        else
            break;
        cnt = cnt + 1;
    }
    printf("sift features: PeakThreshold = %.4f\n",PeakThreshold);
    /*L2 norm and rescale to 255*/
    double l2 = 0;
    for(i = 0;i<numFeatures*128;i+=128){
        l2 = 0;
        for(j = 0;j<128;j++)
            l2 += features[i+j]*features[i+j];
        l2 = sqrt(l2);
        for(j = 0;j<128;j++)
            features[i+j] = round(255.0*features[i+j]/l2);
    }
    return numFeatures;
}



int sift_old(unsigned char* img, const int& widthStep, const int& H, const int&W, void* buffer, float* features){
    VlSiftFilt      *filt = 0 ;
    vl_size          q ;
    int              i,j ;
    vl_bool          first ;
    double           angles [4] ;
    vl_sift_pix descr [128] ;
    int l;
    int                   nangles ;
    VlSiftKeypoint const *k ;
    VlSiftKeypoint const *keys = 0 ;
    int                   nkeys ;
    int nframes = 0;
    int                   err ;
    double x,l1;
    /* allocate buffer */
    vl_sift_pix* fdata = (vl_sift_pix*)buffer;
    /* convert data type */
    for(i=0;i<H;i++)
        for(j=0;j<W;j++)
            fdata[i*W+j] = img[i*widthStep+j];
    filt = vl_sift_new (W, H, 4, 3, 0); //-1,3,-1
    if (!filt){
        printf("[Error] vl_sift_new failed.");
        return 0;
    }
    first = 1 ;
    while (1) {
        /* calculate the GSS for the next octave .................... */
        if (first) {
            first = 0 ;
            err = vl_sift_process_first_octave (filt, fdata) ;
        } else {
            err = vl_sift_process_next_octave  (filt) ;
        }
        
        if (err) {
            err = VL_ERR_OK ;
            break ;
        }
        /* run detector ............................................. */
        vl_sift_detect (filt) ;
        keys  = vl_sift_get_keypoints     (filt) ;
        nkeys = vl_sift_get_nkeypoints (filt) ;
        /* for each keypoint ........................................ */
        for (i=0; i < nkeys ; ++i) {
            /* obtain keypoint orientations ........................... */
            k = keys + i ;
            nangles = vl_sift_calc_keypoint_orientations(filt, angles, k) ;
            /* for each orientation ................................... */
            for (q = 0 ; q < nangles ; ++q) {
                vl_sift_calc_keypoint_descriptor(filt, descr, k, angles [q]);
                l1 = 0;
                for (l = 0 ; l < 128 ; ++l) {
                    if(descr[l] > 0.5) descr[l] = 0.5;
                    l1 += descr[l];
                    /*x = 512.0 * descr[l] ;
                     * x = (x < 255.0) ? x : 255.0 ;
                     * features[nframes*128+l] = (vl_uint8) (x);*/
                }
                for (l = 0 ; l < 128 ; ++l)
                    features[nframes*128+l] = descr[l]/l1;
                ++ nframes;
            }
        }
    }
    /*clean up*/
    vl_sift_delete (filt) ;
    return nframes;
}
