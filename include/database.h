#ifndef DATABASE_H
#define	DATABASE_H

#include <tr1/unordered_map>
#include <map>
#include <string>
#include <vector>
#include <flann/flann.hpp>
#include "common.h"

#define MAX_NUM_ITEMS 60000000 //6M


typedef struct{
	float* centres;
	int nb_cents;
	float* queries;
	int nb_samples;
	int* indice;
}BFENC;


template<class T_DATABASE>
class KDTREE{
	flann::SearchParams sp;
	flann::KDTreeIndexParams ip;
	flann::Index<flann::L2<T_DATABASE> >* index;
	int nrow, ncol; //size of the dataset based on which the KD tree was built.
public:
	inline KDTREE():ip(1){ //the number of randomized trees.
		index = NULL;
		/*setting SearchParams*/
		sp.checks = 150;  //specifies the maximum leafs to visit when searching for neighbours
		//eps; Search for eps-approximate neighbors. 
		//sorted; Used only by radius search, specifies if the neighbors returned should be sorted by distance
		//max_neighbors; //Specifies the maximum number of neighbors radiu search should return (default: -1 = unlimited). Only used for radius search.
		sp.cores = 1; //How many cores to assign to the search (specify 0 for automatic core selection).
	}
    inline ~KDTREE()
    {
    	if(index){
		    free(index);
		    index = NULL;
        }
    };
	
	/*data is nrow x ncol matrix*/
	inline void Build(T_DATABASE* data, int nrow, int ncol){
		this->nrow = nrow;
		this->ncol = ncol;
		flann::Matrix<T_DATABASE> dataset (data, nrow, ncol);
		if(index) free(index);
		index = new flann::Index< flann::L2<T_DATABASE> >(dataset, ip);
		index->buildIndex();
	}
	
	/* SEARCH: Perform nearest neighbor search.
	 * keys[In]: nb_key x ncol matrix, each row containing one sample to be queried.
	 * nb_key[In]: the number of keys to be queried.
	 * dist[Out]: nb_key x 1 matrix, with each element represents the distance to nearest neighbor.
	 * indice[Out]: nb_key x 1 matrix, with each element represents the index of the nearest neighbor in the dataset based on which KD tree was built.
	*/
	inline void Search(T_DATABASE* keys, const int& nb_key, float* dist, int* indice){
		flann::Matrix<T_DATABASE> queries (keys, nb_key, this->ncol);
		flann::Matrix<int> indices(indice, nb_key, 1);
		flann::Matrix<float> dists(dist, nb_key, 1);
		index->knnSearch(queries, indices, dists, 1, sp);
	}
};


class DB;

typedef struct{
    int start_row;  //start index of this section in model_matrix
    int end_row;  //end index of this section in model_matrix
    const std::vector<float*>& features;  //features to be encodeded, global
    int row_features;  //total number of feature
    float*  pMax_coss;  //[out] max cosine value of all features in terms of current thread section, size: row_queries
    int* pLine_nums;  //line number in matrix model , max cosine value of all features in terms of current thread section, size: row_queries
    DB* db;
} PARALLEL_ENCODE_PARAM;

class DB{
    int* encoded; //the encoded features [text+image].
    float* count; //The L2 normalized (within product) scores for correlation matching.
    std::vector<int> freq; //frequency of the code [text+image].
    std::vector<std::string> titles;
    std::vector<std::string> urls;
    std::vector<std::string> asins; 
    std::vector<int>img_encoder; // img_encoder[k] is the code of k-th mean vector [in model_matrix].
    float* model_matrix; // The matrix of dimension nb_clustersxdim_feat where nb_clusters is the number of kmeans clusters and dim_feat is 128 (sift dimension).
    int dim_feat; //dimension of sift features [128]]
    long long int* offset; 
    int nb_item;
    int nb_txt_code; //number of text codes.
    int nb_clusters; //number of kmeans clusters (image codes) [approx. 5e4].
    int nb_code; //total number of codes: txt+img.
    std::tr1::unordered_map<std::string,int> text_encoder;
    double* weights;
    KDTREE<float>* kdt;
    float* kdt_dist;
    int* kdt_indice;
	int i,j,k;
	std::map<int, int> Map_encode;
    std::map<int, int>::iterator it;
public:
    DB();
    /*GET_ROW: Get a list of "codes" and the corresponding "counts" of a product.
     * item_id[In]: the product id, begin from 0.
     * code[Out]: the pointer to the beginning address of item codes [SORTED ascendingly].
     * cnt[Out]: the pointer to the beginning address of code counts.
     * Return value: 
     *      successful: the length of 'code' and 'count' [they have the same length].
     *      failed: -1 [you should validate 'item_id' is greater or equal to 0 and smaller than the total number of items in the database].
     */
     
    inline ~DB()
    {
        delete model_matrix;
        model_matrix = NULL;
        delete kdt;
        free(kdt_dist);
        free(kdt_indice);
    };
    
    
    int get_row(const int& item_id, int*& code);
    
    std::string get_title(const int& item_id);
    
    std::string get_asin(const int& item_id);
    
    std::string get_url(const int& item_id);
    
    /*GET_WEIGHT: Get the weights of "codes" (TEXT+IMAGE).
     * weigths[Out]: the weight values.
     */
    void get_weight(double* weights);
    
    /*GET_NB_ITEMS: Get the total number of items int he database.
     * Return value: number of items.
     */
    int get_nb_items();
    
    /*GET_NB_ITEMS: Get the total number of items int he database.
     * Return value: number of codes (TEXT+IMAGE).
     */
    int get_nb_codes();
    
    /*ENCODE: convert text keywords into codes
     * strings_to_be_encoded[In]: the strings to be encoded.
     * codes[Out]: the codes of strings. For strings not found, simply ignore. [SORTED ascendingly].
     * counts[Out]: the L2 normalized frequence of codes.
     * max_code_len[In]: the size of 'codes' buffer.
     * actual_code_len [Out]: number of the 'codes'.
     * Return value: 0 if success. -1 if buffer not enough, -2 if no matching keyword found.
     */
    int encode(const std::vector<std::string>& strings_to_be_encoded, int* codes, float* counts, const int& max_code_len, int& actual_code_len);

    
    /*ENCODE: convert sift features into codes using brute force
     * sift_features_to_be_encoded[In]: the sift feature to be encoded. nb_featsx128 matrix. Each row is one sift feature.
     * nb_feats[In]: number of sift features in the 'sift_features_to_be_encoded'.
     * codes[Out]: the codes of strings. For strings not found, simply ignore. [SORTED ascendingly].
     * counts[Out]: the L2 normalized frequence of codes.
     * max_code_len[In]: the size of 'codes' buffer.
     * actual_code_len [Out]: number of the 'codes'.
     * Return value: 0 if success. -1 if buffer not enough, -2 if no matching keyword found.
     */
    int encode_bf(float* sift_features_to_be_encoded, const int& nb_feats, int* codes, float* counts, const int& max_code_len, int& actual_code_len);
    
    
    /*ENCODE: convert sift features into codes using KD tree
     * sift_features_to_be_encoded[In]: the sift feature to be encoded. nb_featsx128 matrix. Each row is one sift feature.
     * nb_feats[In]: number of sift features in the 'sift_features_to_be_encoded'.
     * codes[Out]: the codes of strings. For strings not found, simply ignore. [SORTED ascendingly].
     * counts[Out]: the L2 normalized frequence of codes.
     * max_code_len[In]: the size of 'codes' buffer.
     * actual_code_len [Out]: number of the 'codes'.
     * Return value: 0 if success. -1 if buffer not enough, -2 if no matching keyword found.
     */
    int encode_kdt(float* sift_features_to_be_encoded, const int& nb_feats, int* codes, float* counts, const int& max_code_len, int& actual_code_len);

    //return private member attribute
    inline int get_dim_feat()
    {
        return dim_feat;
    }

    inline int get_nb_clusters()
    {
        return nb_clusters;
    }

    inline float* get_model_matrix()
    {
        return model_matrix;
    }

};


#endif	/* DATABASE_H */

