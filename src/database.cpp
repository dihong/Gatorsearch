#include "database.h"
#include "stdio.h"
#include "stdlib.h"
#include <unistd.h>
#include <memory.h>
#include "main.h"

extern int RANK;

using namespace std::tr1;

DB::DB() {
    
    /*Initialize the database*/
    tic();
    char buffer[10000];
    int pbuf;
    int tmp,line_num,ret;
    double sum;
    dim_feat = DIM_FEAT;
    FILE* fp = 0;
    
    /*MASTER: TITLE. URL. ASIN. DICT.*/
    if(RANK==MASTER_RANK){
        sleep(30);
        //---------TITLE.txt--------
        fp = fopen(ROOT_DIR_MASTER"/data/TITLE.txt", "r");
        if (!fp) {
            printf("Failed to open %s\n", ROOT_DIR_MASTER"/data/TITLE.txt");
            exit(-1);
        }
        while (!feof(fp)) {
            if (fgets(buffer, sizeof (buffer), fp) != NULL)
                titles.push_back(buffer);
            else
                break;
        }
        fclose(fp);
        nb_item = titles.size();
        
        //---------URL.txt--------
        fp = fopen(ROOT_DIR_MASTER"/data/URL.txt", "r");
        if (!fp) {
            printf("Failed to open %s\n", ROOT_DIR_MASTER"/data/URL.txt");
            exit(-1);
        }
        while (!feof(fp)) {
            if (fgets(buffer, sizeof (buffer), fp) != NULL)
                urls.push_back(buffer);
            else
                break;
        }
        fclose(fp);
        if (urls.size() != nb_item) {
            puts("Error loading URL.txt: file is corrupted");
            exit(-1);
        }
        
        //-------ASIN.txt-----------
        fp = fopen(ROOT_DIR_MASTER"/data/ASIN.txt", "r");
        if (!fp) {
            printf("Failed to open %s\n", ROOT_DIR_MASTER"/data/ASIN.txt");
            exit(-1);
        }
        for (i = 0; i < nb_item; i++) {
            if (fgets(buffer, 100, fp) != NULL){ //read 10 characters only.
                buffer[10] = 0;
                asins.push_back(buffer);
            }
            else {
                puts("Error loading ASIN.txt: file is corrupted");
                exit(-1);
            }
        }
        fclose(fp);
        
        //-------TXT_DICT.txt-------
        printf("Loading text dictionary ... "); fflush(stdout); tic();
        fp = fopen(ROOT_DIR_MASTER"/model/TXT_DICT.txt", "r");
        if (!fp) {
            printf("Failed to open %s\n", ROOT_DIR_MASTER"/model/TXT_DICT.txt");
            exit(-1);
        }
        line_num = 0;
        while (!feof(fp)) {
            pbuf = 0;
            if (fgets(buffer, sizeof (buffer), fp) != NULL) {
                while (buffer[pbuf] != ' ' && pbuf<sizeof (buffer)) pbuf++;
                if (pbuf == sizeof (buffer)) {
                    puts("Error loading TXT_DICT.txt: file is corrupted");
                    exit(-1);
                }
                buffer[pbuf++] = 0;
                sscanf(buffer + pbuf, "%d", &tmp);
                if(tmp<=0){
                    printf("Error: text freq value must be positive.\n");
                    exit(-1);
                }
                freq.push_back(tmp);  //frequency
                while (buffer[pbuf] != ' ' && pbuf<sizeof (buffer)) pbuf++; //search for second ' '.
                sscanf(buffer + pbuf, "%d", &tmp);
                if(tmp<0){
                    printf("Error: text code value must be nonnegative.\n");
                    exit(-1);
                }
                text_encoder[buffer] = tmp; //txt code: beginning at zero.
                line_num++;
            } else
                break;
        }
        fclose(fp);
        nb_txt_code = line_num;
        printf("done. Elapsed time is %.2f second. Number of codes in text dictionary is: %d\n",toc(),nb_txt_code);
        //-------IMG_DICT.txt-------
        printf("Loading image dictionary ... "); fflush(stdout); tic();
        fp = fopen(ROOT_DIR_MASTER"/model/IMG_DICT.txt", "r");
        if (!fp) {
            printf("Failed to open %s\n", ROOT_DIR_MASTER"/model/IMG_DICT.txt");
            exit(-1);
        }
        nb_clusters = 0;
        while (!feof(fp) && fgets(buffer, sizeof (buffer), fp) != NULL) nb_clusters++;
        model_matrix = new float [nb_clusters * DIM_FEAT];
        rewind(fp);
        for (line_num = 0; line_num < nb_clusters; line_num++) {
            for (i = 0; i < DIM_FEAT; i++) {
                fscanf(fp, "%f", model_matrix + line_num * DIM_FEAT + i);
            }
            fscanf(fp, "%d", &tmp); //count
            if(tmp<=0){
                printf("Error: image freq value must be positive.\n");
                exit(-1);
            }
            freq.push_back(tmp);
            fscanf(fp, "%d", &tmp); //code, beginning at 0.
            if(line_num==0 && tmp!=0){
                printf("Error: image code must begin at zero. But it begins at %d.\n",tmp);
                exit(-1);
            }
            if(tmp<0){
                printf("Error: image code value must be nonnegative.\n");
                exit(-1);
            }
            img_encoder.push_back(tmp + nb_txt_code);
        }
        fclose(fp);
        printf("done. Elapsed time is %.2f second. Number of codes in image dictionary is: %d\n",toc(),nb_clusters);
        nb_code = nb_txt_code + nb_clusters;
        //------------KD tree----------------
        printf("Building KD tree ... "); fflush(stdout); tic();
        kdt = new KDTREE<float>;
        kdt_dist = (float*) malloc(sizeof(float)*MAX_NUM_CODE_PER_QUERY);
        kdt_indice = (int*) malloc(sizeof(int)*MAX_NUM_CODE_PER_QUERY);
        kdt->Build(model_matrix,nb_clusters, DIM_FEAT);
        printf("done. Elapsed time is %.2f second.\n",toc());
    }
    
    /*SLAVE: OFFSET. ENCODE. COUNT.*/
    else{
        //-------TXT_DICT.txt-------
        fp = fopen(ROOT_DIR_MASTER"/model/TXT_DICT.txt", "r");
        if (!fp) {
            printf("Failed to open %s\n", ROOT_DIR_MASTER"/model/TXT_DICT.txt");
            exit(-1);
        }
        line_num = 0;
        while (!feof(fp)) {
            if (fgets(buffer, sizeof (buffer), fp) != NULL) {
                line_num++;
            } else
                break;
        }
        fclose(fp);
        nb_txt_code = line_num;
        //-------IMG_DICT.txt-------
        fp = fopen(ROOT_DIR_MASTER"/model/IMG_DICT.txt", "r");
        if (!fp) {
            printf("Failed to open %s\n", ROOT_DIR_MASTER"/model/IMG_DICT.txt");
            exit(-1);
        }
        nb_clusters = 0;
        while (!feof(fp) && fgets(buffer, sizeof (buffer), fp) != NULL) nb_clusters++;
        fclose(fp);
        nb_code = nb_txt_code + nb_clusters;
        //-----------weights-----------
        weights = new double [nb_code];
        for (i = 0; i < nb_code; i++){ //debug.
            if(i<nb_txt_code)
                weights[i] = 1.0;//log(1+freq[i]);
            else
                weights[i] = 0.1;//log(1+freq[i]);
        }
        //-------TXT_OFFSET.int64-------
        if(RANK==1) printf("Loading TXT_OFFSET ... "); fflush(stdout); tic();
        fp = fopen(ROOT_DIR"/model/TXT_OFFSET.int64", "rb");
        if (!fp) {
            printf("[%d] Failed to open %s\n",RANK, ROOT_DIR"/model/TXT_OFFSET.int64");
            exit(-1);
        }
        long long int* offset_txt = new long long int [MAX_NUM_ITEMS];
        nb_item = fread(offset_txt, sizeof (long long int), MAX_NUM_ITEMS, fp) - 1;
        fclose(fp);
        if(nb_item==0){
            printf("[%d] Read empty TXT_OFFSET.int64\n",RANK);
            exit(-1);
        }else if(nb_item+1==MAX_NUM_ITEMS){
            printf("[%d] MAX_NUM_ITEMS is not large enough.\n",RANK);
            exit(-1);
        }
        if(RANK==1) printf("done. Elapsed time is %.2f second. Number of items is: %d\n",toc(),nb_item);
        //-------IMG_OFFSET.int64-------
        if(RANK==1) printf("Loading IMG_OFFSET ... "); fflush(stdout); tic();
        fp = fopen(ROOT_DIR"/model/IMG_OFFSET.int64", "rb");
        if (!fp) {
            printf("Failed to open %s\n", ROOT_DIR"/model/IMG_OFFSET.int64");
            exit(-1);
        }
        long long int* offset_img = new long long int [nb_item+1];
        if(fread(offset_img, sizeof (long long int), nb_item+1, fp)!=nb_item+1){
            printf("[%d] Cannot read IMG_OFFSET.int64 successfully.\n", RANK);
            exit(-1);
        }
        fclose(fp);
        if(RANK==1) printf("done. Elapsed time is %.2f second. Length of encoded image is: %d\n",toc(),offset_img[nb_item]);
        //-------TXT_ENCODE_NDUP.int32-------
        if(RANK==1) printf("Loading TXT_ENCODE_NDUP ... "); fflush(stdout); tic();
        fp = fopen(ROOT_DIR"/model/TXT_ENCODE_NDUP.int32", "rb");
        if (!fp) {
            printf("[%d] Failed to open %s\n", RANK, ROOT_DIR"/model/TXT_ENCODE_NDUP.int32");
            exit(-1);
        }
        int* encoded_txt = new int [offset_txt[nb_item]];
        ret = fread(encoded_txt, sizeof (int), offset_txt[nb_item], fp);
        if (ret != offset_txt[nb_item]) {
            printf("Error reading TXT_ENCODE_NDUP.int32: file corrupted. %d bytes were read, but expecting %d bytes.", ret, offset_txt[nb_item]);
            exit(-1);
        }
        fclose(fp);
        if(RANK==1) printf("done. Elapsed time is %.2f second. Length of encoded text is: %d\n",toc(),offset_txt[nb_item]);
        //-------IMG_ENCODE_NDUP.int32-------
        if(RANK==1) printf("Loading IMG_ENCODE_NDUP ... "); fflush(stdout); tic();
        fp = fopen(ROOT_DIR"/model/IMG_ENCODE_NDUP.int32", "rb");
        if (!fp) {
            printf("Failed to open %s\n", ROOT_DIR"/model/IMG_ENCODE_NDUP.int32");
            exit(-1);
        }
        int* encoded_img = new int [offset_img[nb_item]];
        if (fread(encoded_img, sizeof (int), offset_img[nb_item], fp) != offset_img[nb_item]) {
            printf("Error reading IMG_ENCODE_NDUP.int32: file corrupted.");
            exit(-1);
        }
        
        fclose(fp);
        for(i = 0;i<offset_img[nb_item];i++)  //convert local encoding to global encoding
            encoded_img[i] += nb_txt_code;
        if(RANK==1) printf("done. Elapsed time is %.2f second. Length of encoded image is: %d\n",toc(),offset_img[nb_item]);
        //-------TXT_COUNT_NDUP.single-------
        /*if(RANK==1) printf("Loading TXT_COUNT_NDUP ... "); fflush(stdout); tic();
        fp = fopen(ROOT_DIR"/model/TXT_COUNT_NDUP.single", "rb");
        if (!fp) {
            printf("Failed to open %s\n", ROOT_DIR"/model/TXT_COUNT_NDUP.single");
            exit(-1);
        }
        float* count_txt = new float [offset_txt[nb_item]];
        if (fread(count_txt, sizeof (float), offset_txt[nb_item], fp) != offset_txt[nb_item]) {
            printf("Error reading TXT_COUNT_NDUP.single: file corrupted.");
            exit(-1);
        }
        fclose(fp);
        if(RANK==1) printf("done. Elapsed time is %.2f second. \n",toc());*/
        //-------IMG_COUNT_NDUP.single-------
        /*if(RANK==1) printf("Loading IMG_COUNT_NDUP ... "); fflush(stdout); tic();
        if(RANK==MASTER_RANK)
            fp = fopen(ROOT_DIR_MASTER"/model/IMG_COUNT_NDUP.single", "rb");
        else
            fp = fopen(ROOT_DIR"/model/IMG_COUNT_NDUP.single", "rb");
        
        if (!fp) {
            printf("Failed to open %s\n", ROOT_DIR"/model/IMG_COUNT_NDUP.single");
            exit(-1);
        }
        float* count_img = new float [offset_img[nb_item]];
        if (fread(count_img, sizeof (float), offset_img[nb_item], fp) != offset_img[nb_item]) {
            printf("Error reading IMG_COUNT_NDUP.single: file corrupted.");
            exit(-1);
        }
        fclose(fp);
        if(RANK==1) printf("done. Elapsed time is %.2f second. \n",toc());*/
        //-------------COMBINE TXT & IMG [count]----------------
        /*if(RANK==1) printf("Combining TXT & IMG ... "); fflush(stdout); tic();
        offset = new long long int [nb_item+1];
        encoded = new int [offset_txt[nb_item]+offset_img[nb_item]];
        count = new float [offset_txt[nb_item]+offset_img[nb_item]];
        offset[0] = 0;
        for(i = 0;i<nb_item;i++){ //for each item.
            memcpy(encoded+offset[i],encoded_txt+offset_txt[i],sizeof(int)*(offset_txt[i+1]-offset_txt[i]));
            memcpy(encoded+offset[i]+offset_txt[i+1]-offset_txt[i],encoded_img+offset_img[i],sizeof(int)*(offset_img[i+1]-offset_img[i]));
            memcpy(count+offset[i],count_txt+offset_txt[i],sizeof(float)*(offset_txt[i+1]-offset_txt[i]));
            memcpy(count+offset[i]+offset_txt[i+1]-offset_txt[i],count_img+offset_img[i],sizeof(float)*(offset_img[i+1]-offset_img[i]));
            offset[i+1] = offset[i] + (offset_txt[i+1]-offset_txt[i]) + (offset_img[i+1]-offset_img[i]);
        }
        if(RANK==1) printf("done. Elapsed time is %.2f second. \n",toc());
        delete offset_txt, encoded_txt, count_txt, offset_img, encoded_img, count_img;*/
        
        //-------------COMBINE TXT & IMG [NO count]----------------
        if(RANK==1) printf("Combining TXT & IMG ... "); fflush(stdout); tic();
        offset = new long long int [nb_item+1];
        encoded = new int [offset_txt[nb_item]+offset_img[nb_item]];
        offset[0] = 0;
        for(i = 0;i<nb_item;i++){ //for each item.
            memcpy(encoded+offset[i],encoded_txt+offset_txt[i],sizeof(int)*(offset_txt[i+1]-offset_txt[i]));
            memcpy(encoded+offset[i]+offset_txt[i+1]-offset_txt[i],encoded_img+offset_img[i],sizeof(int)*(offset_img[i+1]-offset_img[i]));
            offset[i+1] = offset[i] + (offset_txt[i+1]-offset_txt[i]) + (offset_img[i+1]-offset_img[i]);
        }
        if(RANK==1) printf("done. Elapsed time is %.2f second. \n",toc());
        delete offset_txt, encoded_txt, offset_img, encoded_img;
    }
    //--------------------------------------------------------------------------------------------------------------------------------
    printf("*[%d] Loading database done*\n", RANK);
}

int DB::get_row(const int& item_id, int*& code) {
    //if (item_id >= nb_item || item_id < 0) return -1;
    code = encoded + offset[item_id];
    //cnt = count + offset[item_id];
    return offset[item_id + 1] - offset[item_id];
}

string DB::get_asin(const int& item_id) {
    return asins[item_id];
}

string DB::get_url(const int& item_id) {
    return urls[item_id];
}

string DB::get_title(const int& item_id) {
    if (item_id >= nb_item || item_id < 0) {
        printf("[Error] DB::get_title: index %d was out of range.\n", item_id);
        exit(-1);
    }
    return titles[item_id];
}

void DB::get_weight(double* weights) {
    memcpy(weights,this->weights,sizeof(double)*nb_code);
    for (int i = 0; i < nb_code; i++){ //debug.
     if(i<nb_txt_code)
      weights[i] = 1.0;//log(1+freq[i]);
      else
      weights[i] = 0.01;//log(1+freq[i]);
    }
}

int DB::get_nb_items() {
    return nb_item;
}

int DB::get_nb_codes() {
    return nb_code;
}

//==================================================================================================================================================

/*ENCODE: convert text keywords into codes
 * strings_to_be_encoded[In]: the strings to be encoded.
 * codes[Out]: the codes of strings. For strings not found, simply ignore. [SORTED ascendingly].
 * counts[Out]: the L2 normalized frequence of codes.
 * max_code_len[In]: the size of 'codes' buffer.
 * actual_code_len [Out]: number of the 'codes'.
 * Return value: 0 if success. -1 if buffer not enough, -2 if no matching keyword found.
 */
int DB::encode(const std::vector<std::string>& strings_to_be_encoded, int* codes, float* counts, const int& max_code_len, int& actual_code_len) {
    actual_code_len = 0;
    std::tr1::unordered_map<std::string, int>::iterator it;
    map<int, int>::iterator it2;
    map<int, int> mapper;
    for (i = 0; i < strings_to_be_encoded.size(); i++) {
        it = text_encoder.find(strings_to_be_encoded[i]);
        if (it != text_encoder.end()) { //the code is in DB
            it2 = mapper.find(it->second);
            if (it2 == mapper.end()) { //not seen.
                mapper[it->second] = 1;
            } else {
                it2->second++; //scanned.
            }
        }
    }
    if (mapper.size() > max_code_len)
        return -1;
    float l2 = 0;
    for (it2 = mapper.begin(); it2 != mapper.end(); it2++) {
        codes [actual_code_len] = it2->first;
        counts[actual_code_len] = it2->second;
        actual_code_len++;
        l2 += counts[actual_code_len]*counts[actual_code_len];
    }
    l2 = sqrt(l2);
    
    if(l2>0)
        for (i = 0;i<actual_code_len;i++)
            counts[i] /= l2;
    
    if (actual_code_len == 0)
        return -2;
    else
        return 0;
}

void bfenc(void* par){
	BFENC* param = (BFENC*) par;
	int i,j,k;
	float max;  ///max correlation [L2 norm].
	float tmp1; 
	float* q, *c;
	float* queries = param->queries;
	float* centres = param->centres;
	int nb_cents = param->nb_cents;
	int nb_samples = param->nb_samples;
	int* indice = param->indice;
	for(i=0;i<param->nb_samples;i++){
		max = 0;
		q = queries + i*128;
		for(k=0;k<128;k++)
			max += centres[k]*q[k];
		indice[i] = 0;
		for(j=1;j<nb_cents;j++){
			tmp1 = 0;
			c = centres + j*128;
			for(k=0;k<128 && tmp1<max;k++)
				tmp1 += c[k]*q[k];
			if(tmp1>=max){
				max = tmp1;
				indice[i] = j;
			}
		}
	}
}

/*ENCODE: convert sift features into codes using brute force
 * sift_features_to_be_encoded[In]: the sift feature to be encoded. nb_featsx128 matrix. Each row is one sift feature.
 * nb_feats[In]: number of sift features in the 'sift_features_to_be_encoded'.
 * codes[Out]: the codes of strings. For strings not found, simply ignore. [SORTED ascendingly].
 * counts[Out]: the L2 normalized frequence of codes.
 * max_code_len[In]: the size of 'codes' buffer.
 * actual_code_len [Out]: number of the 'codes'.
 * Return value: 0 if success. -1 if buffer not enough, -2 if no matching keyword found.
 */
int DB::encode_bf(float* sift_features_to_be_encoded, const int& nb_feats, int* codes, float* counts, const int& max_code_len, int& actual_code_len){
    Parallel PL;
    vector<void*> thread_params;
    int nthreads = 12;
    int nb_item_per_thread = floor((double)nb_feats/nthreads);
    BFENC* params = new BFENC [nthreads];
    
    /*FILE* fp = fopen("feat.txt","w+");
    for(i=0;i<nb_feats;i++){
    	for(j=0;j<dim_feat;j++){
    		fprintf(fp,"%.4f ",sift_features_to_be_encoded[i*dim_feat+j]);
    	}
    	fprintf(fp,"\n");
    }
    fclose(fp);*/

    for (k = 0; k < nthreads; k++){
        params[k].centres = model_matrix;
        params[k].nb_cents = nb_clusters;
        params[k].queries = sift_features_to_be_encoded + k*dim_feat*nb_item_per_thread;
        params[k].indice = kdt_indice + k*nb_item_per_thread;
        params[k].nb_samples = nb_item_per_thread;
        if(k==nthreads-1)
            params[k].nb_samples = nb_feats-(nthreads-1)*nb_item_per_thread;
        thread_params.push_back(params + k);
    }
    
    PL.Run(bfenc, thread_params);
    delete params;
    /*histogram & normalization*/
    Map_encode.clear();
    for(i = 0;i<nb_feats;i++){
        codes[i] = img_encoder[kdt_indice[i]];
        if(Map_encode.find(codes[i])==Map_encode.end())
            Map_encode[codes[i]] = 1;
        else
            Map_encode[codes[i]]++;
    }
    i = 0; j = 0;
    for(it = Map_encode.begin();it!=Map_encode.end();it++){
        codes[i] = it->first;
        counts[i] = it->second;
        j += it->second*it->second; //total count.
        i++;
    }
    actual_code_len = i;
    double l2 = sqrt(j);
    for(k=0;k<actual_code_len;k++) //normalize counts into L2 norm.
        counts[k] /= l2;
    
    return 0;
}

/*ENCODE: convert sift features into codes using KD tree
 * sift_features_to_be_encoded[In]: the sift feature to be encoded. nb_featsx128 matrix. Each row is one sift feature.
 * nb_feats[In]: number of sift features in the 'sift_features_to_be_encoded'.
 * codes[Out]: the codes of strings. For strings not found, simply ignore. [SORTED ascendingly].
 * counts[Out]: the L2 normalized frequence of codes.
 * max_code_len[In]: the size of 'codes' buffer.
 * actual_code_len [Out]: number of the 'codes'.
 * Return value: 0 if success. -1 if buffer not enough, -2 if no matching keyword found.
 */
int DB::encode_kdt(float* sift_features_to_be_encoded, const int& nb_feats, int* codes, float* counts, const int& max_code_len, int& actual_code_len){
    if(nb_feats>MAX_NUM_CODE_PER_QUERY || max_code_len>MAX_NUM_CODE_PER_QUERY)
        return -1; //buffer not enough.
    kdt->Search(sift_features_to_be_encoded, nb_feats, kdt_dist, kdt_indice);
    Map_encode.clear();
    for(i = 0;i<nb_feats;i++){
        codes[i] = img_encoder[kdt_indice[i]];
        if(Map_encode.find(codes[i])==Map_encode.end())
            Map_encode[codes[i]] = 1;
        else
            Map_encode[codes[i]]++;
    }
    i = 0; j = 0;
    for(it = Map_encode.begin();it!=Map_encode.end();it++){
        codes[i] = it->first;
        counts[i] = it->second;
        j += it->second*it->second; //total count.
        i++;
    }
    actual_code_len = i;
    double l2 = sqrt(j);
    for(k=0;k<actual_code_len;k++) //normalize counts into L2 norm.
        counts[k] /= l2;
    return 0;
}
