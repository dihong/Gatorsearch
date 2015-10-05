#include "common.h"
#include "main.h"
#include "libstemmer.h"
#include <mpi/mpi.h>
#include <algorithm>
#include "search.h"
#include "sift_feature.h"
#import <fcntl.h>

using namespace std;
using namespace cv;

/*GLOBAL VARIABLES*/
queue<QUERY_QUEUE_STRUCT>Incomming_Query_Queue;
set<RESPONSE_ARR_STRUCT>Response_Array;
pthread_mutex_t mutex_incoming_queue = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex_response_array = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex_machine_status = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t lock_thread_net = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t lock_thread_tcp = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t lock_database = PTHREAD_MUTEX_INITIALIZER;
MACHINE_STATUS machines[N_MACHINE];
int RANK; //my rank.
pthread_t Tid_Thread_Accept_Query;
pthread_t Tid_Thread_Wait_Job_From_Master;
struct sb_stemmer * stemmer = 0; //stemming
DB* database;

/****************************************************************************************************/

/*Threads*/
void* Thread_Accept_Query(void* params) {
    /*This thread accepts queries from php via TCP, and insert the queries into incoming queue.*/
    int sockfd, newsockfd, portno;
    char* buffer = new char [MAX_MPI_MSG_SIZE]; //200x200 image.
    float* features = new float [DIM_FEAT*MAX_NUM_CODE_PER_QUERY];  //sift features.
    int* code_buf = new int [MAX_NUM_CODE_PER_QUERY];
    int* count_buf = new int [MAX_NUM_CODE_PER_QUERY];
    int ncode_img = 0;
    struct sockaddr_in serv_addr, cli_addr;
    int n, sz_text, sz_image,i,j,ret,nkeypnts;
    /* First call to socket() function */
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("[Error] Thread_Accept_Query: opening socket");
        exit(1);
    }
    /* Initialize socket structure */
    bzero((char *) &serv_addr, sizeof (serv_addr));
    portno = 59886; //59886;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(portno);

    /* Now bind the host address using bind() call.*/
    int option = 1;
    if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof (serv_addr)) < 0) {
        MPI_Finalize();
        perror("[Error] Thread_Accept_Query: binding");
        exit(1);
    }
    if (setsockopt(sockfd, SOL_SOCKET, (SO_REUSEPORT | SO_REUSEADDR), (char*) &option, sizeof (option)) < 0) {
        printf("setsockopt failed\n");
        close(sockfd);
        MPI_Finalize();
        exit(1);
    }
    listen(sockfd, 100); //at most 100 connections in the waiting queue.
    socklen_t clilen = sizeof (cli_addr);
    printf("[%d] Thread_Accept_Query starts up. Now waiting for connections...\n", RANK);
    QUERY_QUEUE_STRUCT str;
	if (fcntl(sockfd, F_SETFL, fcntl(sockfd, F_GETFL, 0) | O_NONBLOCK) == -1){
	    perror("calling fcntl");
	   // handle the error.  By the way, I've never seen fcntl fail in this way
	}
    while (1) {
        /* Accept actual connection from the client */
        pthread_mutex_lock(&lock_thread_tcp);
        newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen); //nonblock
        pthread_mutex_unlock(&lock_thread_tcp);
        if(newsockfd<0){
        	usleep(PROBE_SLEEP);
        	continue;
        }
        /*If there is not enough buffer in the queue, reject it*/
        pthread_mutex_lock(&mutex_incoming_queue);
        if (Incomming_Query_Queue.size() >= MAX_PENDING_QUERY) {
            pthread_mutex_unlock(&mutex_incoming_queue);
            close(newsockfd);
            continue;
        }
        pthread_mutex_unlock(&mutex_incoming_queue);
        /* If connection is established then start communicating */
        bzero(buffer, MAX_MPI_MSG_SIZE);
        pthread_mutex_lock(&lock_thread_tcp);
        n = recv(newsockfd, buffer, MAX_MPI_MSG_SIZE, 0);
        pthread_mutex_unlock(&lock_thread_tcp);
        if (n < 0) {
            perror("[Error] Thread_Accept_Query: reading from socket");
            continue;
        } else if (n >= MAX_MPI_MSG_SIZE - 1) {
            printf("Image size is too large.\n");
            continue;
        }
        n -= 2;
        sz_text = strlen(buffer);
        sz_image = n - sz_text - 1;
        //printf("text:%d, image:%d\n",sz_text,sz_image);
        str.sockfd = newsockfd;
        str.qpkt.id = get_unique_query_id();
        str.qpkt.n_code = 0;
        if (sz_text > 0){  //text encoding.
        	tic();
		    ret = database->encode(stem(buffer), str.qpkt.code, str.qpkt.count, MAX_NUM_CODE_PER_QUERY, str.qpkt.n_code);
		    printf("Text stemming+encoding time: %.4f. Number of codes is: %d\n",toc(),str.qpkt.n_code);
		    if (ret < 0) {
		        if (ret == -1)
		            puts("Error encoding the text keywords. Error code = -1");
		        if (ret == -2)
		            puts("Error encoding the text keywords. Error code = -2");
		        string tmp = "No item found.";
		        send(newsockfd, &tmp[0], tmp.size(), 0);
		        close(newsockfd);
		        continue;
		    }
        }
        if (sz_image > 0){ //image encoding.
            CvMat A = Mat(1, n, CV_8UC1, buffer + sz_text + 1); //initialize image object.
            IplImage* im_gray = cvDecodeImage(&A, CV_LOAD_IMAGE_GRAYSCALE); //load as grayscale image. 
            //==============
            /*printf("Image: height = %d, width = %d\n", im_gray->height, im_gray->width);
            FILE* fp = fopen("opencv_gray.txt","w+");
            for(int i = 0;i<im_gray->height;i++){
            	for(int j = 0;j<im_gray->width;j++){
            		fprintf(fp,"%d ",(unsigned char)im_gray->imageData[i*im_gray->widthStep+j]);
            	}
            	fprintf(fp,"\n");
            }
            fclose(fp);*/
            //==============
            DESC* sift2 = new DESC (0);
            tic();
            if(im_gray->height<20 || im_gray->width<20){
            	printf("Bad Image.");
            	continue;
            }
            nkeypnts = sift2->extract_feat((unsigned char*)im_gray->imageData, im_gray->widthStep, im_gray->height, im_gray->width, features, MAX_NUM_CODE_PER_QUERY);
            //nkeypnts = sift((unsigned char*)im_gray->imageData, im_gray->widthStep, im_gray->height, im_gray->width, buffer, features);
            printf("Feature extraction time = %.5f, nkeypnts = %d\n",toc(),nkeypnts);
            if(nkeypnts<0) continue;
            delete sift2;
            tic();
            //ret = database->encode_bf(features, nkeypnts, str.qpkt.code+str.qpkt.n_code, str.qpkt.count+str.qpkt.n_code, MAX_NUM_CODE_PER_QUERY-str.qpkt.n_code, ncode_img);
            ret = database->encode_kdt(features, nkeypnts, str.qpkt.code+str.qpkt.n_code, str.qpkt.count+str.qpkt.n_code, MAX_NUM_CODE_PER_QUERY-str.qpkt.n_code, ncode_img);
		    if (ret < 0) {
		        if (ret == -1)
		            puts("Error encoding the image keywords. Error code = -1");
		        if (ret == -2)
		            puts("Error encoding the image keywords. Error code = -2");
		        continue;
		    }
            printf("Image encoding time: %.4f. Number of codes is: %d\n",toc(),ncode_img);
            cvReleaseImage(&im_gray);
            str.qpkt.n_code += ncode_img;
            //cvShowImage("Win", im_gray);
            //cvWaitKey(0);
        }
        str.ts_enqueue = tic();
        pthread_mutex_lock(&mutex_incoming_queue);
        Incomming_Query_Queue.push(str);
        pthread_mutex_unlock(&mutex_incoming_queue);
    }
    return 0;
}

void* Thread_Scheduler(void* params) {
    int sort_by_pending_jobs_val[N_MACHINE];
    int sort_by_pending_jobs_ind[N_MACHINE];
    const int L = N_MACHINE * NUM_ITEMS_NEEDED * MAX_THREAD_PER_MACHINE; //number of buffer required by one job.
    float* buffer_distance_of_responses = new float [N_MACHINE * MAX_PENDING_JOB_SLAVE * L];
    int* buffer_item_id_of_responses = new int [N_MACHINE * MAX_PENDING_JOB_SLAVE * L];
    QUERY_QUEUE_STRUCT query;
    RESPONSE_ARR_STRUCT response;
    MACHINE_STATUS status_snapshot[N_MACHINE];
    int i, j, cnt, pkt_size, max_pending_jobs, best_nb_machines;
    float best_assignment_cost, new_cost;
    char* buffer = new char [MAX_MPI_MSG_SIZE]; //used to send query to worker machines.
    cnt = 0;
    
    printf("[%d] Thread_Scheduler starts up.\n",RANK);
    while (1) {
        pthread_mutex_lock(&mutex_incoming_queue);
        if (Incomming_Query_Queue.size() == 0) { //no query currently.
            pthread_mutex_unlock(&mutex_incoming_queue);
            usleep(PROBE_SLEEP);
            continue;
        }
        pthread_mutex_unlock(&mutex_incoming_queue);
        /*Determine whether and how to take care of the next query*/
        pthread_mutex_lock(&mutex_machine_status);
        memcpy(status_snapshot, machines, sizeof (MACHINE_STATUS) * N_MACHINE);
        pthread_mutex_unlock(&mutex_machine_status);
        quick_sort(sort_by_pending_jobs_val, N_MACHINE, sort_by_pending_jobs_ind, sort_by_pending_jobs_val, false); //sort ascending.
        best_assignment_cost = (1.0 + status_snapshot[sort_by_pending_jobs_ind[0]].n_pending_jobs) / (0.01 + status_snapshot[sort_by_pending_jobs_ind[0]].jobs_per_second);
        best_nb_machines = 1;
        for (i = 2; i <= N_MACHINE; i++) {
            new_cost = (1.0 / i + status_snapshot[sort_by_pending_jobs_ind[0]].n_pending_jobs) / (0.01 + status_snapshot[sort_by_pending_jobs_ind[0]].jobs_per_second);
            for (j = 1; j < i; j++)
                if ((1 / i + status_snapshot[sort_by_pending_jobs_ind[j]].n_pending_jobs) / (0.01 + status_snapshot[sort_by_pending_jobs_ind[j]].jobs_per_second) > new_cost)
                    new_cost = (1.0 / i + status_snapshot[sort_by_pending_jobs_ind[j]].n_pending_jobs) / (0.01 + status_snapshot[sort_by_pending_jobs_ind[j]].jobs_per_second);
            if (new_cost < best_assignment_cost) {
                best_assignment_cost = new_cost;
                best_nb_machines = i;
            }
        }
        max_pending_jobs = sort_by_pending_jobs_val[best_nb_machines - 1];
        while (1) { /*to prevent overflow the slave machine*/
            if (max_pending_jobs < MAX_PENDING_JOB_SLAVE) {
                break;
            } else {
                usleep(10000);
                pthread_mutex_lock(&mutex_machine_status);
                memcpy(status_snapshot, machines, sizeof (MACHINE_STATUS) * N_MACHINE);
                pthread_mutex_unlock(&mutex_machine_status);
                max_pending_jobs = 0;
                for (i = 0; i < best_nb_machines; i++) {
                    if (max_pending_jobs < status_snapshot[sort_by_pending_jobs_ind[i]].n_pending_jobs)
                        max_pending_jobs = status_snapshot[sort_by_pending_jobs_ind[i]].n_pending_jobs;
                }
            }
        }
        /*assign the next query to worker machines*/
        pthread_mutex_lock(&mutex_incoming_queue);
        query = Incomming_Query_Queue.front(); //take the first element in the queue.
        Incomming_Query_Queue.pop(); //remove the first element.
        pthread_mutex_unlock(&mutex_incoming_queue);
        response.id = query.qpkt.id;
        response.num_machines = best_nb_machines;
        response.ts_enqueue = query.ts_enqueue;
        response.merged_matching_distances = buffer_distance_of_responses + cnt*L;
        response.merged_item_ids = buffer_item_id_of_responses + cnt*L;
        response.sockfd = query.sockfd;
        //response.num_results = 0;
        cnt = (cnt + 1) % (N_MACHINE * MAX_PENDING_JOB_SLAVE);
        query.qpkt.n_workers = best_nb_machines;
        //printf("Query id = %d has been assigned to machine(s):", query.qpkt.id);
        response.ts_schuduled = tic();
        for (i = 0; i < best_nb_machines; i++) {
            query.qpkt.part = i;
            if (!Encode_MPI_Query_Packet(query, best_nb_machines, buffer, MAX_MPI_MSG_SIZE, pkt_size)) {
                response.errtxt = "Encode_MPI_Packet failed.";
                break;
            }
            pthread_mutex_lock(&lock_thread_net);
            MPI_Send(buffer, pkt_size, MPI_CHAR, status_snapshot[sort_by_pending_jobs_ind[i]].rank, TAG_MPI_JOB_ASSIGN, MPI_COMM_WORLD);
            pthread_mutex_unlock(&lock_thread_net);
            pthread_mutex_lock(&mutex_machine_status);
            machines[sort_by_pending_jobs_ind[i]].n_pending_jobs += 1.0 / best_nb_machines;
            pthread_mutex_unlock(&mutex_machine_status);
        }
        pthread_mutex_lock(&mutex_response_array);
        Response_Array.insert(response);
        pthread_mutex_unlock(&mutex_response_array);
        //debug
        pthread_mutex_lock(&mutex_machine_status);
        print_status(machines);
        pthread_mutex_unlock(&mutex_machine_status);
    }
}

void* Thread_Response_Collector(void* params) {
    int dataWaitingFlag, pkt_size;
    MPI_Status Stat;
    RESPONSE_ARR_STRUCT response;
    RESPONSE_PACKET* pkt_response = new RESPONSE_PACKET;
    set<RESPONSE_ARR_STRUCT>::iterator it_response;
    char* buffer = new char [MAX_MPI_MSG_SIZE]; //used to receive results from worker machines.
    const int L = NUM_ITEMS_NEEDED*MAX_THREAD_PER_MACHINE;
    double rtt, processing_time;
    double ts;
    while (1) {
        /*check whether there is any reply from worker machines*/
        pthread_mutex_lock(&lock_thread_net);
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_MPI_JOB_REPLY, MPI_COMM_WORLD, &dataWaitingFlag, &Stat);
        if (dataWaitingFlag) { //Receive a packet
            ts = tic(); //timestamp receiving the response.
            /*Extract the packet*/
            MPI_Recv(buffer, MAX_MPI_MSG_SIZE, MPI_CHAR, Stat.MPI_SOURCE, TAG_MPI_JOB_REPLY, MPI_COMM_WORLD, &Stat); //blocking
            pthread_mutex_unlock(&lock_thread_net);
            MPI_Get_count(&Stat, MPI_CHAR, &pkt_size);
            if (!Decode_MPI_Response_Packet(pkt_response, buffer, pkt_size)) {
                printf("Thread_Response_Collector: failed to decode packet from %d with size %d.\n", Stat.MPI_SOURCE, pkt_size);
                continue;
            }
            
            /*Look up the corresponding response*/
            response.id = pkt_response->id;
            pthread_mutex_lock(&mutex_response_array);
            it_response = Response_Array.find(response);
            pthread_mutex_unlock(&mutex_response_array);
            if (it_response == Response_Array.end()) {
                printf("[Error]: cannot find response with id = %d in Response_Array.\n", pkt_response->id);
                continue;
            }
            /*Update response*/
            pthread_mutex_lock(&mutex_response_array);
            memcpy(it_response->merged_matching_distances + it_response->num_results*L, pkt_response->distances, sizeof (float)*L);
            memcpy(it_response->merged_item_ids + it_response->num_results*L, pkt_response->item_ids, sizeof (int)*L);
            it_response->num_results++;
            if (it_response->num_results == it_response->num_machines) {//done: create a thread to merge the result and return to the web user.
                it_response->ts_finished = tic();
                MERGE_PARAM* par = new MERGE_PARAM;
                par->distances = it_response->merged_matching_distances;
                par->item_ids = it_response->merged_item_ids;
                par->length = it_response->num_machines*L;
                par->sockfd = it_response->sockfd;
                pthread_t Tid_Thread_Merge_Score;
                Response_Array.erase(it_response);
                printf("Enqueue->Schedule: %.6f seconds. Schedule->Finished: %.6f seconds.\n",it_response->ts_schuduled-it_response->ts_enqueue,it_response->ts_finished-it_response->ts_schuduled);
                if (pthread_create(&Tid_Thread_Merge_Score, NULL, &Thread_Merge_Score, par) != 0) {
                    printf("[Error] System_Initialization: fail to create Thread_Merge_Score thread.");
                }
                pthread_join(Tid_Thread_Merge_Score, NULL);
            }
            pthread_mutex_unlock(&mutex_response_array);
            /*update machine status*/
            rtt = (ts - it_response->ts_schuduled) - (pkt_response->ts_end_process - pkt_response->ts_recv_by_slave); //round trip time [transmission].
            processing_time = pkt_response->ts_end_process - pkt_response->ts_begin_process;
            pthread_mutex_lock(&mutex_machine_status);
            machines[Stat.MPI_SOURCE].network_speed = 0.7 * machines[Stat.MPI_SOURCE].network_speed + 0.3 * 2.0 / rtt;
            machines[Stat.MPI_SOURCE].n_pending_jobs -= 1.0 / it_response->num_machines;
            if(machines[Stat.MPI_SOURCE].jobs_per_second==0)
                machines[Stat.MPI_SOURCE].jobs_per_second = (1.0 / it_response->num_machines) / processing_time;
            else
                machines[Stat.MPI_SOURCE].jobs_per_second = 0.7 * machines[Stat.MPI_SOURCE].jobs_per_second + 0.3 * (1.0 / it_response->num_machines) / processing_time;
            print_status(machines); //debug
            pthread_mutex_unlock(&mutex_machine_status);
        } else {
            pthread_mutex_unlock(&lock_thread_net);
            usleep(PROBE_SLEEP); //prevent busy wait.
        }
    }
}

void* Thread_Merge_Score(void* params) {
    MERGE_PARAM* par = (MERGE_PARAM*)params;
    float* distances = par->distances;
    int* item_ids = par->item_ids;
    int* order = new int [par->length];
    quick_sort(distances, par->length, order, 0, false); //sort ascendingly.
    vector<string> titles;
    vector<string> asins;
    vector<string> urls;
    for(int i = 0;i<NUM_ITEMS_NEEDED;i++){
        titles.push_back(database->get_title(item_ids[order[i]]));
        asins.push_back(database->get_asin(item_ids[order[i]]));
        urls.push_back(database->get_url(item_ids[order[i]]));
        //titles.push_back("");
        //asins.push_back("");
        //urls.push_back(database->get_asin(item_ids[order[i]])+"_0.JPG");
        //printf("item_line = %d, score = %.2f, title = %s\n",1+item_ids[order[i]],distances[order[i]], database->get_title(item_ids[order[i]]).c_str());
    }
    string table = format_into_web(titles,asins,urls);
    pthread_mutex_lock(&lock_thread_tcp);
    send(par->sockfd, &table[0], table.size(), 0);
    pthread_mutex_unlock(&lock_thread_tcp);
    /*int start = 0;
    int sz = 0;
    while(start<table.size()){
    	if(table.size()-start<MAX_TCP_PKT_SIZE)
    		sz = table.size()-start;
    	else
    		sz = MAX_TCP_PKT_SIZE;
    	send(par->sockfd, &table[start], sz, 0);
    	start += sz;
    	//printf("%d:%d\n",start,table.size());
    }*/
    close(par->sockfd);
    delete par, order;
}

/*
    MERGE_PARAM* par = (MERGE_PARAM*)params;
    float* distances = par->distances;
    int* item_ids = par->item_ids;
    int* order = new int [par->length];
    quick_sort(distances, par->length, order, 0, false); //sort ascendingly.
    vector<string> titles;
    string asins;
    vector<string> urls;
    for(int i = 0;i<NUM_ITEMS_NEEDED;i++){
        //titles.push_back(database->get_title(item_ids[order[i]]));
        if(i>0)
        	asins += ","+("'"+database->get_asin(item_ids[order[i]])+"'");
        else
        	asins = "'"+database->get_asin(item_ids[order[i]])+"'";
    }
    printf("asin: %s\n",asins.c_str());
    //string table = format_into_web(titles,asins,urls);
    string table = get_html_page(asins);
    pthread_mutex_lock(&lock_thread_tcp);
    send(par->sockfd, &table[0], table.size(), 0);
    pthread_mutex_unlock(&lock_thread_tcp);
    close(par->sockfd);
    delete par, order;
*/

void* Thread_Wait_Job_From_Master(void* params) {
    char* buffer = new char [MAX_MPI_MSG_SIZE]; //used to receive query from master.
    char* send_response_packet_buffer_shared = new char [MAX_MPI_MSG_SIZE];
    RESPONSE_PACKET* responses = new RESPONSE_PACKET [MAX_PENDING_JOB_SLAVE * N_MACHINE + 1];
    double* code_weight = new double [database->get_nb_codes()];
    database->get_weight(code_weight);
    int pkt_size, dataWaitingFlag, cnt = 0;
    QUERY_EVAL_PARAM* param = 0;
    MPI_Status Stat;
    pthread_t Tid_Thread_Slave_Do_Query;
    double ts;
    printf("[%d] Thread_Wait_Job_From_Master starts up.\n", RANK);
    while (1) {
        pthread_mutex_lock(&lock_thread_net);
        MPI_Iprobe(MASTER_RANK, TAG_MPI_JOB_ASSIGN, MPI_COMM_WORLD, &dataWaitingFlag, MPI_STATUS_IGNORE);
        if (dataWaitingFlag) { //Receive a packet
            ts = tic();
            MPI_Recv(buffer, MAX_MPI_MSG_SIZE, MPI_CHAR, MASTER_RANK, TAG_MPI_JOB_ASSIGN, MPI_COMM_WORLD, &Stat); //blocking
            pthread_mutex_unlock(&lock_thread_net);
            MPI_Get_count(&Stat, MPI_CHAR, &pkt_size);
            param = new QUERY_EVAL_PARAM;
            param->query = new QUERY_PACKET;
            param->response = responses + cnt;
            param->response->ts_recv_by_slave = ts;
            param->buffer = send_response_packet_buffer_shared;
            param->buffer_size = MAX_MPI_MSG_SIZE;
            param->code_weight = code_weight;
            if (!Decode_MPI_Query_Packet(param->query, buffer, pkt_size)) {
                printf("[%d] Failed to decode query packet.\n", RANK);
                continue;
            }
            //printf("[%d] got job with id = %d.\n",RANK,param->query->id);
            /*Create a thread do the query.*/
            if (pthread_create(&Tid_Thread_Slave_Do_Query, NULL, &Thread_Slave_Do_Query, param) != 0) {
                printf("[Error] System_Initialization: fail to create Thread_Wait_Job_From_Master thread.");
                MPI_Finalize();
                exit(-1);
            }
            cnt = (cnt + 1) % (MAX_PENDING_JOB_SLAVE * N_MACHINE + 1);
        } else {
            pthread_mutex_unlock(&lock_thread_net);
            usleep(PROBE_SLEEP);
        }
    }
}

void Thread_Compute_Matching_Distance(void* params) {
    MATCHING_DISTANCE_THREAD_PARAM& par = *(MATCHING_DISTANCE_THREAD_PARAM*) params;
    int i, j, k, p_length, *p_code;
    float *p_count;
    //float 
    double score, pivot;
    float* distances = par.distances;
    float* distances_buf = par.distances_buf;
    double* weights = par.weights;
    int q_length = par.q_length;
    float* q_count = par.q_count;
    int* q_code = par.q_code;
    int begin = par.begin, end = par.end; //global id. beginning at zero.
    int nitem = end - begin + 1;
    int l = 0;
    for(i=0;i<nitem;i++){  //for each item.
        p_length = database->get_row(begin+i,p_code); //get one item from database.
        /*if(begin+i==0){
        	printf("q_length = %d\n",q_length);
        	puts("query codes:");
        	for(int ss = 0;ss<q_length;ss++)
        		printf("%d ",q_code[ss]);
        	puts("\n");
        	puts("query counts:");
        	for(int ss = 0;ss<q_length;ss++)
        		printf("%.2f ",q_count[ss]);
        	puts("\n");
        	printf("p_length = %d\n",p_length);
        	puts("product codes:");
        	for(int ss = 0;ss<p_length;ss++)
        		printf("%d ",p_code[ss]);
        	puts("\n");
        	puts("product counts:");
        	for(int ss = 0;ss<p_length;ss++)
        		printf("%.2f ",p_count[ss]);
        	puts("\n");
        }*/
        
        
        /*if(p_length<0){
            printf("Error: get one row with -1 length. id = %d\n",begin+i);
            return;
        }*/
        
        j = 0; //j->query, k->product.
        k = 0;
        l = 0;
        score = 0.0;
        while (j < q_length && k < p_length) {
            if (q_code[j] == p_code[k]) {
                score -= weights[p_code[k]]; //p_count[k] * q_count[j];// weights[p_code[k]];
                k++;
                j++;
                l++;
            }
            else if (p_code[k] > q_code[j])
                j++;
            else
                k++;
        }
        if(q_code[0]>162454)
        	par.distances[i] = score/(p_length+q_length-l);
        else
        	par.distances[i] = score;
    }
    
    memcpy(distances_buf,distances,sizeof(float)*nitem);
    nth_element(distances,distances+par.num_items_needed,distances+nitem);
    pivot = distances[par.num_items_needed];
    j = 0;
    for(i=0;i<nitem && j<par.num_items_needed;i++){  //all elements less than pivot.
        if(distances_buf[i]<pivot){
            distances[j] = distances_buf[i];
            par.item_ids[j++] = begin + i;
        }
    }
    for(i=0;i<nitem && j<par.num_items_needed;i++){  //all elements equal to pivot.
        if(distances_buf[i]==pivot){
            distances[j] = distances_buf[i];
            par.item_ids[j++] = begin + i;
        }
    }
}

void* Thread_Slave_Do_Query(void* PAR) {
    QUERY_EVAL_PARAM* p = (QUERY_EVAL_PARAM*) PAR;
    RESPONSE_PACKET* response = p->response;
    response->ts_begin_process = tic();
    int nitems = database->get_nb_items();
    int L = floor(nitems / p->query->n_workers);
    int begin = L * p->query->part; //starting at zero.
    int end = begin + L - 1; //end at nitem-1.
    if (p->query->part == p->query->n_workers)
        end = nitems - 1;   
    //printf("[%d] Thread_Slave_Do_Query: qid=%d,nitems=%d,L=%d,begin=%d,end=%d,q_length=%d,part=%d,nworker=%d,height=%d,width=%d\n",RANK,p->query->id,nitems,L,begin,end,p->query->n_code,p->query->part,p->query->n_workers,p->query->width,p->query->height);
    Parallel PL;
    vector<void*> thread_params;
    int nb_item_per_thread = floor((end - begin + 1) / MAX_THREAD_PER_MACHINE);
    MATCHING_DISTANCE_THREAD_PARAM* params = new MATCHING_DISTANCE_THREAD_PARAM [MAX_THREAD_PER_MACHINE];
    for (int k = 0; k < MAX_THREAD_PER_MACHINE; k++) {
        params[k].begin = begin + k*nb_item_per_thread; //offset within this machine.
        if (k < MAX_THREAD_PER_MACHINE - 1)
            params[k].end = begin + (k + 1) * nb_item_per_thread - 1;  //offset within this machine.
        else
            params[k].end = end;
        params[k].q_code = p->query->code;
        params[k].q_count = p->query->count;
        params[k].q_length = p->query->n_code;
        params[k].weights = p->code_weight;
        params[k].distances = response->distances + k*nb_item_per_thread;
        params[k].distances_buf = response->distances_buf + k*nb_item_per_thread;
        params[k].item_ids = response->item_ids + k*NUM_ITEMS_NEEDED;
        params[k].num_items_needed = NUM_ITEMS_NEEDED;
        thread_params.push_back(params + k);
    }
    PL.Run(Thread_Compute_Matching_Distance, thread_params);
    //merge scores from different threads.
    for (int k = 0; k < MAX_THREAD_PER_MACHINE; k++)
        memcpy(response->distances+k*NUM_ITEMS_NEEDED,response->distances+begin+k*nb_item_per_thread,sizeof(float)*NUM_ITEMS_NEEDED);
    response->id = p->query->id;
    int pkt_size = 0;
    pthread_mutex_lock(&lock_thread_net);
    response->ts_end_process = tic();
    if (!Encode_MPI_Response_Packet(response, p->buffer, p->buffer_size, pkt_size)) {
        printf("[%d] Failed to encode response packet.\n", RANK);
    } else {
        MPI_Send(p->buffer, pkt_size, MPI_CHAR, MASTER_RANK, TAG_MPI_JOB_REPLY, MPI_COMM_WORLD);
    }

    pthread_mutex_unlock(&lock_thread_net);
    return 0 ;
}

/*void* Thread_Slave_Wait_For_Ping(void* params) {
    int dataWaitingFlag = 0;
    char* buf = new char [MAX_MPI_MSG_SIZE];
    MPI_Status Stat;
    pthread_mutex_lock(&lock_thread_net);
    for (int i = 0; i < NET_INIT_NUM_RT; i++) {
        dataWaitingFlag = 0;
        while (dataWaitingFlag == 0) { //wait for incoming ping request.
            MPI_Iprobe(MASTER_RANK, TAG_MPI_PING, MPI_COMM_WORLD, &dataWaitingFlag, MPI_STATUS_IGNORE);
            if (dataWaitingFlag) {
                MPI_Recv(buf, MAX_MPI_MSG_SIZE, MPI_CHAR, MASTER_RANK, TAG_MPI_PING, MPI_COMM_WORLD, &Stat); //wait until recv done.
                MPI_Send(buf, MAX_MPI_MSG_SIZE, MPI_CHAR, MASTER_RANK, TAG_MPI_PING, MPI_COMM_WORLD); //block until message received by destination.
            }
        }
    }
    pthread_mutex_unlock(&lock_thread_net);
}*/

void* Thread_Refresh_CPU_Usage(void* params){
    char buf[100];
    sprintf(buf,"cpu_usage_%d.txt",RANK+1);
    string root = string(ROOT_DIR"/system/WebInterface/log/")+buf;
    FILE* fp = fopen(root.c_str(),"w+");
    if(!fp){
        printf("[Warning] Cannot open the %s file for writing at rank = %d\n",root.c_str(),RANK);
        return 0;
    }
    fclose(fp);
    vector<float> cpu_usage;
    int i;
    float mean;
    while(1){    
        if(!fp){
            usleep(1000); continue;
        }
        buf[0] = 0;
        mean = 0;
        cpu_usage = get_load(500); //refresh every 500 ms
        for(i=0;i<cpu_usage.size();i++)
            mean += cpu_usage[i];
        mean /= cpu_usage.size();
        for(i=0;i<cpu_usage.size();i++){
            sprintf(buf+strlen(buf),"%.2f ",cpu_usage[i]);
        }
        sprintf(buf+strlen(buf),"%.2f ",mean);
        fp = fopen(root.c_str(),"w+");
        fprintf(fp,"%s",buf);
        fclose(fp);
    }
}

/*Procedures*/

bool Encode_MPI_Query_Packet(QUERY_QUEUE_STRUCT& query, const int& n_workers, char* buffer, const int& buf_size, int& pkt_size) {
    query.qpkt.n_workers = n_workers;
    pkt_size = sizeof (QUERY_PACKET);
    if (pkt_size <= buf_size) {
        memcpy(buffer, &query.qpkt, pkt_size);
        return true;
    } else
        return false;
}

bool Decode_MPI_Query_Packet(QUERY_PACKET* query, const char* data, const int & data_size) {
    if (data_size == sizeof (QUERY_PACKET)) {
        memcpy(query, data, data_size);
        return true;
    } else {
        printf("s1 = %d, s2 = %d\n", data_size, sizeof (QUERY_PACKET));
        return false;
    }
}

bool Encode_MPI_Response_Packet(RESPONSE_PACKET* response, char* buffer, const int& buf_size, int&pkt_size) {
    pkt_size = NUM_ITEMS_NEEDED*MAX_THREAD_PER_MACHINE * sizeof (float) + (char*)response->distances- (char*)(&response->id);
    if (pkt_size > buf_size){
        printf("Error: pkt_size = %d, buf_size = %d, violating pkt_size < buf_size.",pkt_size, buf_size);
        return false;
    }
    memcpy(buffer, response, pkt_size);
    return true;
}

bool Decode_MPI_Response_Packet(RESPONSE_PACKET* response, const char* data, const int&data_size) {
    int max_pkt_size = sizeof (RESPONSE_PACKET);
    if (data_size > max_pkt_size) return false;
    memcpy(response, data, data_size);
    return true;
}

int get_unique_query_id() {
    static int initial_id = 0;
    initial_id = (initial_id + 1) % (MAX_PENDING_QUERY * N_MACHINE);
    return initial_id;
}

vector<string> stem(char* keywords) { //"There are 3 students" => "THERE ARE 3 STUDENT". The input must ends with '\0'
    int begin = 0;
    int end = 0;
    vector<string> ret;
    string tmp;
    while (keywords[begin]) {
        while (keywords[end] != ' ' && keywords[end] != 0) end++;
        tmp = (char*) sb_stemmer_stem(stemmer, (sb_symbol*) (keywords + begin), end - begin);
        transform(tmp.begin(), tmp.end(), tmp.begin(), ::toupper);
        ret.push_back(tmp);
        if (keywords[end] == 0) break;
        begin = end + 1;
        end = begin;
    }
    return ret;
}

void System_Initialization_Master() {
    /*1) Create a thread to accept incoming query.*/
    FILE* fp = fopen("/home/dihong/AWS-site/log/log.txt", "w+");
    if (fp) fclose(fp); //empty log  
    if (pthread_create(&Tid_Thread_Accept_Query, NULL, &Thread_Accept_Query, 0) != 0) {
        printf("[Error] System_Initialization: fail to create Thread_Accept_Query thread.");
        return;
    }

    /*2) Initialize machine status*/
    for (int i = 0; i < N_MACHINE; i++) {
        machines[i].n_pending_jobs = 0;
        machines[i].rank = i;
        machines[i].jobs_per_second = 0.0;
        machines[i].network_speed = 0.0;
    }
    /*int i, j, dataWaitingFlag, num_replied = 0;
    char* buf = new char [MAX_MPI_MSG_SIZE];
    double ts, rtt;
    pthread_mutex_lock(&lock_thread_net);
    for (i = 0; i < N_MACHINE; i++) {
        rtt = 0;
        for (j = 0; j < NET_INIT_NUM_RT; j++) {
            //send ping signals to slave i.
            ts = tic();
            MPI_Send(buf, MAX_MPI_MSG_SIZE, MPI_CHAR, i, TAG_MPI_PING, MPI_COMM_WORLD); //block until message received by destination.
            //wait for signal from i.
            dataWaitingFlag = 0;
            while (!dataWaitingFlag) {
                MPI_Iprobe(i, TAG_MPI_PING, MPI_COMM_WORLD, &dataWaitingFlag, MPI_STATUS_IGNORE);
                if (dataWaitingFlag) {
                    MPI_Recv(buf, MAX_MPI_MSG_SIZE, MPI_CHAR, i, TAG_MPI_PING, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    rtt += toc(ts);
                }
            }
        }
        rtt /= NET_INIT_NUM_RT;
        machines[i].network_speed = 2.0 / rtt;
        print_status(machines, i);
    }
    pthread_mutex_unlock(&lock_thread_net);
    printf("[%d] Network initialization done.\n", RANK);
    delete buf;
    */
    /*3) Create a thread to collect responses from worker machines.*/
    pthread_t Tid_Thread_Response_Collector;
    if (pthread_create(&Tid_Thread_Response_Collector, NULL, &Thread_Response_Collector, 0) != 0) {
        printf("[Error] System_Initialization: fail to create Thread_Response_Collector thread.");
        return;
    } else {
        printf("[%d] Thread_Response_Collector starts up.\n", RANK);
    }

    /*4) Create a thread to schedule queries.*/
    pthread_t Tid_Thread_Scheduler;
    if (pthread_create(&Tid_Thread_Scheduler, NULL, &Thread_Scheduler, 0) != 0) {
        printf("[Error] System_Initialization: fail to create Tid_Thread_Scheduler thread.");
        MPI_Finalize();
        exit(-1);
    }

    /*6) Initialize stemming*/
    stemmer = sb_stemmer_new("english", 0);
    if (!stemmer) {
        puts("[Error] Failed to initialize stemmer.");
        return;
    }
}

void System_Initialization_Slave() {
    /* THIS FUNCTION INITIALIZES THE SLAVE SYSTEM.*/
    /*1) Wait for ping signals from master*/
    /*pthread_t Tid_Thread_Slave_Wait_For_Ping;
    if (pthread_create(&Tid_Thread_Slave_Wait_For_Ping, NULL, &Thread_Slave_Wait_For_Ping, 0) != 0) {
        printf("[Error] System_Initialization: fail to create Thread_Slave_Wait_For_Ping thread.");
        MPI_Finalize();
        exit(-1);
    }*/

    /*2) Create a thread to wait for job from master.*/
    if (pthread_create(&Tid_Thread_Wait_Job_From_Master, NULL, &Thread_Wait_Job_From_Master, 0) != 0) {
        puts("[Error] System_Initialization: fail to create Thread_Wait_Job_From_Master thread.");
        MPI_Finalize();
        exit(-1);
    }
    
    /*1) Update CPU usage*/
    pthread_t Tid_Thread_Refresh_CPU_Usage;
    if (pthread_create(&Tid_Thread_Refresh_CPU_Usage, NULL, &Thread_Refresh_CPU_Usage, 0) != 0) {
        printf("[Error] System_Initialization: fail to create Tid_Thread_Refresh_CPU_Usage thread at rank = %d\n",RANK);
        MPI_Finalize();
        exit(-1);
    }
}

int main(int argc, char *argv[]) {
    /*MPI initialization.*/
    int num_processors;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processors);
    if (num_processors != N_MACHINE + 1) {
        printf("[Error] num_processors!=N_MACHINE %d:%d\n", num_processors, N_MACHINE);
        MPI_Finalize();
        return -1;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &RANK);
    usleep(1000);
    /*if (RANK == 0) {
        float* x = new float [5];
        x[0] = 2.1;
        x[1] = 1.0;
        x[2] = 1.3;
        x[3] = 1.0;
        x[4] = 4.0;
        int* ind = new int [5];
        insertion_sort(x, 5, ind);
        for (int i = 0; i < 5; i++) printf("[%d %.2f]\n", ind[i],x[i]);
    }*/
    database = new DB;
    /* THIS FUNCTION INITIALIZES THE MASTER SYSTEM.*/
    /*0) Initialize database*/
    if (RANK == MASTER_RANK) { //master
        System_Initialization_Master();
        sleep(1);
        printf("%sMaster (Rank %d) initialization done.%s\n", KGRN,RANK,KNRM);
        pthread_join(Tid_Thread_Accept_Query, NULL);
    } else {
        System_Initialization_Slave();
        printf("%sSlave (Rank %d) initialization done.%s\n", KGRN,RANK,KNRM);
        pthread_join(Tid_Thread_Wait_Job_From_Master, NULL);
    }
    printf("[%d] is now exiting.\n", RANK);
    fflush(stdout);
    MPI_Finalize();
}
