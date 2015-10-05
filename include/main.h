#ifndef MAIN_H
#define MAIN_H
#include "stdio.h"
#include "parallel.h"
#include "database.h"
#include <stdio.h>
#include<fstream>
#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <queue>
#include <string>
#include <vector>
#include <set>
#include<opencv2/highgui/highgui.hpp>
#include<opencv/cv.h>
#include <unistd.h>



#define N_MACHINE 3
#define MAX_THREAD_PER_MACHINE 12  //maximum number of threads on a single machine.
#define MASTER_RANK 3
#define MAX_PENDING_QUERY 100 //the maximum number of pending queries allowed in the Incomming_Query_Queue
#define MAX_PENDING_JOB_SLAVE 5 //the maximum number of pending jobs in the slave allowd.
#define MAX_MPI_MSG_SIZE 1000000 //1MB
#define MAX_NUM_ITEMS_DB 5500000 //6M, the maximum number of different items in the database. 
#define MAX_NUM_CODE_PER_QUERY 2000
#define TAG_MPI_JOB_ASSIGN 1
#define TAG_MPI_JOB_REPLY 2
#define TAG_MPI_PING 3
#define PROBE_SLEEP 1  //sleep every 1 micro-second after MPI_Iprobe to prevent busy wait.
#define NET_INIT_NUM_RT 100  //the number of round trips used to calculate network speed for initialization.
#define ROOT_DIR "/home/dihong" //the root working director.
#define ROOT_DIR_MASTER "/home/dihong" //the root working director.
#define NUM_ITEMS_NEEDED 20 //only return top 100 items.
#define DIM_FEAT 128  //dimension of sift features.
#define LENGTH_ASIN 10
#define MAX_TCP_PKT_SIZE 1000


#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

typedef unsigned char uchar;

typedef struct QUERY_PACKET{
    int id; //query id
    int n_workers; //number of workers are participating the query evaluation
    int part; //the part of which this worker is responsible. part /in {0,1,...,n_workers-1}.
    int n_code; //number of text+image keywords.
    int code[MAX_NUM_CODE_PER_QUERY]; //the code of item (TEXT+IMAGE).
    float count[MAX_NUM_CODE_PER_QUERY]; //the count of codes.
}QUERY_PACKET;

class RESPONSE_PACKET{
public:
    int id; //query id
    double ts_recv_by_slave; //time stamp received by processing.
    double ts_begin_process; //time stamp at the beginning of processing.
    double ts_end_process; //time stamp after processing.
    int item_ids[NUM_ITEMS_NEEDED*MAX_THREAD_PER_MACHINE]; //item id corresponding to distances.
    float distances[MAX_NUM_ITEMS_DB]; //computed matching distances
    float distances_buf[MAX_NUM_ITEMS_DB]; //buffer for distances
};

typedef struct MATCHING_DISTANCE_THREAD_PARAM{
    int* q_code; //codes [txt+image]
    float* q_count; //normalized scores of each feature. [txt+image]
    int q_length; //query length [number of keywords: txt+image]
    int begin;
    int end;
    double* weights; //the code weights
    int num_items_needed; //only return part of the items in the database that 'best' matches the query.
    float* distances; //the computed matching distances between query and items in the database.
    float* distances_buf;
    int* item_ids; //the id of the item corresponding to distances.
}MATCHING_DISTANCE_THREAD_PARAM;


typedef struct QUERY_EVAL_PARAM{
    QUERY_PACKET* query;
    double* code_weight; //weights of the codes. read only shared by threads.
    RESPONSE_PACKET* response; //memory used to store the response object. Cannot be shared by threads.
    char* buffer; //shared by all threads, synchronized by lock_thread_net
    int buffer_size; //size of the buffer.
}QUERY_EVAL_PARAM;


typedef struct MERGE_PARAM{
    float* distances;
    int* item_ids;
    int length;
    int sockfd;
}MERGE_PARAM;

typedef struct QUERY_QUEUE_STRUCT{
    QUERY_PACKET qpkt; //used for transmission between master and slave machines.
    int sockfd; //the socket connection corresponding to the query.
    double ts_enqueue; //time stamp entering the queue.
}QUERY_QUEUE_STRUCT;


typedef struct RESPONSE_ARR_STRUCT{
    int id; //id of the query.
    int sockfd; //the socket connection corresponding to the query.
    mutable double ts_enqueue; //time stamp entering the queue.
    mutable double ts_schuduled; //time stamp being scheduled.
    mutable double ts_finished; //time stamp finish.
    mutable int num_machines; //number of machines used for calculating the query.
    mutable int num_results; //number of results received.
    mutable std::string errtxt; //if there is any error, errtxt will be set.
    float* merged_matching_distances; //the scores computed by worker machines.
    int* merged_item_ids; //the corresponding id.
    bool operator<(const RESPONSE_ARR_STRUCT& other) const {
        return id < other.id;
    }
} RESPONSE_ARR_STRUCT;


typedef struct MACHINE_STATUS{
    float n_pending_jobs;
    float jobs_per_second;
    int rank; //0,1,... the rank value assigned by MPI.
    float network_speed; //[packet/second]the network communication speed between master machine and slave machine. 
}MACHINE_STATUS;

/*This thread accepts queries from php via TCP, and insert the queries into incoming queue.*/
void* Thread_Accept_Query(void* params);

/*This thread Takes one query from the Incomming_Query_Queue, and assigns it to the machines according to their workloads.*/
void* Thread_Scheduler(void* params);

/*This thread collects responses from worker machines.*/
void* Thread_Response_Collector(void* params);


/*This thread merge and sort the scores and return result to web user.*/
void* Thread_Merge_Score(void* params);

/*This thread waits for job from master and insert the job into a pending queue*/
void* Thread_Wait_Job_From_Master(void* params);

/*This thread actually implements the query: text+image. Return scores.*/
void* Thread_Slave_Do_Query(void* params);

/*serialize objects into a block of memory for network transmission
 * query[In]: the query to be serialized.
 * n_workers[In]: the number of machines used for computing this query.
 * buffer[Out]: the buffer memory to store the output serialized data.
 * buf_size[In]: the size of the buffer provided.
 * pkt_size[Out]: the size of encoded packet [pkt_size <= buf_size].
 * Return true if success, return false otherwise.
 */
bool Encode_MPI_Query_Packet(QUERY_QUEUE_STRUCT& query, const int& n_workers, char* buffer, const int& buf_size, int& pkt_size);

/*reconstruct objects from a block of memory
 * query[Out]: the decoded query structure.
 * data[In]: the data received from MPI to be decoded.
 * data_size[In]: the size in bytes of the data.
 * Return true if success, return false otherwise.
 */
bool Decode_MPI_Query_Packet(QUERY_PACKET* query, const char* data, const int & data_size);

/*Encode a response into packet for transmission from worker to master.
 * response[In]: the response packet to be serialized.
 * buffer[Out]: the encoded data
 * buf_size[In]: the maximum size of the buffer.
 * pkt_size[Out]: the actual size of the encoded packet.
 * Return true is success, false otherwise.
 */
bool Encode_MPI_Response_Packet(RESPONSE_PACKET* response, char* buffer, const int& buf_size, int&pkt_size);

/*Decode a response packet.
 * RESPONSE_PACKET[Out]: decoded response packet.
 * data[In]: the buffer used to stored decoded data.
 * data_size[In]: the size of 'data'.
 * Return true is success, false otherwise.
 */
bool Decode_MPI_Response_Packet(RESPONSE_PACKET* response, const char* data, const int&data_size);


int get_unique_query_id(); //this function gets a unique query id

void print_status(MACHINE_STATUS* M, int index_of_machine=-1);

std::vector<std::string> stem(char* keywords);

#endif
