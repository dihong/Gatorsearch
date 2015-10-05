/*
 * File:   parallel.h
 * Author: dihong
 *
 * Created on September 22, 2013, 4:46 PM
 */

#ifndef PARALLEL_H
#define	PARALLEL_H
#include<vector>
#include <pthread.h>
#include <cstring>
#include "common.h"

typedef struct{
    std::vector<int> unique_is;
    std::vector< std::pair<int, double> > result;

} PARALLEL_THREAD_PARAM;

class ARG_EXE_THREAD{
public:
    void* param;
    void (*func)(void*);
};

class Parallel{
    ARG_EXE_THREAD* args;
    int nw; //number of worker threads.
    pthread_t* tid;
	static void* execute_thread(void* arg);
public:
	void Run(void (*f)(void*),std::vector<void*>& param);
	Parallel(){args = 0; tid = 0;}
    ~Parallel();
};
#endif	/* PARALLEL_H */
