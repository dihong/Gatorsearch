#ifndef COMMON_H
#define COMMON_H

#include "main.h"
#include <string.h>

#include <memory.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <queue>
#include <string>
#include <ctime>
#include <fstream>
#include <map>
#include <iostream>
#include <sstream>
#include <utility>
#include <unistd.h>
#include <boost/lexical_cast.hpp>
#include <boost/regex.hpp>
#include <boost/date_time.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <stdio.h>
#include "mysql_connection.h"
#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <cppconn/resultset.h>
#include <cppconn/statement.h>
#include <cppconn/prepared_statement.h>

using namespace std;

extern double begin_ts; //used for measuring time elapsed.

const int sz_dict = 548;  //number of dict
const int sz_title = 1470;  //number of titles
const int sz_total_index = 14836;  //size of array storing inverted index

const int n_thread = 6;
const int n_machine = 3;
const int n_query_returned = 10;  //specify the number of results to be returned

const bool debug = false;

typedef struct {
    int unique_i;

    int row_begin;
    int row_end;
} R_INFO;


class CompareFirstInt {
public:
    bool operator()(pair<int, int>& l, pair<int, int>& r)
    {
       return l.first > r.first;
    }
};


class CompareSecondDouble {
public:
    bool operator()(pair<int, double>& l, pair<int, double>& r)
    {
       return l.first < r.first;
    }
};

/*
const string prefix = "./model/";
const string indexfilename = prefix + "index.txt";
string freqfilename = prefix + "freq.txt";
string wordfilename = prefix + "mapper.txt;";

string indexfilename_bin = prefix +  "index.bin";
string freqfilename_bin = prefix + "freq.bin";
string wordfilename_bin = prefix + "mapper.bin";
*/



void quick_sort(float* arr, int N, int* order, float* sorted = 0, bool descend = true);
void quick_sort(int* arr, int N, int* order, int* sorted = 0, bool descend = true);
void quick_sort(double* arr, int N, int* order, double* sorted = 0, bool descend = true);
double tic();
double toc();
double toc(double bts);
std::string format_into_web(std::vector<std::string> titles, std::vector<std::string> asins, std::vector<std::string>urls);
std::string format_into_web(std::vector<std::string> titles);
vector<float> get_load(unsigned interval_milliseconds);
void insertion_sort(int* arr, int N);
void insertion_sort(float* arr, int N, int* order);
std::string get_html_page(std::string ASIN);
#endif
