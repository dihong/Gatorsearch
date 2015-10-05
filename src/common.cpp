#include "common.h"
#include <mpi.h>

using namespace std;
using namespace boost;

double begin_ts = MPI_Wtime();

/*
bool comp_int_double(const pair<int, double>& l, const pair<int, double>& r) {
    return l.second > r.second;
}

bool comp_int_int_small(const pair<int, int>& l, const pair<int, int>& r) {
    return l.first > r.first;
}
 */

bool mycompfunc_double(const pair<double, int>& l, const pair<double, int>& r) {
    return l.first > r.first;
}

bool mycompfunc_int(const pair<int, int>& l, const pair<int, int>& r) {
    return l.first > r.first;
}

bool mycompfunc_float(const pair<float, int>& l, const pair<float, int>& r) {
    return l.first > r.first;
}

void quick_sort(float* arr, int N, int* order, float* sorted, bool descend) {
    vector< pair<float, int> > WI;
    pair<float, int> val_ind;
    for (int i = 0; i < N; i++) {
        val_ind.first = arr[i]; //value.
        val_ind.second = i; //index.
        WI.push_back(val_ind);
    }
    sort(WI.begin(), WI.end(), mycompfunc_float);
    if (descend)
        for (int i = 0; i < N; i++) {
            if (sorted) sorted[i] = WI[i].first;
            order[i] = WI[i].second;
        }
    else
        for (int i = 0; i < N; i++) {
            if (sorted) sorted[i] = WI[N-i-1].first;
            order[i] = WI[N-i-1].second;
        }
}

void quick_sort(double* arr, int N, int* order, double* sorted, bool descend) {
    vector< pair<double, int> > WI;
    pair<double, int> val_ind;
    for (int i = 0; i < N; i++) {
        val_ind.first = arr[i]; //value.
        val_ind.second = i; //index.
        WI.push_back(val_ind);
    }
    sort(WI.begin(), WI.end(), mycompfunc_double);
    if (descend)
        for (int i = 0; i < N; i++) {
            if (sorted) sorted[i] = WI[i].first;
            order[i] = WI[i].second;
        }
    else
        for (int i = 0; i < N; i++) {
            if (sorted) sorted[i] = WI[N-i-1].first;
            order[i] = WI[N-i-1].second;
        }
}

void quick_sort(int* arr, int N, int* order, int* sorted, bool descend) {
    vector< pair<int, int> > WI;
    pair<int, int> val_ind;
    for (int i = 0; i < N; i++) {
        val_ind.first = arr[i]; //value.
        val_ind.second = i; //index.
        WI.push_back(val_ind);
    }
    sort(WI.begin(), WI.end(), mycompfunc_int);
    if (descend)
        for (int i = 0; i < N; i++) {
            if (sorted) sorted[i] = WI[i].first;
            order[i] = WI[i].second;
        } 
    else
        for (int i = 0; i < N; i++) {
            if (sorted) sorted[i] = WI[N-i-1].first;
            order[i] = WI[N-i-1].second;
        }
}

void insertion_sort(int* arr, int N){ //ascending in-place.
    static int i, j, tmp;
    for(i=1;i<N;i++){
        j = i-1;
        tmp = arr[i];
        while(j>-1 && arr[j]>tmp){
            arr[j+1] = arr[j];
            j--;
        }
        arr[j+1] = tmp;
    }
}

void insertion_sort(float* arr, int N){ //ascending in-place.
    static int i, j;
    static float key;
    for(i=1;i<N;i++){
        key = arr[i];
        for(j=i-1; j>-1 && arr[j]>key;j--){
            arr[j+1] = arr[j];
        }
        arr[j+1] = key;
    }
}

double tic() {
    begin_ts = MPI_Wtime();
    return begin_ts;
}

double toc() {
    return MPI_Wtime() - begin_ts;
}


double toc(double bts) {
    return MPI_Wtime() - bts;
}

string format_into_web(vector<string> titles){
    string ret = "<table border=\"0\"  width=\"100%\">";
    for(int i = 0;i<titles.size();i++){
        ret += "<tr><td>" +titles[i] + "</td></tr>";
    }
    ret += "</table>";
    return ret;
}

string format_into_web(vector<string> titles, vector<string> asins, vector<string>urls){
    string ret = "<table border=\"0\"  width=\"600\" cellpadding='20'>";
    char imglink [500];
    for(int i = 0;i<titles.size();i++){
    	FILE* fp = fopen((string("/media/dihong/Datascience/system/WebInterface/image/") + asins[i]).c_str(),"r");
    	if(fp){
    		fscanf (fp,"%s",imglink);
    		fclose(fp);
    	}else
    		printf("Cannot open: %s\n",(string("/media/dihong/Datascience/system/WebInterface/image/") + asins[i]).c_str());
        ret += "<tr><td align=\"center\"><img src=\""+string(imglink)+"\" style=\"height:100px;max-width:100px;\"></td><td><a href=\""+urls[i]+"\" target = \"_blank\">" +titles[i] + "</a></td></tr>";
    }
    ret += "</table>";
    return ret;
}

void print_status(MACHINE_STATUS* M, int index_of_machine){
    char buf[5];
    string fn = ROOT_DIR"/system/WebInterface/log/machine_status_";
    if(index_of_machine<0){ //update all status.
        for(int i = 0;i<N_MACHINE;i++){
            sprintf(buf,"%d",M[i].rank+1);
            FILE* fp = fopen((fn+buf+".txt").c_str(),"w+");
            if(fp){
                fprintf(fp,"Pending Jobs:%.2f\nNetwork (pkt/sec):%d\nJob (job/sec):%.2f\n",M[i].n_pending_jobs<0? 0:M[i].n_pending_jobs,(int)M[i].network_speed,M[i].jobs_per_second);
                fclose(fp);
            }
        }
    } else if (index_of_machine < N_MACHINE) {
        sprintf(buf, "%d", M[index_of_machine].rank+1);
        FILE* fp = fopen((fn + buf + ".txt").c_str(), "w+");
        if (fp) {
            fprintf(fp, "Pending Jobs:%.2f\nNetwork (pkt/sec):%d\nJob (job/sec):%.2f\n", M[index_of_machine].n_pending_jobs<0?0:M[index_of_machine].n_pending_jobs, (int)M[index_of_machine].network_speed, M[index_of_machine].jobs_per_second);
            fclose(fp);
        }
    }
}

//This function reads /proc/stat and returns the idle value for each cpu in a vector
vector<long long> get_idle() {

	//Virtual file, created by the Linux kernel on demand
	ifstream in( "/proc/stat" );

	vector<long long> result;

	//This might broke if there are not 8 columns in /proc/stat
	regex reg("cpu(\\d+) (\\d+) (\\d+) (\\d+) (\\d+) (\\d+) (\\d+) (\\d+) (\\d+) (\\d+) (\\d+)");

	string line;
	
	while ( getline(in, line) ) {
		smatch match;
		if ( regex_match( line, match, reg ) ) {
			long long idle_time = lexical_cast<long long>(match[5]);
			result.push_back( idle_time );
		}
	}
	return result;
}

//This function returns the avarege load in the next interval_milliseconds for each cpu in a vector
vector<float> get_load(unsigned interval_milliseconds) {
	double current_time_1 = tic();
	vector<long long> idle_time_1 = get_idle();

	usleep(interval_milliseconds*1000);

	double total_seconds_elpased = toc(current_time_1);
	vector<long long> idle_time_2 = get_idle();
	
	vector<float> cpu_loads;
	for ( unsigned i = 0; i < idle_time_1.size(); ++i ) {
		//This might get slightly negative, because our time measurment is not accurate
		float load = 100 - float(idle_time_2[i] - idle_time_1[i])/(total_seconds_elpased);
                if(load<0) load = 0;
		cpu_loads.push_back( load );

	}
	return cpu_loads;
}

//This function constructs a html page based on ASIN.
std::string get_html_page(std::string ASIN){  //ASIN looks like this: "'0000','1102','2203'"
    string page = "<table border=\"0\"  width=\"600\" cellpadding='20'>";
    sql::Statement *stmt = 0;
    sql::ResultSet *res = 0;
    string title, img_link, item_link, description, retailer;
    float price;
    static sql::Driver *driver = get_driver_instance();
    static char qs[5000];
    sql::Connection *conn;
    try {
        conn = driver->connect("tcp://localhost:3306", "root", "gatorsearch2014");
        stmt = conn->createStatement();
        stmt->execute("USE GatorDB");
        sprintf(qs,"SELECT Title, ImageLinks, ItemLink, Price, Description, Retailer FROM Item WHERE ASIN IN (%s)",ASIN.c_str());
        printf("query = %s\n",qs);
        res = stmt->executeQuery(qs);
		while(res->next()){
			title = res->getString(0);
			img_link = res->getString(1);
  			std::size_t found = img_link.find(' ');
  			if (found!=std::string::npos){ //there are more than one image links.
    			img_link = img_link.substr(0,found+1);
    		}
			item_link = res->getString(2);
			price = res->getDouble(3);
			description = res->getString(4);
			retailer = res->getString(5);
			printf("title: %s\n, price: %.2f\n, description: %s\n",title.c_str(),price,description.c_str());
			page += "<tr><td align=\"center\"><img src=\""+img_link+"\" style=\"height:100px;max-width:100px;\"></td><td><a href=\""+item_link+"\" target = \"_blank\">" + title + "</a></td></tr>";
		}
		delete stmt,res;
    } catch (sql::SQLException &e) {
        cout << "# ERR: SQLException in " << __FILE__;
        cout << "(" << __FUNCTION__ << ") on line " << __LINE__ << endl;
        cout << "# ERR: " << e.what();
        cout << " (MySQL error code: " << e.getErrorCode();
        cout << ", SQLState: " << e.getSQLState() << " )" << endl;
		exit(-1);
    }
    page += "</table>";
	return page;
}


