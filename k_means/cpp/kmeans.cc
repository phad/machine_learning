#include <algorithm>
#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdexcept>
#include <stdlib.h>
#include <sys/time.h>
#include <vector>

using namespace std;

class Value {
 public:
  Value() : _v1(0.0), _v2(0.0), _v3(0.0), _v4(0.0), _v5(0.0) {}
  Value(double v1, double v2, double v3, double v4, double v5)
    : _v1(v1), _v2(v2), _v3(v3), _v4(v4), _v5(v5) {}
  double distance_from(const Value& other) const {
    double dist_squared = 0.0;
    dist_squared += pow(other._v1 - _v1, 2.0);
    dist_squared += pow(other._v2 - _v2, 2.0);
    dist_squared += pow(other._v3 - _v3, 2.0);
    dist_squared += pow(other._v4 - _v4, 2.0);
    dist_squared += pow(other._v5 - _v5, 2.0);
    return sqrt(dist_squared);
  }
  bool equals(const Value& other) const {
    bool is_equal = true;
    is_equal &= (_v1 == other._v1);
    is_equal &= (_v2 == other._v2);
    is_equal &= (_v3 == other._v3);
    is_equal &= (_v4 == other._v4);
    is_equal &= (_v5 == other._v5);
    return is_equal;
  }
  void add(const Value& other) {
    _v1 += other._v1;
    _v2 += other._v2;
    _v3 += other._v3;
    _v4 += other._v4;
    _v5 += other._v5;
  }
  void divide_by(int divisor) {
    if (divisor == 0.0) {
      throw new invalid_argument("divide by zero");
    }
    _v1 /= divisor;
    _v2 /= divisor;
    _v3 /= divisor;
    _v4 /= divisor;
    _v5 /= divisor;
  }
  string output() const {
    ostringstream os;
    os << '[' << _v1 << ',' << _v2 << ',' << _v3 << ',' << _v4 << ','
       << _v5 << ']';
    return os.str();
  }

 private:
  double _v1;
  double _v2;
  double _v3;
  double _v4;
  double _v5;
};

class Cluster {
 public:
  explicit Cluster(const Value& initial_mean) {
    add_value(initial_mean);
    update_mean();
  }
  void add_value(const Value& val) {
    _values.push_back(&val);
  }
  const Value& mean() const {
    return _mean;
  }
  Value update_mean() {
    Value sum;
    for (vector<const Value*>::iterator it = _values.begin();
	 it != _values.end(); ++it) {
      sum.add(**it);
    }
    sum.divide_by(_values.size());
    _mean = sum;
    return _mean;
  }
  void reset_values() {
    _values.clear();
  }
  string output() const {
    ostringstream os;
    os << "Cluster(size:" << _values.size() << ", mean:" << _mean.output()
       << ')';
    return os.str();
  }

 private:
  Cluster(const Cluster& );
  void operator=(const Cluster& );

  vector<const Value*> _values;
  Value _mean;
};

class KMeansClustering {
 public:
  static KMeansClustering* new_clustering(int size, const vector<Value>& data_set) {
    KMeansClustering* self = new KMeansClustering(size, data_set);
    self->initialize();
    return self;
  }
  virtual ~KMeansClustering() {
    for (vector<Cluster*>::iterator cl = _clusters.begin();
	 cl != _clusters.end(); cl++) {
      delete *cl;
    }
  }
  bool perform_iteration() {
    // Step 0: empty each cluster of values, retaining previously-calculated mean.
    //         Record initial state of clusters (ie. their mean values) to check
    //         for convergence.
    vector<Value> initial_state;
    for (vector<Cluster*>::iterator cluster = _clusters.begin();
	 cluster != _clusters.end(); cluster++) {
      (*cluster)->reset_values();
      initial_state.push_back((*cluster)->mean());
    }
    // Step 1: for each value in the data set, identify the cluster whose mean
    //         is closest to the value.  Add the value to that cluster.
    for (vector<Value>::const_iterator value = _data_set.begin();
	 value != _data_set.end(); ++value) {
      Cluster* assigned_cluster = NULL;
      double shortest_distance = 0.0;
      for (vector<Cluster*>::iterator cluster = _clusters.begin();
	   cluster != _clusters.end(); cluster++) {
	double distance = (*value).distance_from((*cluster)->mean());
	if ((cluster == _clusters.begin()) || (distance < shortest_distance)) {
	  assigned_cluster = *cluster;
	  shortest_distance = distance;
	}
      }
      assigned_cluster->add_value(*value);
    }
    // Step 2: when all values are assigned, recalculate the mean for each
    //         cluster, and figure out if convergence occurs.
    bool converged = true;
    vector<Value>::const_iterator initial_mean = initial_state.begin();
    for (vector<Cluster*>::iterator cluster = _clusters.begin();
	 cluster != _clusters.end(); ++cluster, ++initial_mean) {
      Value updated_mean = (*cluster)->update_mean();
      converged &= ((*initial_mean).equals(updated_mean));
    }
    return converged;
  }
  string output() const {
    ostringstream os;
    os << "KMeansClustering(k:" << _clusters.size() << ", [";
    for (vector<Cluster*>::const_iterator it = _clusters.begin();
	 it != _clusters.end(); ++it) {
      os << (*it)->output();
      os << ',';
    }
    os << "])";
    return os.str();
  }

 private:
  KMeansClustering(int size, const vector<Value>& data_set)
    : _size(size), _data_set(data_set) {
  }
  KMeansClustering(const KMeansClustering& );
  void operator=(const KMeansClustering& );

  void initialize() {
    int remain = _size;
    vector<int> seed_indices;
    while (remain > 0) {
      int index = -1;
      do {
	index = rand() % _data_set.size();
      } while (find(seed_indices.begin(), seed_indices.end(), index)
	       != seed_indices.end());
      seed_indices.push_back(index);
      const Value& initial_mean = _data_set.at(index);
      Cluster* cl = new Cluster(initial_mean);
      _clusters.push_back(cl);
      --remain;
    }
  }

  int _size;
  vector<Cluster*> _clusters;
  const vector<Value>& _data_set;
};

int main(int argc, char** argv) {
  if (argc < 3) {
    cerr << "Usage: kmeans <k> <tuples_file>" << endl;
    return 1;
  }
  int k = atoi(argv[1]);
  cout << "Using k: " << k << endl;
  string data_filename(argv[2]);
  ifstream data_file(data_filename.c_str());
  if (data_file.fail()) {
    cerr << "Error opening file: " << data_filename << endl;
    return 2;
  }
  cout << "Reading data set from: " << data_filename << endl;
  stringstream ss;
  string line;
  vector<Value> values;
  string v1, v2;
  while (getline(data_file, line)) {
    if (line.length() == 0) {
      break;
    }
    ss.clear();
    ss.str(line);
    getline(ss, v1, ',');
    getline(ss, v2);
    values.push_back(Value(atof(v1.c_str()), atof(v2.c_str()), 0, 0, 0));
  }
  cout << "Done reading data set. Read " << values.size() << " values."  << endl;

  time_t random_seed = time(NULL);
  srand(random_seed);
  KMeansClustering* cl = KMeansClustering::new_clustering(k, values);
  cout << "Initial state (seed = " << random_seed << ")" << endl;
  cout << cl->output() << endl;
  int iter = 0;
  bool done = false;
  timeval start_time;
  gettimeofday(&start_time, NULL);
  do {
    ++iter;
    done = cl->perform_iteration();
    cout << "Iteration " << iter << endl;
    cout << cl->output() << endl;
  } while (!done);
  timeval end_time;
  gettimeofday(&end_time, NULL);
  suseconds_t elapsed_us = 1E6 * (end_time.tv_sec - start_time.tv_sec)
    + end_time.tv_usec - start_time.tv_usec;
  cout << "Elapsed microsec: " << elapsed_us << endl;
  delete cl;
  return 0;
}

