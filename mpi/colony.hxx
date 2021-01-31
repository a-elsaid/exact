#ifndef COLONY_HXX
#define COLONY_HXX
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>
#include "rnn/point.hxx"
#include "rnn/ant.hxx"
#include <chrono>
#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;
#include "rnn/cants_dbscan.hxx"

#include "mpi.h"
#include <mutex>
using std::mutex;

#include <iomanip>
using std::setprecision ;

#include "rnn/rnn_genome.hxx"
#include "rnn/delta_node.hxx"
#include "rnn/ugrnn_node.hxx"
#include "rnn/gru_node.hxx"
#include "rnn/mgu_node.hxx"
#include "rnn/lstm_node.hxx"
#include "rnn/rnn_edge.hxx"
#include "rnn/rnn_genome.hxx"
#include "rnn/rnn_node.hxx"
#include "rnn/rnn_node_interface.hxx"

#include "common/arguments.hxx"
#include "common/log.hxx"


#define NUMBER_OF_NODE_TYPES 5
#define EVAPORATION_RATE 0.05
#define POINT_DIEING_THRESHOLD 0.25
#define INITIAL_PHEROMONE    1.0
#define MAX_PHEROMONE    10.0

#define MINIMUM_POINTS 1     // minimum number of cluster
#define EPSILON (0.05*0.05)  // distance for clustering, metre^2

#define WORK_REQUEST_TAG 1
#define GENOME_LENGTH_TAG 2
#define GENOME_TAG 3
#define TERMINATE_TAG 4

struct rnn_segment_type {
    map < POINT*, map<int32_t, POINT*> > segments;
    // vector <POINT*> input_points ;
};

class COLONY {
    private:
        int32_t max_recurrent_depth;
        int32_t numberOfAnts ;
        vector < double > levels_pheromones ;           // lag levels pheromones
        vector < POINT* > inputs ;                      // Inputs pheromones
        vector<vector < POINT > > outputs ;             // Output <position, pheromone>
        map <int, map<int32_t, POINT*> > levels ;       // dimension pheromones
        minstd_rand0 generator;
        int types[5] = { 1, 2, 3, 4, 5 } ;
        void depositePheromone( map < POINT*, map<int32_t, POINT*> > &segments, RNN_Genome* genome) ;
        void evaporatePheromone () ;
        void writeColonyToFile (int number) ;

        mutex colony_mutex;

        int32_t population_size;
        int32_t number_islands;
        vector<RNN_Genome*> population;
        int32_t max_genomes;
        int32_t generated_genomes;
        int32_t inserted_genomes;
        int32_t total_bp_epochs;

        int32_t hidden_layers_depth;
        int32_t hidden_layer_nodes;

        int32_t edge_innovation_count;
        int32_t node_innovation_count;

        map<string, int32_t> inserted_from_map;
        map<string, int32_t> generated_from_map;

        int32_t number_inputs;
        int32_t number_outputs;
        int32_t bp_iterations;
        double learning_rate;

        bool use_high_threshold;
        double high_threshold;

        bool use_low_threshold;
        double low_threshold;

        bool use_dropout;
        double dropout_probability;

        vector<string> output_parameter_names;
        map<string,double> normalize_mins;
        map<string,double> normalize_maxs;
        map<int32_t, int32_t> node_types;

        string output_directory;
        ofstream *log_file;
        ofstream *op_log_file;
        vector<string> op_log_ordering;
        int32_t generated_counts;
        int32_t inserted_counts;

        string normalize_type ;
        const map<string,double> normalize_avgs ;
        const map<string,double> normalize_std_devs ;

        ostringstream memory_log;

        int edge_counter = 0 ;
        int rec_edge_counter =0 ;
        void copy_point(POINT* org, POINT* trg) ;

        bool verbose ;
        bool log_ants_paths ;

        std::chrono::time_point<std::chrono::system_clock> startClock;

    public:
        vector<string> input_parameter_names;

        COLONY(int32_t _numberOfAnts,
            int32_t _population_size,
            int32_t _max_genomes,
            const vector<string> &_input_parameter_names,
            const vector<string> &_output_parameter_names,
            string _normalize_type,
            const map<string,double> &_normalize_mins,
            const map<string,double> &_normalize_maxs,
            const map<string,double> &_normalize_avgs,
            const map<string,double> &_normalize_std_devs,
            int32_t _bp_iterations,
            double _learning_rate,
            bool _use_high_threshold,
            double _high_threshold,
            bool _use_low_threshold,
            double _low_threshold,
            string _output_directory,
            int32_t _hidden_layers_depth,
            int32_t _max_recurrent_depth,
            int32_t _hidden_layer_nodes,
            bool _verbose,
            bool _log_ants_paths) ;
        vector < ANT > ants ;
        int32_t getNumberOfAnts () ;
        rnn_segment_type createSegments( map<int, map<int, POINT*> > centeroids ) ;
        RNN_Genome* generateRNN( ) ;
        void startLiving( int max_rank ) ;
        map <int32_t, rnn_segment_type> rnns_blueprints ;
        RNN_Node* buildNode(vector<RNN_Node_Interface*> &rnn_nodes, POINT* point, vector <double> &nodes_parameters ) ;
        RNN_Genome* buildGenome(vector<POINT*> picked_input_points) ;

        void send_work_request(int target) ;
        void receive_work_request(int source) ;
        RNN_Genome* receive_genome_from(int source) ;
        void send_genome_to(int target, RNN_Genome* genome) ;
        void send_terminate_message(int target) ;
        void receive_terminate_message(int source) ;
        RNN_Genome* get_best_genome() ;
        void print_population() ;
        int32_t population_contains(RNN_Genome* genome);
        bool populations_full() const;
        bool insert_genome(RNN_Genome* genome) ;

        void write_to_file(string bin_filename, bool verbose) ;
        void write_to_stream(ostream &bin_ostream, bool verbose) ;

        void worker(int rank) ;
        void set_possible_node_types(vector<string> possible_node_type_strings) ;

        vector< vector< vector<double> > > training_inputs;
        vector< vector< vector<double> > > training_outputs;
        vector< vector< vector<double> > > validation_inputs;
        vector< vector< vector<double> > > validation_outputs;
        vector<string> arguments;
        vector<int> possible_node_types;

    ~COLONY();
};




#endif
