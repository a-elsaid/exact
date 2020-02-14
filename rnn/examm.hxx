#ifndef EXAMM_HXX
#define EXAMM_HXX

#include <fstream>
using std::ofstream;

#include <map>
using std::map;

#include <sstream>
using std::ostringstream;

#include <string>
using std::string;
using std::to_string;

#include <vector>
using std::vector;

#include "rnn_genome.hxx"
#include "speciation_strategy.hxx"

#define GLOBAL_POPULATION 0
#define ISLAND_POPULATION 1

#define UNIFORM_DISTRIBUTION 0
#define HISTOGRAM_DISTRIBUTION 1
#define NORMAL_DISTRIBUTION 2

#define N_STIR_MUTATIONS_DEFAULT 32

class EXAMM {
    private:
        int32_t population_size;
        int32_t number_islands;

        vector< vector<RNN_Genome*> > genomes;

        int32_t max_genomes;
        int32_t total_bp_epochs;

        string speciation_method;
        SpeciationStrategy *speciation_strategy;

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

        minstd_rand0 generator;
        uniform_real_distribution<double> rng_0_1;
        uniform_real_distribution<double> rng_crossover_weight;

        int32_t min_recurrent_depth;
        int32_t max_recurrent_depth;

        bool epigenetic_weights;

        double more_fit_crossover_rate;
        double less_fit_crossover_rate;

        double clone_rate;

        double add_edge_rate;
        double add_recurrent_edge_rate;
        double enable_edge_rate;
        double disable_edge_rate;
        double split_edge_rate;

        double add_node_rate;
        double enable_node_rate;
        double disable_node_rate;
        double split_node_rate;
        double merge_node_rate;

        vector<int> possible_node_types;


        string output_directory;
        ofstream *log_file;

        vector<string> input_parameter_names;
        vector<string> output_parameter_names;

        map<string,double> normalize_mins;
        map<string,double> normalize_maxs;

        ostringstream memory_log;

        std::chrono::time_point<std::chrono::system_clock> startClock;

        int32_t rec_sampling_population;
        int32_t rec_sampling_distribution;

        string  genome_file_name ;
        int     no_extra_inputs ;
        int     no_extra_outputs ;

        int     no_stir_mutations = N_STIR_MUTATIONS;

        vector<string> inputs_to_remove ;
        vector<string> outputs_to_remove ;

        bool tl_ver1;
        bool tl_ver2;
        bool tl_ver3;

    public:
        EXAMM(int32_t _population_size, int32_t _number_islands, int32_t _max_genomes, int32_t _num_genomes_check_on_island, string _speciation_method,
            const vector<string> &_input_parameter_names,
            const vector<string> &_output_parameter_names,
            const map<string,double> &_normalize_mins,
            const map<string,double> &_normalize_maxs,
            int32_t _bp_iterations, double _learning_rate,
            bool _use_high_threshold, double _high_threshold,
            bool _use_low_threshold, double _low_threshold,
            bool _use_dropout, double _dropout_probability,
            int32_t _min_recurrent_depth, int32_t _max_recurrent_depth,
            string _rec_sampling_population, string _rec_sampling_distribution, string _output_directory,
            string _genome_file_name,
            int _no_extra_inputs, int _no_extra_outputs,
            vector<string> &_inputs_to_remove, vector<string> &_outputs_to_remove,
            bool _tl_ver1, bool _tl_ver2, bool _tl_ver3 );

        ~EXAMM();

        void print();
        void update_log();
        void write_memory_log(string filename);

        void set_possible_node_types(vector<string> possible_node_type_strings);

        Distribution *get_recurrent_depth_dist(int32_t island);

        int get_random_node_type();

        RNN_Genome* generate_genome();
        bool insert_genome(RNN_Genome* genome);

        void mutate(int32_t max_mutations, RNN_Genome *p1);

        void attempt_node_insert(vector<RNN_Node_Interface*> &child_nodes, const RNN_Node_Interface *node, const vector<double> &new_weights);
        void attempt_edge_insert(vector<RNN_Edge*> &child_edges, vector<RNN_Node_Interface*> &child_nodes, RNN_Edge *edge, RNN_Edge *second_edge, bool set_enabled);
        void attempt_recurrent_edge_insert(vector<RNN_Recurrent_Edge*> &child_recurrent_edges, vector<RNN_Node_Interface*> &child_nodes, RNN_Recurrent_Edge *recurrent_edge, RNN_Recurrent_Edge *second_edge, bool set_enabled);
        RNN_Genome* crossover(RNN_Genome *p1, RNN_Genome *p2);

        double get_best_fitness();
        double get_worst_fitness();
        RNN_Genome* get_best_genome();
        RNN_Genome* get_worst_genome();

        string get_output_directory() const;
        RNN_Genome* generate_for_transfer_learning(string file_name, int extra_inputs, int extra_outputs) ;
};

#endif
