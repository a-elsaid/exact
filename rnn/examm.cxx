#include <algorithm>
using std::sort;

#include <chrono>
#include <cstring>

#include <functional>
using std::bind;
using std::function;

#include <fstream>
using std::ofstream;

#include <iomanip>
using std::setw;
using std::setprecision;

#include <iostream>
using std::endl;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;
using std::uniform_int_distribution;

#include <string>
using std::string;
using std::to_string;

#include "examm.hxx"
#include "rnn_genome.hxx"
#include "generate_nn.hxx"
#include "speciation_strategy.hxx"
#include "island_speciation_strategy.hxx"
#include "beta_thompson_sampling.hxx"

#include "common/files.hxx"
#include "common/log.hxx"



EXAMM::~EXAMM() {
    RNN_Genome *genome;
    for (uint32_t i = 0; i < genomes.size(); i++) {
        while (genomes[i].size() > 0) {
            genome = genomes[i].back();
            genomes[i].pop_back();
            delete genome;
        }
    }
}

EXAMM::EXAMM(
        int32_t _population_size, 
        int32_t _number_islands, 
        int32_t _max_genomes, 
        int32_t _extinction_event_generation_number,
        int32_t islands_to_exterminate,
        string _island_ranking_method, 
        string _repopulation_method, 
        int32_t _repopulation_mutations,
        string _speciation_method, 
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
        bool _use_dropout, 
        double _dropout_probability,
        int32_t _min_recurrent_depth,
        int32_t _max_recurrent_depth,
        vector<string> possible_node_type_strings,
        string _output_directory,
        RNN_Genome *seed_genome,
        bool _start_filled,
        double _node_type_sampling_decay_rate,
        bool _use_mutation_thompson_sampling,
        double _mutation_sampling_decay_rate,
        bool _use_node_type_thompson_sampling,
        int32_t _max_number_mutations,
        double _number_mutation_sampling_decay_rate,
        bool _use_number_mutations_sampling) :
                        population_size(_population_size),
                        number_islands(_number_islands),
                        max_genomes(_max_genomes),
                        extinction_event_generation_number(_extinction_event_generation_number),
                        island_ranking_method(_island_ranking_method), 
                        speciation_method(_speciation_method),
                        repopulation_method(_repopulation_method), 
                        repopulation_mutations(_repopulation_mutations),
                        number_inputs(_input_parameter_names.size()),
                        number_outputs(_output_parameter_names.size()),
                        bp_iterations(_bp_iterations),
                        learning_rate(_learning_rate),
                        use_high_threshold(_use_high_threshold),
                        high_threshold(_high_threshold),
                        use_low_threshold(_use_low_threshold),
                        low_threshold(_low_threshold),
                        use_dropout(_use_dropout),
                        dropout_probability(_dropout_probability),
                        output_directory(_output_directory),
                        normalize_type(_normalize_type),
                        start_filled(_start_filled) {

    input_parameter_names = _input_parameter_names;
    output_parameter_names = _output_parameter_names;
    normalize_mins = _normalize_mins;
    normalize_maxs = _normalize_maxs;
    normalize_avgs = _normalize_avgs;
    normalize_std_devs = _normalize_std_devs;

    total_bp_epochs = 0;

    edge_innovation_count = 0;
    node_innovation_count = 0;

    //update to now have islands of genomes
    genomes = vector< vector<RNN_Genome*> >(number_islands);

    uint16_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator = minstd_rand0(seed);
    rng_0_1 = uniform_real_distribution<double>(0.0, 1.0);

    //rng_crossover_weight = uniform_real_distribution<double>(0.0, 0.0);
    //rng_crossover_weight = uniform_real_distribution<double>(-0.10, 0.1);
    rng_crossover_weight = uniform_real_distribution<double>(-0.5, 1.5);
    //rng_crossover_weight = uniform_real_distribution<double>(0.45, 0.55);

    min_recurrent_depth = _min_recurrent_depth;
    max_recurrent_depth = _max_recurrent_depth;

    epigenetic_weights = true;

    more_fit_crossover_rate = 1.00;
    less_fit_crossover_rate = 0.50;
    //more_fit_crossover_rate = 0.75;
    //less_fit_crossover_rate = 0.25;

    clone_rate = 1.0;

    add_edge_rate = 1.0;
    //add_recurrent_edge_rate = 3.0;
    add_recurrent_edge_rate = 1.0;
    enable_edge_rate = 1.0;
    //disable_edge_rate = 3.0;
    disable_edge_rate = 1.0;
    split_edge_rate = 1.0;
    //split_edge_rate = 0.0;

    possible_node_types.clear();
    possible_node_types.push_back(SIMPLE_NODE);
    possible_node_types.push_back(JORDAN_NODE);
    possible_node_types.push_back(ELMAN_NODE);
    possible_node_types.push_back(UGRNN_NODE);
    possible_node_types.push_back(MGU_NODE);
    possible_node_types.push_back(GRU_NODE);
    possible_node_types.push_back(LSTM_NODE);
    possible_node_types.push_back(DELTA_NODE);

    bool node_ops = true;
    if (node_ops) {
        add_node_rate = 1.0;
        enable_node_rate = 1.0;
        //disable_node_rate = 3.0;
        disable_node_rate = 1.0;
        split_node_rate = 1.0;
        merge_node_rate = 1.0;

    } else {
        add_node_rate = 0.0;
        enable_node_rate = 0.0;
        disable_node_rate = 0.0;
        split_node_rate = 0.0;
        merge_node_rate = 0.0;
    }

    Log::info("Speciation method is: \"%s\" (Default is the island-based speciation strategy).\n", speciation_method.c_str());
    if (speciation_method.compare("island") == 0 || speciation_method.compare("") == 0) {
        //generate a minimal feed foward network as the seed genome
        
        bool seed_genome_was_minimal = false;
        if (seed_genome == NULL) {
            seed_genome_was_minimal = true;
            seed_genome = create_ff(input_parameter_names, 0, 0, output_parameter_names, 0);
            seed_genome->initialize_randomly();
        } //otherwise the seed genome was passed into EXAMM

        //make sure we don't duplicate node or edge innovation numbers
        edge_innovation_count = seed_genome->get_max_edge_innovation_count() + 1;
        node_innovation_count = seed_genome->get_max_node_innovation_count() + 1;

        seed_genome->set_generated_by("initial");

        //insert a copy of it into the population so
        //additional requests can mutate it

        seed_genome->best_validation_mse = EXAMM_MAX_DOUBLE;
        seed_genome->best_validation_mae = EXAMM_MAX_DOUBLE;
        //seed_genome->best_parameters.clear();
        
        double mutation_rate = 0.70, intra_island_co_rate = 0.20, inter_island_co_rate = 0.10;
        
        if (number_islands == 1) {
            inter_island_co_rate = 0.0;
            intra_island_co_rate = 0.30;
        }
       
        // Only difference here is that the apply_stir_mutations lambda is passed if the island is supposed to start filled.
        if (start_filled) {
            // Only used if start_filled is enabled
            function<void (RNN_Genome *)> apply_stir_mutations = [this](RNN_Genome *genome) {
                RNN_Genome *copy = genome->copy();
                this->mutate(repopulation_mutations, copy);
                return copy;
            };

            speciation_strategy = new IslandSpeciationStrategy(
                    number_islands, population_size, mutation_rate, intra_island_co_rate, inter_island_co_rate,
                    seed_genome, island_ranking_method, repopulation_method, extinction_event_generation_number, repopulation_mutations, islands_to_exterminate, seed_genome_was_minimal, apply_stir_mutations);
        } else {
            speciation_strategy = new IslandSpeciationStrategy(
                    number_islands, population_size, mutation_rate, intra_island_co_rate, inter_island_co_rate,
                    seed_genome, island_ranking_method, repopulation_method, extinction_event_generation_number, repopulation_mutations, islands_to_exterminate, seed_genome_was_minimal);
        }
    }

    // The following section just generates names for logging information about mutation success rates:
    op_log_ordering = {
        "genomes",
        "crossover",
        "island_crossover",
        "clone",
        "add_edge",
        "add_recurrent_edge",
        "enable_edge",
        "disable_edge",
        "enable_node",
        "disable_node",
    };

    // To get data about these ops without respect to node type,
    // you'll have to calculate the sum, e.g. sum split_node(x) for all node types x
    // to get information about split_node as a whole.
    vector<string> ops_with_node_type = {
        "add_node",
        "split_node",
        "merge_node",
        "split_edge"
    };
	
    printf("1\n");
    for (int i = 0; i < ops_with_node_type.size(); i++) {
        string op = ops_with_node_type[i];
        for (int j = 0; j < possible_node_types.size(); j++) {
            string s = op + "(" + NODE_TYPES[possible_node_types[j]] + ")";
            op_log_ordering.push_back(s);
        }
    }

    printf("2\n");
    // Set the possible node types here so we can create our ThompsonSampling with the appropriate number of actions
    if (possible_node_type_strings.size() > 0) this->set_possible_node_types(possible_node_type_strings);
    
    for (int i = 0; i < ops_with_node_type.size(); i++) {
        string op = ops_with_node_type[i];
        for (int j = 0; j < possible_node_types.size(); j++) {
            string s = op + "(" + NODE_TYPES[possible_node_types[j]] + ")";
            mutation_string_to_possible_node_ty_index[s] = j;
        }
    }

    printf("3\n");
    // Make the ThompsanSamplig if need be
    if (_use_node_type_thompson_sampling) {
        node_type_selector = new BetaThompsonSampling(possible_node_types.size(), _node_type_sampling_decay_rate);
    }

    if (_use_number_mutations_sampling) {
        max_number_mutations = _max_number_mutations;
        number_mutations_selector = new BetaThompsonSampling(max_number_mutations, _number_mutation_sampling_decay_rate);
    }

    // This section fills up the 'mutation_string_to_mutation_index' which maps
    // all possible generated by strings that represent a mutation to the index in the thompson_sampling (i.e. the action which is an integer).
    // Node that mutations that require a node type e.g. add_node(lstm) and add_node(delta) will both map to the same action

    vector<string> mutations_without_node_type = {
        "clone",
        "add_edge",
        "add_recurrent_edge",
        "enable_edge",
        "disable_edge",
        "enable_node",
        "disable_node"
    };

    int mutation_actions = 0;

    for (int i = 0; i < mutations_without_node_type.size(); i++) {
        string mut = mutations_without_node_type[i];
        mutation_string_to_mutation_index[mut] = mutation_actions;
        
        possible_mutations.push_back(mut);
        mutation_actions += 1;
    }

    for (int i = 0; i < ops_with_node_type.size(); i++) {
        string op = ops_with_node_type[i];
        for (int j = 0; j < possible_node_types.size(); j++) {
            string s = op + "(" + NODE_TYPES[possible_node_types[j]] + ")";
            mutation_string_to_mutation_index[s] = mutation_actions;
        }

        possible_mutations.push_back(op);
        mutation_actions += 1;
    }

    if (_use_mutation_thompson_sampling) {
        mutation_selector = new BetaThompsonSampling(possible_mutations.size(), _mutation_sampling_decay_rate);
    }
    
    printf("4\n");
    if (output_directory != "") {
        mkpath(output_directory.c_str(), 0777);
        log_file = new ofstream(output_directory + "/" + "fitness_log.csv");
        (*log_file) << "Inserted Genomes, Total BP Epochs, Time, Best Val. MAE, Best Val. MSE, Enabled Nodes, Enabled Edges, Enabled Rec. Edges";
        memory_log << "Inserted Genomes, Total BP Epochs, Time, Best Val. MAE, Best Val. MSE, Enabled Nodes, Enabled Edges, Enabled Rec. Edges";

        (*log_file) << speciation_strategy->get_strategy_information_headers();
        (memory_log) << speciation_strategy->get_strategy_information_headers();

        (*log_file) << endl;
        memory_log << endl;

        op_log_file = new ofstream(output_directory + "/op_log.csv");

        for (int i = 0; i < op_log_ordering.size(); i++) {
            string op = op_log_ordering[i];
            (*op_log_file) << op;
            (*op_log_file) << " Generated, ";
            (*op_log_file) << op;
            (*op_log_file) << " Inserted, ";
            
            inserted_counts[op] = 0;
            generated_counts[op] = 0;
        }

        (*op_log_file) << endl;

        thompson_mutation_log_file = new ofstream(output_directory + "/thompson_mutation_log.csv");
        for (int i = 0; i < max_number_mutations; i++) {
            (*thompson_mutation_log_file) << (i + 1) << " Mutations Alpha, ";
            (*thompson_mutation_log_file) << (i + 1) << " Mutations Beta, ";
        }
    } else {
        log_file = NULL;
        op_log_file = NULL;
    }

    printf("5\n");
    startClock = std::chrono::system_clock::now();
}

void EXAMM::print() {
    if (Log::at_level(Log::INFO)) {
        speciation_strategy->print();
    }
}

void EXAMM::update_log() {
    if (log_file != NULL) {

        //make sure the log file is still good
        if (!log_file->good()) {
            log_file->close();
            delete log_file;

            string output_file = output_directory + "/fitness_log.csv";
            log_file = new ofstream(output_file, std::ios_base::app);

            if (!log_file->is_open()) {
                Log::error("could not open EXAMM output log: '%s'\n", output_file.c_str());
                exit(1);
            }
        }

        if (!op_log_file->good()) {
            op_log_file->close();
            delete op_log_file;

            string output_file = output_directory + "/op_log.csv";
            op_log_file = new ofstream(output_file, std::ios_base::app);

            if (!op_log_file->is_open()) {
                Log::error("could not open EXAMM output log: '%s'\n", output_file.c_str());
                exit(1);
            }
        }

        if (!thompson_mutation_log_file->good()) {
            op_log_file->close();
            delete op_log_file;

            string output_file = output_directory + "/thompson_mutation_log.csv";
            op_log_file = new ofstream(output_file, std::ios_base::app);

            if (!op_log_file->is_open()) {
                Log::error("could not open EXAMM output log: '%s'\n", output_file.c_str());
                exit(1);
            }
        }

        RNN_Genome *best_genome = get_best_genome();
        if (best_genome == NULL){
            best_genome = speciation_strategy->get_global_best_genome();
        }

        std::chrono::time_point<std::chrono::system_clock> currentClock = std::chrono::system_clock::now();
        long milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(currentClock - startClock).count();

        (*log_file) << speciation_strategy->get_inserted_genomes()
            << "," << total_bp_epochs
            << "," << milliseconds
            << "," << best_genome->best_validation_mae
            << "," << best_genome->best_validation_mse
            << "," << best_genome->get_enabled_node_count()
            << "," << best_genome->get_enabled_edge_count()
            << "," << best_genome->get_enabled_recurrent_edge_count()
            << speciation_strategy->get_strategy_information_values()
            << endl;

        memory_log << speciation_strategy->get_inserted_genomes()
            << "," << total_bp_epochs
            << "," << milliseconds
            << "," << best_genome->best_validation_mae
            << "," << best_genome->best_validation_mse
            << "," << best_genome->get_enabled_node_count()
            << "," << best_genome->get_enabled_edge_count()
            << "," << best_genome->get_enabled_recurrent_edge_count()
            << speciation_strategy->get_strategy_information_values()
            << endl;
        
        for (int i = 0; i < op_log_ordering.size(); i++) {
            string op = op_log_ordering[i];
            (*op_log_file) << generated_counts[op] << ", " << inserted_counts[op]  << ", ";
        }

        (*op_log_file) << endl;
        
        for (int i = 0; i < max_number_mutations; i++) {
            double alpha = number_mutations_selector->alphas[i];
            double beta = number_mutations_selector->betas[i];

            (*thompson_mutation_log_file) << alpha << ", " << beta << ", ";
        }

        (*thompson_mutation_log_file) << endl;
    }
}

void EXAMM::write_memory_log(string filename) {
    ofstream log_file(filename);
    log_file << memory_log.str();
    log_file.close();
}

void EXAMM::set_possible_node_types(vector<string> possible_node_type_strings) {
    possible_node_types.clear();

    for (uint32_t i = 0; i < possible_node_type_strings.size(); i++) {
        string node_type_s = possible_node_type_strings[i];

        bool found = false;

        for (int32_t j = 0; j < NUMBER_NODE_TYPES; j++) {
            if (NODE_TYPES[j].compare(node_type_s) == 0) {
                found = true;
                possible_node_types.push_back(j);
            }
        }

        if (!found) {
            Log::error("unknown node type: '%s'\n", node_type_s.c_str());
            exit(1);
        }
    }
}

string EXAMM::get_output_directory() const {
    return output_directory;
}

double EXAMM::get_best_fitness() {
    return speciation_strategy->get_best_fitness();
}

double EXAMM::get_worst_fitness() {
    return speciation_strategy->get_worst_fitness();
}

RNN_Genome* EXAMM::get_best_genome() {
    return speciation_strategy->get_best_genome();
}

RNN_Genome* EXAMM::get_worst_genome() {
    return speciation_strategy->get_worst_genome();
}

//this will insert a COPY, original needs to be deleted
bool EXAMM::insert_genome(RNN_Genome* genome) {
    total_bp_epochs += genome->get_bp_iterations();

    // Log::info("genomes evaluated: %10d , attempting to insert: %s\n", (speciation_strategy->get_inserted_genomes() + 1), parse_fitness(genome->get_fitness()).c_str());

    if (!genome->sanity_check()) {
        Log::error("genome failed sanity check on insert!\n");
        exit(1);
    }

    //updates EXAMM's mapping of which genomes have been generated by what
    genome->update_generation_map(generated_from_map);

    int32_t insert_position = speciation_strategy->insert_genome(genome);
    //write this genome to disk if it was a new best found genome
    if (insert_position == 0) {
        genome->write_graphviz(output_directory + "/rnn_genome_" + to_string(genome->get_generation_id()) + ".gv");
        genome->write_to_file(output_directory + "/rnn_genome_" + to_string(genome->get_generation_id()) + ".bin");
    }

    // Name of the operator
    const map<string, int> *generated_by_map = genome->get_generated_by_map();
    int32_t number_mutations = genome->get_number_mutations_generated_by();

    double worst_fitness = get_worst_fitness();
    double best_fitness = get_best_fitness();
    double fitness = genome->get_fitness();

    printf("worst: %f, best: %f, fitness: %f\n", worst_fitness, best_fitness, fitness);

    double reward = 1 - ((fitness - best_fitness) / (worst_fitness - best_fitness));

    for (auto it = generated_by_map->begin(); it != generated_by_map->end(); it++) {
        string generated_by = it->first;
        if (generated_counts.count(generated_by) > 0) {
            generated_counts["genomes"] += 1;

            // Add one to the number of genomes generated by this operator
            generated_counts[generated_by] += 1;

            // If it was inserted add one to the number of genomes generated AND inserted by this operator
            if (insert_position >= 0) {
                inserted_counts["genomes"] += 1;
                inserted_counts[generated_by] += 1;
            }

            // if we are using a ThompsonSampling for the node types and this mutation
            // used a node type we need to update the ThompsonSampling, 
            // and the islands are full so rejecting a genome is a possibility
            if (node_type_selector != NULL &&
                speciation_strategy->get_generated_genomes() >= number_islands * population_size * 4 &&
                mutation_string_to_possible_node_ty_index.count(generated_by) > 0) {
                int32_t index = mutation_string_to_possible_node_ty_index[generated_by];
                // printf("Generated %d\n", speciation_strategy->get_generated_genomes()); 
                // double reward = 1.0 - ((double) insert_position / (double) population_size) - 0.5;

                if (insert_position < 0)
                    reward = 0;
                if (insert_position == 0)
                    reward = 1;

                // For now the reward is always 1.0, but in the future the reward can be increased for
                // a better increase in performance.
                node_type_selector->update(index, reward);
                node_type_selector->print(this->possible_node_types, NODE_TYPES);
            }
        } else {
            if (generated_by != "initial")
                Log::error("unrecognized generated_by string '%s'\n", generated_by.c_str());
        }
    }
    
    printf("reward = %f\n", reward);

    if (number_mutations_selector != NULL && number_mutations > 0 && speciation_strategy->get_generated_genomes() >= 4000)
        number_mutations_selector->update(number_mutations - 1, reward);

    speciation_strategy->print();
    update_log();

    return insert_position >= 0;
}

RNN_Genome* EXAMM::generate_genome() {
    if (speciation_strategy->get_inserted_genomes() > max_genomes) return NULL;

    function<void (int32_t, RNN_Genome*)> mutate_function =
        [=](int32_t max_mutations, RNN_Genome *genome) {
            int32_t n_mutations = this->get_random_number_mutations();
            this->mutate(n_mutations, genome);
        };

    function<RNN_Genome* (RNN_Genome*, RNN_Genome*)> crossover_function =
        [=](RNN_Genome *parent1, RNN_Genome *parent2) {
            return this->crossover(parent1, parent2);
        };

    RNN_Genome *genome = speciation_strategy->generate_genome(rng_0_1, generator, mutate_function, crossover_function);

    genome->set_parameter_names(input_parameter_names, output_parameter_names);
    genome->set_normalize_bounds(normalize_type, normalize_mins, normalize_maxs, normalize_avgs, normalize_std_devs);
    genome->set_bp_iterations(bp_iterations);
    genome->set_learning_rate(learning_rate);

    if (use_high_threshold) genome->enable_high_threshold(high_threshold);
    if (use_low_threshold) genome->enable_low_threshold(low_threshold);
    if (use_dropout) genome->enable_dropout(dropout_probability);

    if (!epigenetic_weights) genome->initialize_randomly();

    //this is just a sanity check, can most likely comment out (checking to see
    //if all the paramemters are sane)
    Log::debug("getting mu/sigma after random initialization of copy!\n");
    double _mu, _sigma;
    genome->get_mu_sigma(genome->best_parameters, _mu, _sigma);

    return genome;
}

int EXAMM::get_random_node_type() {
    int32_t node_type_index;

    if (node_type_selector == NULL || speciation_strategy->get_inserted_genomes() < 2000)
        node_type_index = rng_0_1(generator) * possible_node_types.size();
    else
        node_type_index = node_type_selector->sample_action(generator);

    return possible_node_types[node_type_index];
}

int32_t EXAMM::get_random_number_mutations() {
    int32_t number_mutations;

    if (number_mutations_selector == NULL || speciation_strategy->get_generated_genomes() < 4000)
        number_mutations = 1;
    else
        number_mutations = number_mutations_selector->sample_action(generator) + 1;

    return number_mutations;
}

void EXAMM::mutate(int32_t max_mutations, RNN_Genome *g) {
    double total = clone_rate + add_edge_rate + add_recurrent_edge_rate + enable_edge_rate + disable_edge_rate + split_edge_rate + add_node_rate + enable_node_rate + disable_node_rate + split_node_rate + merge_node_rate;

    bool modified = false;

    double mu, sigma;

    //g->write_graphviz("rnn_genome_premutate_" + to_string(g->get_generation_id()) + ".gv");
    Log::debug("generating new genome by mutation.\n");
    g->get_mu_sigma(g->best_parameters, mu, sigma);
    g->clear_generated_by();

    //the the weights in the genome to it's best parameters
    //for epigenetic iniitalization
    if (g->best_parameters.size() == 0) {
        g->set_weights(g->initial_parameters);
        g->get_mu_sigma(g->initial_parameters, mu, sigma);
    } else {
        g->set_weights(g->best_parameters);
        g->get_mu_sigma(g->best_parameters, mu, sigma);
    }

    int number_mutations = 0;
    
    // Not using thompson sampling

    for (;;) {
        if (modified) {
            modified = false;
            number_mutations++;
        }
        if (number_mutations >= max_mutations) break;

        g->assign_reachability();
        
        double rng = rng_0_1(generator) * total;
        
        int new_node_type = get_random_node_type();
        string node_type_str = NODE_TYPES[new_node_type];
        
        Log::debug( "rng: %lf, total: %lf, new node type: %d (%s)\n", rng, total, new_node_type, node_type_str.c_str());

        if (mutation_selector == NULL) {
            if (rng < clone_rate) {
                Log::debug("\tcloned\n");
                g->set_generated_by("clone");
                modified = true;
                continue;
            }
            rng -= clone_rate;
            if (rng < add_edge_rate) {
                modified = g->add_edge(mu, sigma, edge_innovation_count);
                Log::debug("\tadding edge, modified: %d\n", modified);
                if (modified) g->set_generated_by("add_edge");
                continue;
            }
            rng -= add_edge_rate;

            if (rng < add_recurrent_edge_rate) {
                uniform_int_distribution<int32_t> dist = get_recurrent_depth_dist();
                modified = g->add_recurrent_edge(mu, sigma, dist, edge_innovation_count);
                Log::debug("\tadding recurrent edge, modified: %d\n", modified);
                if (modified) g->set_generated_by("add_recurrent_edge");
                continue;
            }
            rng -= add_recurrent_edge_rate;

            if (rng < enable_edge_rate) {
                modified = g->enable_edge();
                Log::debug("\tenabling edge, modified: %d\n", modified);
                if (modified) g->set_generated_by("enable_edge");
                continue;
            }
            rng -= enable_edge_rate;

            if (rng < disable_edge_rate) {
                modified = g->disable_edge();
                Log::debug("\tdisabling edge, modified: %d\n", modified);
                if (modified) g->set_generated_by("disable_edge");
                continue;
            }
            rng -= disable_edge_rate;

            if (rng < split_edge_rate) {
                uniform_int_distribution<int32_t> dist = get_recurrent_depth_dist();
                modified = g->split_edge(mu, sigma, new_node_type, dist, edge_innovation_count, node_innovation_count);
                Log::debug("\tsplitting edge, modified: %d\n", modified);
                if (modified) g->set_generated_by("split_edge(" + node_type_str + ")");
                continue;
            }
            rng -= split_edge_rate;

            if (rng < add_node_rate) {
                uniform_int_distribution<int32_t> dist = get_recurrent_depth_dist();
                modified = g->add_node(mu, sigma, new_node_type, dist, edge_innovation_count, node_innovation_count);
                Log::debug("\tadding node, modified: %d\n", modified);
                if (modified) g->set_generated_by("add_node(" + node_type_str + ")");
                continue;
            }
            rng -= add_node_rate;

            if (rng < enable_node_rate) {
                modified = g->enable_node();
                Log::debug("\tenabling node, modified: %d\n", modified);
                if (modified) g->set_generated_by("enable_node");
                continue;
            }
            rng -= enable_node_rate;

            if (rng < disable_node_rate) {
                modified = g->disable_node();
                Log::debug("\tdisabling node, modified: %d\n", modified);
                if (modified) g->set_generated_by("disable_node");
                continue;
            }
            rng -= disable_node_rate;

            if (rng < split_node_rate) {
                uniform_int_distribution<int32_t> dist = get_recurrent_depth_dist();
                modified = g->split_node(mu, sigma, new_node_type, dist, edge_innovation_count, node_innovation_count);
                Log::debug("\tsplitting node, modified: %d\n", modified);
                if (modified) g->set_generated_by("split_node(" + node_type_str + ")");
                continue;
            }
            rng -= split_node_rate;

            if (rng < merge_node_rate) {
                uniform_int_distribution<int32_t> dist = get_recurrent_depth_dist();
                modified = g->merge_node(mu, sigma, new_node_type, dist, edge_innovation_count, node_innovation_count);
                Log::debug("\tmerging node, modified: %d\n", modified);
                if (modified) g->set_generated_by("merge_node(" + node_type_str + ")");
                continue;
            }
            rng -= merge_node_rate;
        } else {
            int mutation = mutation_selector->sample_action(generator);
            
            string selected_mutation = possible_mutations[mutation];

            if (selected_mutation == "clone") {
                Log::debug("\tcloned\n");
                g->set_generated_by("clone");
                modified = true;
                continue;
            }

            if (selected_mutation == "add_edge") {
                modified = g->add_edge(mu, sigma, edge_innovation_count);
                Log::debug("\tadding edge, modified: %d\n", modified);
                if (modified) g->set_generated_by("add_edge");
                continue;
            }

            if (selected_mutation == "add_recurrent_edge") {
                uniform_int_distribution<int32_t> dist = get_recurrent_depth_dist();
                modified = g->add_recurrent_edge(mu, sigma, dist, edge_innovation_count);
                Log::debug("\tadding recurrent edge, modified: %d\n", modified);
                if (modified) g->set_generated_by("add_recurrent_edge");
                continue;
            }

            if (selected_mutation == "enable_edge") {
                modified = g->enable_edge();
                Log::debug("\tenabling edge, modified: %d\n", modified);
                if (modified) g->set_generated_by("enable_edge");
                continue;
            }

            if (selected_mutation == "disable_edge") {
                modified = g->disable_edge();
                Log::debug("\tdisabling edge, modified: %d\n", modified);
                if (modified) g->set_generated_by("disable_edge");
                continue;
            }

            if (selected_mutation == "split_edge") {
                uniform_int_distribution<int32_t> dist = get_recurrent_depth_dist();
                modified = g->split_edge(mu, sigma, new_node_type, dist, edge_innovation_count, node_innovation_count);
                Log::debug("\tsplitting edge, modified: %d\n", modified);
                if (modified) g->set_generated_by("split_edge(" + node_type_str + ")");
                continue;
            }

            if (selected_mutation == "add_node") {
                uniform_int_distribution<int32_t> dist = get_recurrent_depth_dist();
                modified = g->add_node(mu, sigma, new_node_type, dist, edge_innovation_count, node_innovation_count);
                Log::debug("\tadding node, modified: %d\n", modified);
                if (modified) g->set_generated_by("add_node(" + node_type_str + ")");
                continue;
            }

            if (selected_mutation == "enable_node") {
                modified = g->enable_node();
                Log::debug("\tenabling node, modified: %d\n", modified);
                if (modified) g->set_generated_by("enable_node");
                continue;
            }

            if (selected_mutation == "disable_node") {
                modified = g->disable_node();
                Log::debug("\tdisabling node, modified: %d\n", modified);
                if (modified) g->set_generated_by("disable_node");
                continue;
            }

            if (selected_mutation == "split_node") {
                uniform_int_distribution<int32_t> dist = get_recurrent_depth_dist();
                modified = g->split_node(mu, sigma, new_node_type, dist, edge_innovation_count, node_innovation_count);
                Log::debug("\tsplitting node, modified: %d\n", modified);
                if (modified) g->set_generated_by("split_node(" + node_type_str + ")");
                continue;
            }

            if (selected_mutation == "merge_node") {
                uniform_int_distribution<int32_t> dist = get_recurrent_depth_dist();
                modified = g->merge_node(mu, sigma, new_node_type, dist, edge_innovation_count, node_innovation_count);
                Log::debug("\tmerging node, modified: %d\n", modified);
                if (modified) g->set_generated_by("merge_node(" + node_type_str + ")");
                continue;
            }

        }
    }

    //get the new set of parameters (as new paramters may have been
    //added duriung mutation) and set them to the initial parameters
    //for epigenetic_initialization

    vector<double> new_parameters;
    g->get_weights(new_parameters);
    g->initial_parameters = new_parameters;

    if (Log::at_level(Log::DEBUG)) {
        Log::debug("getting mu/sigma before assign reachability\n");
        g->get_mu_sigma(new_parameters, mu, sigma);
    }

    g->assign_reachability();

    //reset the genomes statistics (as these carry over on copy)
    g->best_validation_mse = EXAMM_MAX_DOUBLE;
    g->best_validation_mae = EXAMM_MAX_DOUBLE;

    if (Log::at_level(Log::DEBUG)) {
        Log::debug("checking parameters after mutation\n");
        g->get_mu_sigma(g->initial_parameters, mu, sigma);
    }

    g->best_parameters.clear();
    g->number_mutations_generated_by = max_mutations;
}


void EXAMM::attempt_node_insert(vector<RNN_Node_Interface*> &child_nodes, const RNN_Node_Interface *node, const vector<double> &new_weights) {
    for (int32_t i = 0; i < (int32_t)child_nodes.size(); i++) {
        if (child_nodes[i]->get_innovation_number() == node->get_innovation_number()) return;
    }

    RNN_Node_Interface *node_copy = node->copy();
    node_copy->set_weights(new_weights);

    child_nodes.insert( upper_bound(child_nodes.begin(), child_nodes.end(), node_copy, sort_RNN_Nodes_by_depth()), node_copy);
}

void EXAMM::attempt_edge_insert(vector<RNN_Edge*> &child_edges, vector<RNN_Node_Interface*> &child_nodes, RNN_Edge *edge, RNN_Edge *second_edge, bool set_enabled) {
    for (int32_t i = 0; i < (int32_t)child_edges.size(); i++) {
        if (child_edges[i]->get_innovation_number() == edge->get_innovation_number()) {
            Log::fatal("ERROR in crossover! trying to push an edge with innovation_number: %d and it already exists in the vector!\n", edge->get_innovation_number());

            Log::fatal("vector innovation numbers: ");
            for (int32_t i = 0; i < (int32_t)child_edges.size(); i++) {
                Log::fatal("\t%d", child_edges[i]->get_innovation_number());
            }

            Log::fatal("This should never happen!\n");
            exit(1);

            return;
        } else if (child_edges[i]->get_input_innovation_number() == edge->get_input_innovation_number() &&
                child_edges[i]->get_output_innovation_number() == edge->get_output_innovation_number()) {

            Log::debug("Not inserting edge in crossover operation as there was already an edge with the same input and output innovation numbers!\n");
            return;
        }
    }

    vector<double> new_input_weights, new_output_weights;
    double new_weight = 0.0;
    if (second_edge != NULL) {
        double crossover_value = rng_crossover_weight(generator);
        new_weight = crossover_value * (second_edge->weight - edge->weight) + edge->weight;

        Log::trace("EDGE WEIGHT CROSSOVER :: better: %lf, worse: %lf, crossover_value: %lf, new_weight: %lf\n", edge->weight, second_edge->weight, crossover_value, new_weight);

        vector<double> input_weights1, input_weights2, output_weights1, output_weights2;
        edge->get_input_node()->get_weights(input_weights1);
        edge->get_output_node()->get_weights(output_weights1);

        second_edge->get_input_node()->get_weights(input_weights2);
        second_edge->get_output_node()->get_weights(output_weights2);

        new_input_weights.resize(input_weights1.size());
        new_output_weights.resize(output_weights1.size());

        //can check to see if input weights lengths are same
        //can check to see if output weights lengths are same

        for (int32_t i = 0; i < (int32_t)new_input_weights.size(); i++) {
            new_input_weights[i] = crossover_value * (input_weights2[i] - input_weights1[i]) + input_weights1[i];
            Log::trace("\tnew input weights[%d]: %lf\n", i, new_input_weights[i]);
        }

        for (int32_t i = 0; i < (int32_t)new_output_weights.size(); i++) {
            new_output_weights[i] = crossover_value * (output_weights2[i] - output_weights1[i]) + output_weights1[i];
            Log::trace("\tnew output weights[%d]: %lf\n", i, new_output_weights[i]);
        }

    } else {
        new_weight = edge->weight;
        edge->get_input_node()->get_weights(new_input_weights);
        edge->get_output_node()->get_weights(new_output_weights);
    }

    attempt_node_insert(child_nodes, edge->get_input_node(), new_input_weights);
    attempt_node_insert(child_nodes, edge->get_output_node(), new_output_weights);

    RNN_Edge *edge_copy = edge->copy(child_nodes);

    edge_copy->enabled = set_enabled;
    edge_copy->weight = new_weight;

    //edges have already been copied
    child_edges.insert( upper_bound(child_edges.begin(), child_edges.end(), edge_copy, sort_RNN_Edges_by_depth()), edge_copy);
}

void EXAMM::attempt_recurrent_edge_insert(vector<RNN_Recurrent_Edge*> &child_recurrent_edges, vector<RNN_Node_Interface*> &child_nodes, RNN_Recurrent_Edge *recurrent_edge, RNN_Recurrent_Edge *second_edge, bool set_enabled) {
    for (int32_t i = 0; i < (int32_t)child_recurrent_edges.size(); i++) {
        if (child_recurrent_edges[i]->get_innovation_number() == recurrent_edge->get_innovation_number()) {
            Log::fatal("ERROR in crossover! trying to push an recurrent_edge with innovation_number: %d  and it already exists in the vector!\n", recurrent_edge->get_innovation_number());
            Log::fatal("vector innovation numbers:\n");
            for (int32_t i = 0; i < (int32_t)child_recurrent_edges.size(); i++) {
                Log::fatal("\t %d", child_recurrent_edges[i]->get_innovation_number());
            }

            Log::fatal("This should never happen!\n");
            exit(1);

            return;
        } else if (child_recurrent_edges[i]->get_input_innovation_number() == recurrent_edge->get_input_innovation_number() &&
                child_recurrent_edges[i]->get_output_innovation_number() == recurrent_edge->get_output_innovation_number()) {

            Log::debug("Not inserting recurrent_edge in crossover operation as there was already an recurrent_edge with the same input and output innovation numbers!\n");
            return;
        }
    }


    vector<double> new_input_weights, new_output_weights;
    double new_weight = 0.0;
    if (second_edge != NULL) {
        double crossover_value = rng_crossover_weight(generator);
        new_weight = crossover_value * (second_edge->weight - recurrent_edge->weight) + recurrent_edge->weight;

        Log::debug("RECURRENT EDGE WEIGHT CROSSOVER :: better: %lf, worse: %lf, crossover_value: %lf, new_weight: %lf\n", recurrent_edge->weight, second_edge->weight, crossover_value, new_weight);

        vector<double> input_weights1, input_weights2, output_weights1, output_weights2;
        recurrent_edge->get_input_node()->get_weights(input_weights1);
        recurrent_edge->get_output_node()->get_weights(output_weights1);

        second_edge->get_input_node()->get_weights(input_weights2);
        second_edge->get_output_node()->get_weights(output_weights2);

        new_input_weights.resize(input_weights1.size());
        new_output_weights.resize(output_weights1.size());

        for (int32_t i = 0; i < (int32_t)new_input_weights.size(); i++) {
            new_input_weights[i] = crossover_value * (input_weights2[i] - input_weights1[i]) + input_weights1[i];
            Log::trace("\tnew input weights[%d]: %lf\n", i, new_input_weights[i]);
        }

        for (int32_t i = 0; i < (int32_t)new_output_weights.size(); i++) {
            new_output_weights[i] = crossover_value * (output_weights2[i] - output_weights1[i]) + output_weights1[i];
            Log::trace("\tnew output weights[%d]: %lf\n", i, new_output_weights[i]);
        }

    } else {
        new_weight = recurrent_edge->weight;
        recurrent_edge->get_input_node()->get_weights(new_input_weights);
        recurrent_edge->get_output_node()->get_weights(new_output_weights);
    }

    attempt_node_insert(child_nodes, recurrent_edge->get_input_node(), new_input_weights);
    attempt_node_insert(child_nodes, recurrent_edge->get_output_node(), new_output_weights);

    RNN_Recurrent_Edge *recurrent_edge_copy = recurrent_edge->copy(child_nodes);

    recurrent_edge_copy->enabled = set_enabled;
    recurrent_edge_copy->weight = new_weight;


    //recurrent_edges have already been copied
    child_recurrent_edges.insert( upper_bound(child_recurrent_edges.begin(), child_recurrent_edges.end(), recurrent_edge_copy, sort_RNN_Recurrent_Edges_by_depth()), recurrent_edge_copy);
}


RNN_Genome* EXAMM::crossover(RNN_Genome *p1, RNN_Genome *p2) {
    Log::debug("generating new genome by crossover!\n");
    Log::debug("p1->island: %d, p2->island: %d\n", p1->get_group_id(), p2->get_group_id());
    Log::debug("p1->number_inputs: %d, p2->number_inputs: %d\n", p1->get_number_inputs(), p2->get_number_inputs());

    for (uint32_t i = 0; i < p1->nodes.size(); i++) {
        Log::debug("p1 node[%d], in: %d, depth: %lf, layer_type: %d, node_type: %d, reachable: %d, enabled: %d\n", i, p1->nodes[i]->get_innovation_number(), p1->nodes[i]->get_depth(), p1->nodes[i]->get_layer_type(), p1->nodes[i]->get_node_type(), p1->nodes[i]->is_reachable(), p1->nodes[i]->is_enabled());
    }

    for (uint32_t i = 0; i < p2->nodes.size(); i++) {
        Log::debug("p2 node[%d], in: %d, depth: %lf, layer_type: %d, node_type: %d, reachable: %d, enabled: %d\n", i, p2->nodes[i]->get_innovation_number(), p2->nodes[i]->get_depth(), p2->nodes[i]->get_layer_type(), p2->nodes[i]->get_node_type(), p2->nodes[i]->is_reachable(), p2->nodes[i]->is_enabled());
    }

    double _mu, _sigma;
    Log::debug("getting p1 mu/sigma!\n");
    if (p1->best_parameters.size() == 0) {
        p1->set_weights(p1->initial_parameters);
        p1->get_mu_sigma(p1->initial_parameters, _mu, _sigma);
    } else {
        p1->set_weights(p1->best_parameters);
        p1->get_mu_sigma(p1->best_parameters, _mu, _sigma);
    }

    Log::debug("getting p2 mu/sigma!\n");
    if (p2->best_parameters.size() == 0) {
        p2->set_weights(p2->initial_parameters);
        p2->get_mu_sigma(p2->initial_parameters, _mu, _sigma);
    } else {
        p2->set_weights(p2->best_parameters);
        p2->get_mu_sigma(p2->best_parameters, _mu, _sigma);
    }

    //nodes are copied in the attempt_node_insert_function
    vector< RNN_Node_Interface* > child_nodes;
    vector< RNN_Edge* > child_edges;
    vector< RNN_Recurrent_Edge* > child_recurrent_edges;

    //edges are not sorted in order of innovation number, they need to be
    vector< RNN_Edge* > p1_edges = p1->edges;
    vector< RNN_Edge* > p2_edges = p2->edges;

    sort(p1_edges.begin(), p1_edges.end(), sort_RNN_Edges_by_innovation());
    sort(p2_edges.begin(), p2_edges.end(), sort_RNN_Edges_by_innovation());

    Log::debug("\tp1 innovation numbers AFTER SORT:\n");
    for (int32_t i = 0; i < (int32_t)p1_edges.size(); i++) {
        Log::trace("\t\t%d\n", p1_edges[i]->innovation_number);
    }
    Log::debug("\tp2 innovation numbers AFTER SORT:\n");
    for (int32_t i = 0; i < (int32_t)p2_edges.size(); i++) {
        Log::debug("\t\t%d\n", p2_edges[i]->innovation_number);
    }

    vector< RNN_Recurrent_Edge* > p1_recurrent_edges = p1->recurrent_edges;
    vector< RNN_Recurrent_Edge* > p2_recurrent_edges = p2->recurrent_edges;

    sort(p1_recurrent_edges.begin(), p1_recurrent_edges.end(), sort_RNN_Recurrent_Edges_by_innovation());
    sort(p2_recurrent_edges.begin(), p2_recurrent_edges.end(), sort_RNN_Recurrent_Edges_by_innovation());

    int32_t p1_position = 0;
    int32_t p2_position = 0;

    while (p1_position < (int32_t)p1_edges.size() && p2_position < (int32_t)p2_edges.size()) {
        RNN_Edge* p1_edge = p1_edges[p1_position];
        RNN_Edge* p2_edge = p2_edges[p2_position];

        int p1_innovation = p1_edge->innovation_number;
        int p2_innovation = p2_edge->innovation_number;

        if (p1_innovation == p2_innovation) {
            attempt_edge_insert(child_edges, child_nodes, p1_edge, p2_edge, true);

            p1_position++;
            p2_position++;
        } else if (p1_innovation < p2_innovation) {
            bool set_enabled = rng_0_1(generator) < more_fit_crossover_rate;
            if (p1_edge->is_reachable()) set_enabled = true;
            else set_enabled = false;

            attempt_edge_insert(child_edges, child_nodes, p1_edge, NULL, set_enabled);

            p1_position++;
        } else {
            bool set_enabled = rng_0_1(generator) < less_fit_crossover_rate;
            if (p2_edge->is_reachable()) set_enabled = true;
            else set_enabled = false;

            attempt_edge_insert(child_edges, child_nodes, p2_edge, NULL, set_enabled);

            p2_position++;
        }
    }

    while (p1_position < (int32_t)p1_edges.size()) {
        RNN_Edge* p1_edge = p1_edges[p1_position];

        bool set_enabled = rng_0_1(generator) < more_fit_crossover_rate;
        if (p1_edge->is_reachable()) set_enabled = true;
        else set_enabled = false;

        attempt_edge_insert(child_edges, child_nodes, p1_edge, NULL, set_enabled);

        p1_position++;
    }

    while (p2_position < (int32_t)p2_edges.size()) {
        RNN_Edge* p2_edge = p2_edges[p2_position];

        bool set_enabled = rng_0_1(generator) < less_fit_crossover_rate;
        if (p2_edge->is_reachable()) set_enabled = true;
        else set_enabled = false;

        attempt_edge_insert(child_edges, child_nodes, p2_edge, NULL, set_enabled);

        p2_position++;
    }

    //do the same for recurrent_edges
    p1_position = 0;
    p2_position = 0;

    while (p1_position < (int32_t)p1_recurrent_edges.size() && p2_position < (int32_t)p2_recurrent_edges.size()) {
        RNN_Recurrent_Edge* p1_recurrent_edge = p1_recurrent_edges[p1_position];
        RNN_Recurrent_Edge* p2_recurrent_edge = p2_recurrent_edges[p2_position];

        int p1_innovation = p1_recurrent_edge->innovation_number;
        int p2_innovation = p2_recurrent_edge->innovation_number;

        if (p1_innovation == p2_innovation) {
            //do weight crossover
            attempt_recurrent_edge_insert(child_recurrent_edges, child_nodes, p1_recurrent_edge, p2_recurrent_edge, true);

            p1_position++;
            p2_position++;
        } else if (p1_innovation < p2_innovation) {
            bool set_enabled = rng_0_1(generator) < more_fit_crossover_rate;
            if (p1_recurrent_edge->is_reachable()) set_enabled = true;
            else set_enabled = false;

            attempt_recurrent_edge_insert(child_recurrent_edges, child_nodes, p1_recurrent_edge, NULL, set_enabled);

            p1_position++;
        } else {
            bool set_enabled = rng_0_1(generator) < less_fit_crossover_rate;
            if (p2_recurrent_edge->is_reachable()) set_enabled = true;
            else set_enabled = false;

            attempt_recurrent_edge_insert(child_recurrent_edges, child_nodes, p2_recurrent_edge, NULL, set_enabled);

            p2_position++;
        }
    }

    while (p1_position < (int32_t)p1_recurrent_edges.size()) {
        RNN_Recurrent_Edge* p1_recurrent_edge = p1_recurrent_edges[p1_position];

        bool set_enabled = rng_0_1(generator) < more_fit_crossover_rate;
        if (p1_recurrent_edge->is_reachable()) set_enabled = true;
        else set_enabled = false;

        attempt_recurrent_edge_insert(child_recurrent_edges, child_nodes, p1_recurrent_edge, NULL, set_enabled);

        p1_position++;
    }

    while (p2_position < (int32_t)p2_recurrent_edges.size()) {
        RNN_Recurrent_Edge* p2_recurrent_edge = p2_recurrent_edges[p2_position];

        bool set_enabled = rng_0_1(generator) < less_fit_crossover_rate;
        if (p2_recurrent_edge->is_reachable()) set_enabled = true;
        else set_enabled = false;

        attempt_recurrent_edge_insert(child_recurrent_edges, child_nodes, p2_recurrent_edge, NULL, set_enabled);

        p2_position++;
    }

    sort(child_nodes.begin(), child_nodes.end(), sort_RNN_Nodes_by_depth());
    sort(child_edges.begin(), child_edges.end(), sort_RNN_Edges_by_depth());
    sort(child_recurrent_edges.begin(), child_recurrent_edges.end(), sort_RNN_Recurrent_Edges_by_depth());

    RNN_Genome *child = new RNN_Genome(child_nodes, child_edges, child_recurrent_edges);
    child->set_parameter_names(input_parameter_names, output_parameter_names);
    child->set_normalize_bounds(normalize_type, normalize_mins, normalize_maxs, normalize_avgs, normalize_std_devs);


    if (p1->get_group_id() == p2->get_group_id()) {
        child->set_generated_by("crossover");
    } else {
        child->set_generated_by("island_crossover");
    }

    double mu, sigma;

    vector<double> new_parameters;
    child->get_weights(new_parameters);
    Log::debug("getting mu/sigma before assign reachability\n");
    child->get_mu_sigma(new_parameters, mu, sigma);

    child->assign_reachability();

    //reset the genomes statistics (as these carry over on copy)
    child->best_validation_mse = EXAMM_MAX_DOUBLE;
    child->best_validation_mae = EXAMM_MAX_DOUBLE;

    //get the new set of parameters (as new paramters may have been
    //added duriung mutatino) and set them to the initial parameters
    //for epigenetic_initialization
    child->get_weights(new_parameters);
    child->initial_parameters = new_parameters;

    Log::debug("checking parameters after crossover\n");
    child->get_mu_sigma(child->initial_parameters, mu, sigma);

    child->best_parameters.clear();

    // This signifies generated by crossover
    child->number_mutations_generated_by = 0;

    return child;
}


uniform_int_distribution<int32_t> EXAMM::get_recurrent_depth_dist() {
    return uniform_int_distribution<int32_t>(this->min_recurrent_depth, this->max_recurrent_depth);
}
