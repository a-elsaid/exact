#include "colony.hxx"
#include <iostream>
#include <sstream>
using std::stringstream ;
#include <vector>
#include <fstream>
#include "point.hxx"
#include "ant.hxx"
using std::vector;
using std::cout;
using std::endl;
using std::to_string;

#include <map>
using std::map;
#include <algorithm>
#include <chrono>

#include <time.h>

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;
using std::pair;
using std::clock;
using std::shuffle;


#include "cants_dbscan.hxx"

#include "rnn_genome.hxx"
#include "edge_pheromone.hxx"
#include "node_pheromone.hxx"
#include "rnn_genome.hxx"



COLONY::~COLONY( ) {
    for (auto i: inputs) {
        delete i;
    }
    for (auto l: levels) {
        for (auto p: l.second) {
            delete p.second;
        }
    }
}


COLONY::COLONY(int _timeLag, int32_t _numberOfAnts): numberOfAnts(_numberOfAnts), timeLag(_timeLag) {

    int NUMBER_INPUTS  = 6 ;
    int NUMBER_OUTPUTS = 1 ;

    inserted_genomes  = 0;
    generated_genomes = 0;
    total_bp_epochs   = 0;

    edge_innovation_count = 0;
    node_innovation_count = 0;

    for ( int i=0; i< NUMBER_INPUTS; i++ ) {
        POINT* point = new POINT( i, 0, -1, -1, INITIAL_PHEROMONE ) ;
        inputs.push_back(point) ;
    }

    for ( int a=0; a<numberOfAnts; a++ ) {
        ants.push_back( ANT ( a ) );
    }

    for (int i=0; i<timeLag; i++) {
        levels_pheromones.push_back( INITIAL_PHEROMONE ) ;
    }
    cout << "NUMBER OF LEVELS: " << levels_pheromones.size() <<  endl ;

    double output_nodes_minus_one = NUMBER_OUTPUTS - 1 ;
    for ( int level=0; level<timeLag; level++ ) {
        vector<double> dum ;
        for (int i=1; i<NUMBER_OUTPUTS; i++) {
            dum.push_back( INITIAL_PHEROMONE ) ;
        }
        outputs.push_back( dum ) ;
    }
}

void COLONY::generateRNN() {
    for ( int a=0; a<ants.size(); a++ )
        ants[a].reset();


    vector<int> shuffeled_outputs ;
    for (int i=0; i<outputs.size(); i++ )
        shuffeled_outputs.push_back(i) ;
    {
    // uint16_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    // default_random_engine rng ;
    // rng.seed(seed);
    // shuffle(shuffeled_outputs.begin(), shuffeled_outputs.end(), rng);
    //
    // if ( shuffeled_outputs.size() > ants.size() ) {
    //     int taken_outputs = 0 ;
    //     bool keep_going = true ;
    //     while( keep_going ) {
    //         for ( int a=0; a<ants.size(); a++ ) {
    //             ants[a].favorit_outputs.push_back( shuffeled_outputs[taken_outputs] ) ;
    //             taken_outputs++ ;
    //             if ( taken_outputs > shuffeled_outputs.size() ) {
    //                 break;
    //                 keep_going = false ;
    //             }
    //         }
    //     }
    // }
    // else {
    //     int taken_ants = 0 ;
    //     bool keep_going = true ;
    //     while( keep_going ) {
    //         for ( int i=0; i<shuffeled_outputs.size(); i++ ) {
    //             ants[taken_ants].favorit_outputs.push_back( shuffeled_outputs[i] ) ;
    //             taken_ants++ ;
    //             if ( taken_ants > ants.size() ) {
    //                 break;
    //                 keep_going = false ;
    //             }
    //         }
    //     }
    // }
    }

    for ( int a=0; a<ants.size(); a++ ) {
        cout << "\n***-> ANT(" << ants[a].id  << "): FRD_BCK_RANGE: " << ants[a].searchRange << " EXPLORE_INSTINCT: " << ants[a].explorationInstinct << endl;

        bool iterate = true ;
        ants[a].climb(levels_pheromones, 0) ;
        ants[a].pickInput(inputs) ;
        iterate = ants[a].move( levels, outputs ) ;
        cout << "\t Input Node: " << ants[a].path.input->x << endl ;
        int count = 0;
        while ( iterate ) {

            if ( ants[a].path.points.back()->type==-2 )
                ants[a].climb(levels_pheromones, 1 ) ;
            else
                ants[a].climb(levels_pheromones, 0) ;
            iterate = ants[a].move( levels, outputs ) ;

        }
        {
        // for (auto x: levels[4]) {
        //     cout << "RRRRRR: " << "Level: " << " 4 " <<  x.second << endl;
        //     cout << "****************************\n";
        // }
        // for (int i=0; i< ants[a].path.points.size(); i++) {
        //     cout << "11 jjjj: " << levels[4].size() << endl;
        //     POINT* p = ants[a].path.points[i] ;
        //     POINT* p_ = levels[p->level][p->id] ;
        //     if ( p!=p_ && p->type == (-2) ) {
        //         // cout << levels[p->level][p->id] << endl ;
        //         // cout << p << endl;
        //         // cout << "Error: Missmatch between Ants points and levels points!" << endl;
        //         // exit(0);
        //     }
        //     cout << "22 jjjj: " << levels[4].size() <<"  " << p->type << endl;
        // }
        // cout << "\t Output Node: " << ants[a].path.points.back()->x << endl ;
        // for (auto x: levels[4]) {
        //     cout << "OOOOOOO: " << "Level: " << " 4 " <<  x.second << endl;
        //     cout << "****************************\n";
        // }
        }
    }

    cout << "* Centroids *\n" ;
    map<int, map<int, POINT*> > levels_ceteroids ;
    for (auto level : levels ) {
        // cout << "\tLEVEL: " << level.first << endl ;
        if ( level.second.empty() )
            continue ;
        vector <POINT*> points;
        for ( auto point: level.second ) {
            point.second->clusterID = -1 ;
            points.push_back( point.second ) ;
        }

        DBSCAN ds(MINIMUM_POINTS, EPSILON, points, level.first) ;
        ds.run();

        for (auto c: ds.centroids) {
            levels[c.second->level][c.second->id] = c.second ;
            {
            // cout << "\t\t\tCenteroid(" << c.first << "): x= " << c.second->x << " y= " << c.second->y << endl ;
            // cout <<  "\t\t\t\tEdge Weight: " << c.second->edge_weight << endl ;
            // for (auto i: c.second->node_weights[c.second->type]) {
            //     cout <<  "\t\t\t\tNode Weight: " << i << endl;
            // }
            }
        }
        levels_ceteroids[level.first] = ds.centroids ;

        // cout << "____________________\n\n" ;
    }
    rnns_blueprints[generated_genomes] = createSegments( levels_ceteroids ) ;

    buildGenome() ;
}

RNN_Node*  COLONY::buildNode(vector<RNN_Node_Interface*> &rnn_nodes, POINT* point) {
    for (auto node: rnn:nodes){
        if ( node->innovation_number == point->id ) {
            return node;
        }
    }
    int innovation_number = point->id;
    int current_layer     = point->level;
    int layer_type        ;
    if ( point->y==0 ) {
        layer_type = 0 ;
    } else if(point->y==1){
        layer_type = 1 ;
    } else{
        layer_type = 2 ;
    }


    if (node_type == LSTM_NODE) {
        rnn_nodes.push_back( new LSTM_Node(innovation_number, layer_type, point) );
    } else if (node_type == DELTA_NODE) {
        rnn_nodes.push_back( new Delta_Node(innovation_number, layer_type, current_layer) );
    } else if (node_type == GRU_NODE) {
        rnn_nodes.push_back( new GRU_Node(innovation_number, layer_type, current_layer) );
    } else if (node_type == MGU_NODE) {
        rnn_nodes.push_back( new MGU_Node(innovation_number, layer_type, current_layer) );
    } else if (node_type == UGRNN_NODE) {
        rnn_nodes.push_back( new UGRNN_Node(innovation_number, layer_type, current_layer) );
    } else if (node_type == FEED_FORWARD_NODE ) {
        rnn_nodes.push_back( new RNN_Node(innovation_number, layer_type, current_layer, node_type) );
    } else {
        cerr << "ACNNTO:: Error reading node from stream, unknown node_type: " << node_type << endl;
        exit(1);
    }

    return rnn_nodes.back() ;
}

void COLONY::drawEdge(vector<RNN_Node_Interface*> &rnn_edges, RNN_Node* in_node, RNN_Node* out_node ) {
    rnn_edges.push_back(new RNN_Edge(innovation_number, in_node, out_node));
}



RNN_Genome* COLONY::buildGenome() {
    rnn_nodes.clear();
    rnn_edges.clear();
    recurrent_edges.clear();

    vector <double> nodes_parameters ;
    vector <double> edges_parameters ;
    vector <double> recedges_parameters ;

    for ( auto p1: rnns_blueprints[generated_genomes] ) {
        RNN_Node* in_node = buildNode(rnn_nodes, p1);
        for ( auto p2: p1.second ) {
            RNN_Node* out_node = buildNode(rnn_nodes, p2) ;
            if ( p1.first->level == p2.second->level ) {
                rnn_edges.push_back(new RNN_Edge(0, in_node, out_node)); //putting innovation number 0 coz it won't be used
            }
            else {
                recurrent_edges.push_back(new RNN_Recurrent_Edge(0, depth, in_node, out_node));
            }
        }
    }


    RNN_Genome* g = NULL;
    g = new RNN_Genome(rnn_nodes, rnn_edges, recurrent_edges);
    if (g->outputs_unreachable()){
        std::cerr << "ERROR: Outputs are not Reachable" << '\n';
        exit(0) ;
    }
    g->set_generation_id(generated_genomes++);
    initialize_genome_parameters( g );
    return g ;
}

rnn_segment_type COLONY::createSegments( map<int, map<int, POINT*> > centeroids ) {
    POINT* source_node ;
    POINT* destination_node ;
    rnn_segment_type rnn_segments ;
    stringstream str ;
    cout << "*Creating RNN Segments*\n" ;
    cout << "*---------------------*\n" ;
    for ( ANT ant: ants ) {
        cout << "\tAnt(" << ant.id << ")" << endl;
        source_node      = ant.path.input  ;
        str << "\tInput: "  << source_node->id << " -> " ;
        if ( ant.path.points.size() == 1 ) {
            destination_node = ant.path.points.back() ;
            str << "Output: " << ant.path.points.back()->x ;
            rnn_segments.segments[source_node].insert( { destination_node->id, destination_node } ) ;
            return rnn_segments;
        }

        if ( ant.path.points[0]->type==-2 ) {
            destination_node = ant.path.points[0] ;
        }
        else {
            destination_node = centeroids[ant.path.points[0]->level][ant.path.points[0]->clusterID] ;
        }
        rnn_segments.segments[source_node].insert( { destination_node->id, destination_node } ) ;
        str << destination_node->id << " -> " ;
        for ( int i=0; i<ant.path.points.size()-1; i++ ) {
            source_node      = destination_node  ;
            if ( ant.path.points[i+1]->type==-2 ) {
                destination_node = ant.path.points[i+1] ;
            }
            else {
                destination_node = centeroids[ant.path.points[i+1]->level][ant.path.points[i+1]->clusterID] ;
            }

            if ( source_node->id == destination_node->id) {
                continue ;
            }
            rnn_segments.segments[source_node].insert( { destination_node->id, destination_node } ) ;
            str << destination_node->id << " -> "  ;
        }
        source_node      = destination_node  ;
        destination_node = ant.path.points.back() ;
        str << "Output: " << ant.path.points.back()->x << endl ;
        cout << str.str() << endl ;
        str.str( std::string() ) ;
        rnn_segments.segments[source_node].insert( { destination_node->id, destination_node } ) ;
    }
    return rnn_segments ;
}

void COLONY::evaporatePheromone () {
    cout << "Start Evaporation\n" ;
    for ( int i=0; i< inputs.size(); i++ ) {
        inputs[i]->pheromone-=EVAPORATION_RATE ;
        if ( inputs[i]->pheromone < INITIAL_PHEROMONE ) {
            inputs[i]->pheromone = INITIAL_PHEROMONE ;
        }
    }

    for ( int level=0; level<outputs.size(); level++ ) {
        for (int i=0; i<outputs[level].size(); i++) {
            outputs[level][i]-=EVAPORATION_RATE ;
            if ( outputs[level][i] < INITIAL_PHEROMONE ) {
                outputs[level][i] = INITIAL_PHEROMONE ;
            }
        }
    }
    for (int i=0; i<levels_pheromones.size(); i++) {
        levels_pheromones[i]-=EVAPORATION_RATE ;
        if ( levels_pheromones[i] < INITIAL_PHEROMONE ) {
            levels_pheromones[i] = INITIAL_PHEROMONE ;
        }
    }

    for (map <int, map<int32_t, POINT*> >::iterator it=levels.begin(); it!=levels.end(); ++it) {
        vector<int32_t> perishing_points_id ;
        for (map<int32_t, POINT*>::iterator itt=it->second.begin(); itt!=it->second.end(); ++itt) {
            itt->second->pheromone-= EVAPORATION_RATE ;
            if ( itt->second->pheromone <= 0.5) {
                perishing_points_id.push_back( itt->second->id )  ;
                continue ;
            }
            for (int i=0; i<itt->second->node_type_pheromones.size(); i++) {
                itt->second->node_type_pheromones[i]-=EVAPORATION_RATE ;
                if ( itt->second->node_type_pheromones[i] < INITIAL_PHEROMONE ) {
                    itt->second->node_type_pheromones[i] = INITIAL_PHEROMONE ;
                }
            }
        }
        for ( int32_t p_id: perishing_points_id ) {
            // cout << "Evaporation Removing: " << "Point level: " << levels[it->first][p_id]->level << endl ;
            levels[it->first].erase( p_id ) ;
        }
    }
    cout << "Finish Evaporation\n" ;
}

void COLONY::depositePheromone( map < POINT*, map<int32_t, POINT*> > &segments ) {
    map<int32_t, POINT*> points ;
    map<int, double> L;
    for (map < POINT*, map<int32_t, POINT*> >::iterator it=segments.begin(); it!=segments.end(); ++it) {
        points[it->first->id] = it->first ;
        L[it->first->level] = 1 ;
        for (map<int32_t, POINT*>::iterator itt=it->second.begin(); itt!=it->second.end(); ++itt) {
            points[itt->first] = itt->second ;
            L[itt->second->level] = 1 ;
        }
    }
    for ( auto l: L ) {
        levels_pheromones[l.first]++ ;
    }
    for ( map<int32_t, POINT*>::iterator it=points.begin(); it!=points.end(); ++it ) {
        if ( it->second->type!=-2 ) {
            it->second->pheromone++ ;
            it->second->node_type_pheromones[it->second->type]++ ;
             if ( it->second->pheromone > MAX_PHEROMONE ) {
                 it->second->pheromone = MAX_PHEROMONE ;
             }
             if ( it->second->node_type_pheromones[it->second->type] > MAX_PHEROMONE ) {
                 it->second->node_type_pheromones[it->second->type] = MAX_PHEROMONE ;
             }
         }
    }
    cout << "Finished Depositing Pheromone!\n" ;
}

void COLONY::writeColonyToFile (int number) {
    std::ofstream outfile;
    std::ofstream pathsfile;
    stringstream s1 ;
    // outfile.open("points_"+to_string(number)+".txt", std::ios_base::app); // append instead of overwrite
    outfile.open("points_"+to_string(number)+".txt"); // append instead of overwrite
    for (auto l: levels) {
        for (auto p: l.second) {
            if (p.second->edge_weight!=0.0)
                s1 << p.second->x << "," << p.second->y << "," <<l.first << "," << p.second->pheromone << "," << 1 << endl;
            else
                s1 << p.second->x << "," << p.second->y << "," <<l.first << "," << p.second->pheromone << "," << endl;
        }
    }
    outfile << s1.str() ;
    s1.str(std::string());

    stringstream s2;
    pathsfile.open("paths_"+to_string(number)+".txt"); // append instead of overwrite
    for (auto s: rnns_blueprints[number].segments) {
        if (s.second.empty()) {
            continue ;
        }
        s2 << s.first->x << "," << s.first->y << "," << s.first->level << "," << s.first->type ;
        for (auto p: s.second) {
            s1 << s2.str() << "," << p.second->x << "," << p.second->y << "," << p.second->level << ","<< p.second->type << endl;
        }
        s2.str(std::string());
    }
    cout << s1.str();
    pathsfile << s1.str() ;
}

void COLONY::startLiving() {
    int terminates_sent = 0;
    while (true) {
        //wait for a incoming message
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        int source = status.MPI_SOURCE;
        int tag = status.MPI_TAG;
        Log::debug("probe returned message from: %d with tag: %d\n", source, tag);

        //if the message is a work request, send a genome
        if (tag == WORK_REQUEST_TAG) {
            receive_work_request(source);
            examm_mutex.lock();
            RNN_Genome *genome = generateRNN();
            examm_mutex.unlock();
            if (genome == NULL) { //search was completed if it returns NULL for an individual
                //send terminate message
                Log::info("terminating worker: %d\n", source);
                send_terminate_message(source);
                terminates_sent++;
                Log::debug("sent: %d terminates of %d\n", terminates_sent, (max_rank - 1));
                if (terminates_sent >= max_rank - 1) return;
            } else {
                //send genome
                Log::debug("sending genome to: %d\n", source);
                send_genome_to(source, genome);

                //delete this genome as it will not be used again
                delete genome;
            }
        } else if (tag == GENOME_LENGTH_TAG) {
            Log::debug("received genome from: %d\n", source);
            RNN_Genome *genome = receive_genome_from(source);

            insert_genome(genome);

            //delete the genome as it won't be used again, a copy was inserted
            delete genome;
            //this genome will be deleted if/when removed from population
        } else {
            Log::fatal("ERROR: received message from %d with unknown tag: %d", source, tag);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}

/*
void worker(int rank) {
    Log::set_id("worker_" + to_string(rank));

    while (true) {
        Log::debug("sending work request!\n");
        send_work_request(0);
        Log::debug("sent work request!\n");

        MPI_Status status;
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int tag = status.MPI_TAG;

        Log::debug("probe received message with tag: %d\n", tag);

        if (tag == TERMINATE_TAG) {
            Log::debug("received terminate tag!\n");
            receive_terminate_message(0);
            break;

        } else if (tag == GENOME_LENGTH_TAG) {
            Log::debug("received genome!\n");
            RNN_Genome* genome = receive_genome_from(0);

            //have each worker write the backproagation to a separate log file
            string log_id = "genome_" + to_string(genome->get_generation_id()) + "_worker_" + to_string(rank);
            Log::set_id(log_id);
            genome->backpropagate_stochastic(training_inputs, training_outputs, validation_inputs, validation_outputs);
            Log::release_id(log_id);

            //go back to the worker's log for MPI communication
            Log::set_id("worker_" + to_string(rank));

            send_genome_to(0, genome);

            delete genome;
        } else {
            Log::fatal("ERROR: received message with unknown tag: %d\n", tag);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    //release the log file for the worker communication
    Log::release_id("worker_" + to_string(rank));
}

//this will insert a COPY, original needs to be deleted
bool COLONY::insert_genome(RNN_Genome* genome) {
    if (!genome->sanity_check()) {
        cerr << "ERROR, genome failed sanity check on insert!" << endl;
        exit(1);
    }

    double new_fitness = genome->get_fitness();
    bool was_inserted = true;

    inserted_genomes++;
    total_bp_epochs += genome->get_bp_iterations();

    genome->update_generation_map(generated_from_map);
    cout << "genomes evaluated: " << setw(10) << inserted_genomes << ", inserting: " << parse_fitness(genome->get_fitness()) << endl;

    if (population.size() >= population_size  && new_fitness > population.back()->get_fitness()) {
        cout << "ignoring genome, fitness: " << new_fitness << " > worst population" << " fitness: " << population.back()->get_fitness() << endl;
        print_population();
        return false;
    }

    int32_t duplicate_genome = population_contains(genome);
    if (duplicate_genome >= 0) {
        //if fitness is better, replace this genome with new one
        cout << "found duplicate at position: " << duplicate_genome << endl;
        RNN_Genome *duplicate = population[duplicate_genome];
        if (duplicate->get_fitness() > new_fitness) {
            //erase the genome with loewr fitness from the vector;
            cout << "REPLACING DUPLICATE GENOME, fitness of genome in search: " << parse_fitness(duplicate->get_fitness()) << ", new fitness: " << parse_fitness(genome->get_fitness()) << endl;
            population.erase(population.begin() + duplicate_genome);
            delete duplicate;
            depositePheromone ( rnns_blueprints[i].segments) ;
        } else {
            cerr << "\tpopulation already contains genome! not inserting." << endl;
            print_population();
            return false;
        }
    }
    if (population.size() < population_size || population.back()->get_fitness() > new_fitness) {
        //this genome will be inserted
        was_inserted = true;
        depositePheromone ( rnns_blueprints[i].segments) ;

        if (population.size() == 0 || genome->get_fitness() < get_best_genome()->get_fitness()) {
            if (genome->get_fitness() != EXALT_MAX_DOUBLE) {
                //need to set the weights for non-initial genomes so we
                //can generate a proper graphviz file
                vector<double> best_parameters = genome->get_best_parameters();
                genome->set_weights(best_parameters);
            }
            genome->write_graphviz(output_directory + "/rnn_genome_" + to_string(inserted_genomes) + ".gv");
            genome->write_to_file(output_directory + "/rnn_genome_" + to_string(inserted_genomes) + ".bin", false);
        }
    } else {
        was_inserted = false;
        cout << "not inserting genome due to poor fitness" << endl;
    }
    print_population();
    cout << "printed population!" << endl;
    if ( use_pheromone_weight_update )
        pheromones_update_weights( genome );
    return was_inserted ;
}
*/

RNN_Genome* COLONY::get_best_genome() {
  if (population.size() <= 0) {
      return NULL;
  } else {
      return population[0];
  }
}

void COLONY::print_population() {
    cout << "POPULATIONS: " << endl;
    for (int32_t i = 0; i < (int32_t)population.size(); i++) {
        cout << "\tPOPULATION " << i << ":" << endl;

        cout << "\t" << RNN_Genome::print_statistics_header() << endl;

        cout << "\t" << population[i]->print_statistics() << endl;
    }

    cout << endl << endl;

    if (log_file != NULL) {

        //make sure the log file is still good
        if (!log_file->good()) {
            log_file->close();
            delete log_file;

            string output_file = output_directory + "/fitness_log.csv";
            log_file = new ofstream(output_file, std::ios_base::app);

            if (!log_file->is_open()) {
                cerr << "ERROR, could not open ACNNTO output log: '" << output_file << "'" << endl;
                exit(1);
            }
        }

        RNN_Genome *best_genome = get_best_genome();

        std::chrono::time_point<std::chrono::system_clock> currentClock = std::chrono::system_clock::now();
        long milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(currentClock - startClock).count();

        (*log_file) << inserted_genomes
            << "," << total_bp_epochs
            << "," << milliseconds
            << "," << best_genome->best_validation_mae
            << "," << best_genome->best_validation_mse
            << "," << best_genome->get_enabled_node_count()
            << "," << best_genome->get_enabled_edge_count()
            << "," << best_genome->get_enabled_recurrent_edge_count()
            << "," << ants
            << "," << hidden_layers_depth
            << "," << hidden_layer_nodes << endl;


        memory_log << inserted_genomes
            << "," << total_bp_epochs
            << "," << milliseconds
            << "," << best_genome->best_validation_mae
            << "," << best_genome->best_validation_mse
            << "," << best_genome->get_enabled_node_count()
            << "," << best_genome->get_enabled_edge_count()
            << "," << best_genome->get_enabled_recurrent_edge_count()
            << "," << ants
            << "," << hidden_layers_depth
            << "," << hidden_layer_nodes << endl;
    }
}

int32_t COLONY::getNumberOfAnts(){
    return numberOfAnts ;
}

int main(){
    COLONY colony1(5, 30) ;
    colony1.startLiving() ;
    cout << "Number of Ants: " << colony1.getNumberOfAnts() << endl;
}
