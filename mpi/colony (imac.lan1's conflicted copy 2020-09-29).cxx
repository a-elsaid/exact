#include "colony.hxx"
#include <iostream>
#include <sstream>
using std::stringstream ;
#include <vector>
#include <fstream>
#include "rnn/point.hxx"
#include "rnn/ant.hxx"
using std::vector;
using std::cout;
using std::endl;
using std::to_string;

#include <map>
using std::map;
#include <algorithm>
using std::max ;
#include <chrono>

#include <time.h>

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;
using std::pair;
using std::clock;
using std::shuffle;

#include "mpi.h"
#include <mutex>
using std::mutex;

#include "common/files.hxx"
#include "common/log.hxx"
#include "common/arguments.hxx"


#include "rnn/cants_dbscan.hxx"

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


COLONY::COLONY(int32_t _numberOfAnts,
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
    int32_t _hidden_layer_nodes,
    int32_t _max_recurrent_depth):
numberOfAnts(_numberOfAnts),
population_size(_population_size),
max_genomes(_max_genomes),
number_inputs(_input_parameter_names.size()),
input_parameter_names(_input_parameter_names),
number_outputs(_output_parameter_names.size()),
output_parameter_names(_output_parameter_names),
bp_iterations(_bp_iterations),
learning_rate(_learning_rate),
use_high_threshold(_use_high_threshold),
high_threshold(_high_threshold),
use_low_threshold(_use_low_threshold),
low_threshold(_low_threshold),
output_directory(_output_directory),
hidden_layers_depth(_hidden_layers_depth),
hidden_layer_nodes(_hidden_layer_nodes),
normalize_type(_normalize_type),
max_recurrent_depth(_max_recurrent_depth),
normalize_avgs(_normalize_avgs),
normalize_std_devs(_normalize_std_devs) {

    possible_node_types.clear();
    possible_node_types.push_back(SIMPLE_NODE);
    possible_node_types.push_back(JORDAN_NODE);
    possible_node_types.push_back(ELMAN_NODE);
    possible_node_types.push_back(UGRNN_NODE);
    possible_node_types.push_back(MGU_NODE);
    possible_node_types.push_back(GRU_NODE);
    possible_node_types.push_back(LSTM_NODE);
    possible_node_types.push_back(DELTA_NODE);

    int NUMBER_INPUTS  = number_inputs ;
    int NUMBER_OUTPUTS = number_outputs ;

    inserted_genomes  = 0;
    generated_genomes = 0;
    total_bp_epochs   = 0;

    edge_innovation_count = 0;
    node_innovation_count = 0;

    for ( int i=0; i< NUMBER_INPUTS; i++ ) {
        POINT* point = new POINT( i, 0, -1, -1, INITIAL_PHEROMONE ) ;
        point->parameter_name = input_parameter_names[i] ;
        inputs.push_back(point) ;
    }

    for ( int32_t a=0; a<numberOfAnts; a++ ) {
        ants.push_back( ANT ( a, max_recurrent_depth-1 ) );
    }

    for (int i=0; i<max_recurrent_depth; i++) {
        levels_pheromones.push_back( INITIAL_PHEROMONE * 2 * ( max_recurrent_depth - i ) ) ;
    }
    cout << "NUMBER OF LEVELS: " << levels_pheromones.size() <<  endl ;

    // double output_nodes_minus_one = NUMBER_OUTPUTS - 1 ;
    for ( int level=0; level<max_recurrent_depth; level++ ) {
        vector<POINT> dum ;
        for (int i=0; i<NUMBER_OUTPUTS; i++) {
          POINT p ;
          p.pheromone = INITIAL_PHEROMONE ;
          p.x = float(i)/max(1, NUMBER_OUTPUTS) ;
          p.y = 1 ;
          p.type = -2 ;
          p.level = level ;
          p.parameter_name = output_parameter_names[i] ;
          dum.push_back( p ) ;
        }
        outputs.push_back( dum ) ;
    }
    startClock = std::chrono::system_clock::now();




    if (output_directory != "") {
        mkpath(output_directory.c_str(), 0777);
        log_file = new ofstream(output_directory + "/" + "fitness_log.csv");
        (*log_file) << "Inserted Genomes, Total BP Epochs, Time, Best Val. MAE, Best Val. MSE, Enabled Nodes, Enabled Edges, Enabled Rec. Edges";
        //memory_log << "Inserted Genomes, Total BP Epochs, Time, Best Val. MAE, Best Val. MSE, Enabled Nodes, Enabled Edges, Enabled Rec. Edges";


        (*log_file) << endl;
        //memory_log << endl;

        op_log_file = new ofstream(output_directory + "/op_log.csv");

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

        for (int i = 0; i < ops_with_node_type.size(); i++) {
            string op = ops_with_node_type[i];
            for (int j = 0; j < possible_node_types.size(); j++)
                op_log_ordering.push_back(op + "(" + NODE_TYPES[possible_node_types[j]] + ")");
        }

        for (int i = 0; i < op_log_ordering.size(); i++) {
            string op = op_log_ordering[i];
            (*op_log_file) << op;
            (*op_log_file) << " Generated, ";
            (*op_log_file) << op;
            (*op_log_file) << " Inserted, ";

            inserted_counts = 0;
            generated_counts = 0;
        }

        map<string, int>::iterator it;

        (*op_log_file) << endl;

    } else {
        log_file = NULL;
        op_log_file = NULL;
    }






}

RNN_Genome* COLONY::generateRNN() {
    for ( int a=0; a<ants.size(); a++ )
        ants[a].reset();

    vector<POINT*> picked_input_points ;

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
        picked_input_points.push_back(ants[a].path.input) ;
        iterate = ants[a].move( levels, outputs ) ;
        cout << "\t Input Node: " << ants[a].path.input->x << endl ;
        int count = 0;
        while ( iterate ) {
            if ( ants[a].path.points.back()->type==-2 ) {
                ants[a].path.points.pop_back() ;
                ants[a].climb(levels_pheromones, 1 ) ;
            }
            else {
                ants[a].climb(levels_pheromones, 0) ;
            }
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

        DBSCAN ds(MINIMUM_POINTS, EPSILON, points, level.first, level.second ) ;
        ds.run();

        for (auto c: ds.centroids) {
            // levels[c.second->level][c.second->id] = c.second ;
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

    return buildGenome( picked_input_points ) ;
}

RNN_Node*  COLONY::buildNode(vector<RNN_Node_Interface*> &rnn_nodes, POINT* point, vector <double> &parameters ) {
    for (auto node: rnn_nodes){
        if ( node->get_innovation_number() == point->id ) {
            return (RNN_Node*) node;
        }
    }

    int innovation_number = point->id;
    int current_layer     = point->level;
    int node_type     = point->type;
    if ( node_type > 0  ) {
        node_type+=2 ;
    }


    if  (node_type == LSTM_NODE) {
        rnn_nodes.push_back( new LSTM_Node(innovation_number, HIDDEN_LAYER, point->y) );
    } else if (node_type == DELTA_NODE) {
        rnn_nodes.push_back( new Delta_Node(innovation_number, HIDDEN_LAYER, point->y) );
    } else if (node_type == GRU_NODE) {
        rnn_nodes.push_back( new GRU_Node(innovation_number, HIDDEN_LAYER, point->y) );
    } else if (node_type == MGU_NODE) {
        rnn_nodes.push_back( new MGU_Node(innovation_number, HIDDEN_LAYER, point->y) );
    } else if (node_type == UGRNN_NODE) {
        rnn_nodes.push_back( new UGRNN_Node(innovation_number, HIDDEN_LAYER, point->y) );
    } else if (node_type == SIMPLE_NODE || node_type<0) {
        if ( node_type==-1 ) {
            rnn_nodes.push_back( new RNN_Node(innovation_number, INPUT_LAYER, point->y, SIMPLE_NODE, point->parameter_name) );
        } else if ( node_type==-2 ) {
            rnn_nodes.push_back( new RNN_Node(innovation_number, OUTPUT_LAYER, point->y, SIMPLE_NODE, point->parameter_name ) );
        } else {
          rnn_nodes.push_back( new RNN_Node(innovation_number, HIDDEN_LAYER, point->y, SIMPLE_NODE) ) ;
        }

    } else {
        cerr << "ACNNTO:: Error reading node from stream, unknown node_type: " << node_type << endl;
        exit(1);
    }


    // cout << "Number of Entered nodes: " << rnn_nodes.size() << " Node Type: " << point->type << endl ;
    vector<double> ps ;
    rnn_nodes.back()->get_weights(ps) ;
    for (double p: ps){
      parameters.push_back(p) ;
    }
    return (RNN_Node*) rnn_nodes.back() ;
}

RNN_Genome* COLONY::buildGenome( vector<POINT*> picked_input_points ) {
    vector<RNN_Node_Interface*> rnn_nodes;
    vector<RNN_Edge*> rnn_edges;
    vector<RNN_Recurrent_Edge*> recurrent_edges;

    vector <double> parameters ;
    vector <double> edges_parameters ;
    vector <double> rec_edges_parameters ;
    // parameters.assign( nodes_total_number_weights + edges_inherited_weights.size() + recedges_inherited_weights.size(), 0.0 );


    for ( auto p1: rnns_blueprints[generated_genomes].segments ) {

      RNN_Node* in_node = buildNode(rnn_nodes, p1.first, parameters );

      for ( auto p2: p1.second ) {

        RNN_Node* out_node = buildNode(rnn_nodes, p2.second, parameters ) ;
        if ( p1.first->level == p2.second->level || p1.first->type==-1 ) {
            // cout << "Creating Edge between Node: " << in_node->innovation_number << " and Node: " << out_node->innovation_number << endl ;
            rnn_edges.push_back(new RNN_Edge(rec_edge_counter++, in_node, out_node));

            edges_parameters.push_back( p1.first->edge_weight ) ;

        }
        else {

            recurrent_edges.push_back(new RNN_Recurrent_Edge(rec_edge_counter++, ( p2.second->level - p1.first->level), in_node, out_node));

            rec_edges_parameters.push_back( p1.first->edge_weight ) ;

        }
      }
    }



    // Adding input nodes which were not picked by the ants and disabling them
    for (auto input_name: input_parameter_names) {
      bool add_it = true ;
      for (auto p: picked_input_points) {
        if ( p->parameter_name == input_name ) {
          add_it = false ;
          break ;
        }
      }
      if (add_it) {
          RNN_Node* dum = new RNN_Node(-9, INPUT_LAYER, 0, SIMPLE_NODE, input_name ) ;
          dum->enabled = false ;
          rnn_nodes.push_back(dum) ;
          parameters.push_back(0.0) ;
      }
    }


    int count = 0;
    for (auto x: rnn_nodes) {
        if (x->layer_type ==INPUT_LAYER) {
            count++;
        }
    }
    cout << "Number of Input Nodes: " << count << endl;


    for (auto w: edges_parameters ) {
        parameters.push_back( w );
    }
    for (auto w: rec_edges_parameters ) {
        parameters.push_back( w );
    }


    RNN_Genome* g = NULL;
    g = new RNN_Genome(rnn_nodes, rnn_edges, recurrent_edges);
    if (g->outputs_unreachable()){
        cerr << "ERROR: Outputs are not Reachable" << '\n';
        exit(0) ;
    }
    g->set_generation_id(generated_genomes++);
    g->set_parameter_names(input_parameter_names, output_parameter_names);
    g->set_bp_iterations(bp_iterations);
    g->set_learning_rate(learning_rate);
    if (use_high_threshold) g->enable_high_threshold(high_threshold);
    if (use_low_threshold) g->enable_low_threshold(low_threshold);
    if (use_dropout) g->enable_dropout(dropout_probability);
    // if (!epigenetic_weights) g->initialize_randomly();
    g->initial_parameters =  parameters ;
    g->best_parameters =  parameters ;
    generated_counts++ ;
    return g ;
}

void COLONY::copy_point(POINT* org, POINT* trg) {
    trg->parameter_name       = org->parameter_name;
    trg->id                   = org->id ;
    trg->x                    = org->x ;
    trg->y                    = org->y ;
    trg->type                 = org->type ;
    trg->level                = org->level ;
    trg->edge_weight          = org->edge_weight ;
    trg->node_id              = org->edge_weight ;
    trg->node_type_pheromones = trg->node_type_pheromones ;
    trg->pheromone            = org->pheromone ;
    trg->clusterID            = org->clusterID ;
}

rnn_segment_type COLONY::createSegments( map<int, map<int, POINT*> > centeroids ) {
    POINT* source_node ;
    POINT* destination_node ;
    POINT* temp_node  = new POINT();
    rnn_segment_type rnn_segments ;
    stringstream str ;
    cout << "*Creating RNN Segments*\n" ;
    cout << "*---------------------*\n" ;
    for ( ANT ant: ants ) {
        cout << "\tAnt(" << ant.id << ")" << endl;
        source_node      = ant.path.input  ;
        // rnn_segments.input_points.push_back(source_node) ;
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
        str << destination_node->id <<"(" << destination_node->level <<")"<< " -> " ;
        source_node      = destination_node  ;
        for ( int i=0; i<ant.path.points.size()-1; i++ ) {
            temp_node       = source_node  ;
            // copy_point (source_node, temp_node) ;
            if ( ant.path.points[i+1]->type==-2 ) {
                destination_node = ant.path.points[i+1] ;
            }
            else {
                destination_node = centeroids[ant.path.points[i+1]->level][ant.path.points[i+1]->clusterID] ;
            }

            // cout << "1- SRC Y(" << source_node->id << "): " <<  source_node->y << " DEST Y(" << destination_node->id << "): "  << destination_node->y  << endl;
            // if ( source_node->id == destination_node->id || source_node->y >= destination_node->y ) {
            if ( source_node->id == destination_node->id || ( source_node->y>destination_node->y && source_node->level==destination_node->level ) ) {
                source_node = temp_node ;
                // copy_point ( temp_node, source_node) ;
                continue ;
            }
            // cout << "2- SRC Y(" << source_node->id << "): " <<  source_node->y << " DEST Y(" << destination_node->id << "): "  << destination_node->y  << endl;
            rnn_segments.segments[source_node].insert( { destination_node->id, destination_node } ) ;
            str << destination_node->id << " (L: " << destination_node->level << " X: " << destination_node->x << " Y: " << destination_node->y << ")" << " -> "  ;
            source_node      = destination_node  ;
        }
        // rnn_segments.segments[source_node].insert( { destination_node->id, destination_node } ) ;
        // source_node      = destination_node  ;
        // destination_node = ant.path.points.back() ;
        str << "Output: " << ant.path.points.back()->id << endl ;
        cout << str.str() << endl ;
        str.str( std::string() ) ;
    }
    // cout << "------------------------------" << endl;
    // for (auto x: rnn_segments.segments) {
    //     for (auto y: x.second){
    //         cout << "CHECK " << "IN: " << x.first->id << " OUT: "<< y.second->id << endl;
    //     }
    // }
    // cout << "------------------------------" << endl;
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
            outputs[level][i].pheromone-=EVAPORATION_RATE ;
            if ( outputs[level][i].pheromone < INITIAL_PHEROMONE ) {
                outputs[level][i].pheromone = INITIAL_PHEROMONE ;
            }
        }
    }
    for (int i=0; i<levels_pheromones.size(); i++) {
        levels_pheromones[i]-=EVAPORATION_RATE ;
        if ( levels_pheromones[i] < INITIAL_PHEROMONE ) {
            levels_pheromones[i] = INITIAL_PHEROMONE ;
        }
    }


    for (map <int, map<int32_t, POINT*> >::iterator level=levels.begin(); level!=levels.end(); ++level) {
        vector<int32_t> perishing_points_id ;
        if ( level->first == 0) cout << "POINTS IN LEVEL 1: " << level->second.size() << endl ;
        for (map<int32_t, POINT*>::iterator points=level->second.begin(); points!=level->second.end(); ++points) {
            if ( level->first == 0) cout << "BEFORE: " << points->second->pheromone  << "      weight: " << points->second->pheromone << endl;
            points->second->pheromone-= EVAPORATION_RATE ;
            if ( level->first == 0) {
                cout << "AFTER: "  << points->second->pheromone << "  ID: " << points->second->id << endl;
                cout << "X: " << points->second->x << " **Y: " << points->second->y << endl;
                cout << "____________________\n" ;
            }
            if ( points->second->pheromone <= POINT_DIEING_THRESHOLD ) {
                perishing_points_id.push_back( points->second->id )  ;
                continue ;
            }
            for (int i=0; i<points->second->node_type_pheromones.size(); i++) {
                points->second->node_type_pheromones[i]-=EVAPORATION_RATE ;
                if ( points->second->node_type_pheromones[i] < INITIAL_PHEROMONE ) {
                    points->second->node_type_pheromones[i] = INITIAL_PHEROMONE ;
                }
            }
        }
        for ( int32_t p_id: perishing_points_id ) {
            // cout << "Evaporation Removing: " << "Point level: " << levels[it->first][p_id]->level << endl ;
            levels[level->first].erase( p_id ) ;
        }
    }
    cout << "Finish Evaporation\n" ;
}

void COLONY::depositePheromone( map < POINT*, map<int32_t, POINT*> > &segments, RNN_Genome* genome ) {
    map<int32_t, POINT*> points ;
    vector<double> levels;
    for (map < POINT*, map<int32_t, POINT*> >::iterator it=segments.begin(); it!=segments.end(); ++it) {
        points[it->first->id] = it->first ;
        levels.push_back( it->first->level) ;
        for (map<int32_t, POINT*>::iterator itt=it->second.begin(); itt!=it->second.end(); ++itt) {
            points[itt->first] = itt->second ;
            levels.push_back(itt->second->level) ;
        }
    }

    for ( auto l: levels ) {
        levels_pheromones[l]++ ;
    }
    for ( map<int32_t, POINT*>::iterator it=points.begin(); it!=points.end(); ++it ) {
        if ( it->second->type>0 ) {
            it->second->pheromone++ ;
            it->second->node_type_pheromones[it->second->type]++ ;
             if ( it->second->pheromone > MAX_PHEROMONE ) {
                 it->second->pheromone = MAX_PHEROMONE ;
             }
             if ( it->second->node_type_pheromones[it->second->type] > MAX_PHEROMONE ) {
                 it->second->node_type_pheromones[it->second->type] = MAX_PHEROMONE ;
             }
         } else if (it->second->type==-2) {
           for (int l=0; l<outputs.size(); l++) {
             for ( POINT o: outputs[l] ) {
               if ( o.id ==  it->second->id) {
                 o.pheromone++ ;
               }
             }
           }
         } else if ( it->second->type==-1 ) {
           for ( POINT* i: inputs ) {
             if ( i->id ==  it->second->id) {
               i->pheromone++ ;
             }
           }
         }
         for (RNN_Edge* edge: genome->edges) {
           if ( edge->get_innovation_number() == it->first ) {
             it->second->edge_weight = (it->second->edge_weight+edge->weight)/2 ;
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

void COLONY::send_work_request(int target) {
    int work_request_message[1];
    work_request_message[0] = 0;
    MPI_Send(work_request_message, 1, MPI_INT, target, WORK_REQUEST_TAG, MPI_COMM_WORLD);
}


void COLONY::receive_work_request(int source) {
    MPI_Status status;
    int work_request_message[1];
    MPI_Recv(work_request_message, 1, MPI_INT, source, WORK_REQUEST_TAG, MPI_COMM_WORLD, &status);
}

RNN_Genome* COLONY::receive_genome_from(int source) {
    MPI_Status status;
    int length_message[1];
    MPI_Recv(length_message, 1, MPI_INT, source, GENOME_LENGTH_TAG, MPI_COMM_WORLD, &status);

    int length = length_message[0];

    Log::debug("receiving genome of length: %d from: %d\n", length, source);

    char* genome_str = new char[length + 1];

    Log::debug("receiving genome from: %d\n", source);
    MPI_Recv(genome_str, length, MPI_CHAR, source, GENOME_TAG, MPI_COMM_WORLD, &status);

    genome_str[length] = '\0';

    Log::trace("genome_str:\n%s\n", genome_str);

    RNN_Genome* genome = new RNN_Genome(genome_str, length);

    delete [] genome_str;
    return genome;
}

void COLONY::send_genome_to(int target, RNN_Genome* genome) {
    char *byte_array;
    int32_t length;

    genome->write_to_array(&byte_array, length);

    Log::debug("sending genome of length: %d to: %d\n", length, target);

    int length_message[1];
    length_message[0] = length;
    MPI_Send(length_message, 1, MPI_INT, target, GENOME_LENGTH_TAG, MPI_COMM_WORLD);

    Log::debug("sending genome to: %d\n", target);
    MPI_Send(byte_array, length, MPI_CHAR, target, GENOME_TAG, MPI_COMM_WORLD);

    free(byte_array);
}

void COLONY::send_terminate_message(int target) {
    int terminate_message[1];
    terminate_message[0] = 0;
    MPI_Send(terminate_message, 1, MPI_INT, target, TERMINATE_TAG, MPI_COMM_WORLD);
}

void COLONY::receive_terminate_message(int source) {
    MPI_Status status;
    int terminate_message[1];
    MPI_Recv(terminate_message, 1, MPI_INT, source, TERMINATE_TAG, MPI_COMM_WORLD, &status);
}



void COLONY::startLiving(int max_rank) {

    // for (int i=0; i<20; i++){
    //     RNN_Genome *genome = generateRNN();
    // }


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
            colony_mutex.lock();
            RNN_Genome *genome = NULL ;
            cout << "LLLLLLLLLLL: " << max_genomes << endl;
            if ( generated_counts<=max_genomes ) {
                cout << "DDDDDDDD" << endl;
                // RNN_Genome *genome = generateRNN();
                genome = generateRNN();
            }
            cout << "XXXXXXOOOOO" << endl;
            exit(1);
            colony_mutex.unlock();
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
            evaporatePheromone() ;
            writeColonyToFile(genome->get_generation_id()) ;

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
*/
void COLONY::worker(int rank) {
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
            depositePheromone ( rnns_blueprints[genome->generation_id].segments, genome ) ;
        } else {
            cerr << "\tpopulation already contains genome! not inserting." << endl;
            print_population();
            return false;
        }
    }
    if (population.size() < population_size || population.back()->get_fitness() > new_fitness) {
        //this genome will be inserted
        was_inserted = true;
        depositePheromone ( rnns_blueprints[genome->generation_id].segments, genome ) ;

        if (population.size() == 0 || genome->get_fitness() < get_best_genome()->get_fitness()) {
            if (genome->get_fitness() != EXAMM_MAX_DOUBLE) {
                //need to set the weights for non-initial genomes so we
                //can generate a proper graphviz file
                vector<double> best_parameters = genome->get_best_parameters();
                genome->set_weights(best_parameters);
            }
            genome->write_graphviz(output_directory + "/rnn_genome_" + to_string(inserted_genomes) + ".gv");
            genome->write_to_file(output_directory + "/rnn_genome_" + to_string(inserted_genomes) + ".bin");
        }
    } else {
        was_inserted = false;
        cout << "not inserting genome due to poor fitness" << endl;
    }

    // print_population();
    cout << "printed population!" << endl;
    return was_inserted ;
}

int32_t COLONY::population_contains(RNN_Genome* genome) {
    for (int32_t j = 0; j < (int32_t)population.size(); j++) {
        if (population[j]->equals(genome)) {
            return j;
        }
    }

    return -1;
}

bool COLONY::populations_full() const {
    if (population.size() < population_size) return false;
    return true;
}

RNN_Genome* COLONY::get_best_genome() {
  if (population.size() <= 0) {
      return NULL;
  } else {
      return population[0];
  }
}


/*
void COLONY::print_population() {
    cout << "POPULATIONS: " << endl;
    for (int32_t i = 0; i < (int32_t)population.size(); i++) {
        cout << "\tPOPULATION " << i << ":" << endl;
        cout << "\t" << RNN_Genome::print_statistics_header() << endl;

        cout << "\t" << population[i]->print_statistics() << endl;
    }

    cout << endl << endl;

    if (log_file != NULL) {

        cout << "AAAAA==" << endl;
        // make sure the log file is still good
        if (!log_file->good()) {
          cout << "BBBBB" << endl;
            log_file->close();
            delete log_file;

            string output_file = output_directory + "/fitness_log.csv";
            log_file = new ofstream(output_file, std::ios_base::app);

            if (!log_file->is_open()) {
                cerr << "ERROR, could not open ACNNTO output log: '" << output_file << "'" << endl;
                exit(1);
            }
        }

        // cout << "CCCCC" << endl;
        RNN_Genome *best_genome = get_best_genome();
        // cout << "DDDDD" << endl;

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
            // << "," << ants
            << "," << hidden_layers_depth
            << "," << hidden_layer_nodes << endl;


        memory_log << inserted_genomes
            << "," << total_bp_epochs
            << "," << milliseconds
            // << "," << best_genome->best_validation_mae
            // << "," << best_genome->best_validation_mse
            // << "," << best_genome->get_enabled_node_count()
            // << "," << best_genome->get_enabled_edge_count()
            // << "," << best_genome->get_enabled_recurrent_edge_count()
            // << "," << ants
            << "," << hidden_layers_depth
            << "," << hidden_layer_nodes << endl;
    }
}
*/
void COLONY::print_population() {
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


        RNN_Genome *best_genome = get_best_genome();


        std::chrono::time_point<std::chrono::system_clock> currentClock = std::chrono::system_clock::now();
        long milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(currentClock - startClock).count();

        cout << "AAAAA==" << endl ;
        cout << "total_bp_epochs: " << total_bp_epochs << endl ;
        cout << "milliseconds: " << milliseconds << endl ;
        cout << "best_genome->best_validation_mae: " << best_genome->best_validation_mae << endl ;
        cout << "best_genome->best_validation_mse: " << best_genome->best_validation_mse << endl ;
        cout << "best_genome->get_enabled_node_count(): " << best_genome->get_enabled_node_count() << endl ;
        cout << "best_genome->get_enabled_edge_count(): " << best_genome->get_enabled_edge_count() << endl ;
        cout << "best_genome->get_enabled_recurrent_edge_count(): " << best_genome->get_enabled_recurrent_edge_count() << endl ;

        (*log_file) << total_bp_epochs
            << "," << milliseconds
            << "," << best_genome->best_validation_mae
            << "," << best_genome->best_validation_mse
            << "," << best_genome->get_enabled_node_count()
            << "," << best_genome->get_enabled_edge_count()
            << "," << best_genome->get_enabled_recurrent_edge_count() << endl ;

        cout << "BBBBB==" << endl;
        for (int i = 0; i < op_log_ordering.size(); i++) {
            string op = op_log_ordering[i];
            (*op_log_file) << generated_counts << ", " << inserted_counts  << ", ";
        }

        (*op_log_file) << endl;

    }

}



void COLONY::set_possible_node_types(vector<string> possible_node_type_strings) {
    possible_node_types.clear();

    for (int32_t i = 0; i < possible_node_type_strings.size(); i++) {
        string node_type_s = possible_node_type_strings[i];

        bool found = false;

        for (int32_t j = 0; j < NUMBER_NODE_TYPES; j++) {
            if (NODE_TYPES[j].compare(node_type_s) == 0) {
                found = true;
                possible_node_types.push_back(j);
            }
        }

        if (!found) {
            cerr << "ERROR! unknown node type: '" << node_type_s << "'" << endl;
            exit(1);
        }
    }
}

int32_t COLONY::getNumberOfAnts(){
    return numberOfAnts ;
}

COLONY* colony;

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    int rank, max_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &max_rank);


    vector< vector< vector<double> > > training_inputs;
    vector< vector< vector<double> > > training_outputs;
    vector< vector< vector<double> > > validation_inputs;
    vector< vector< vector<double> > > validation_outputs;
    vector<string> arguments;
    bool finished = false;
    arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_rank(rank);
    Log::set_id("main_" + to_string(rank));
    Log::restrict_to_rank(0);



    TimeSeriesSets *time_series_sets = NULL;

    if (rank == 0) {
        //only have the master process print TSS info
        time_series_sets = TimeSeriesSets::generate_from_arguments(arguments);
        if (argument_exists(arguments, "--write_time_series")) {
            string base_filename;
            get_argument(arguments, "--write_time_series", true, base_filename);
            time_series_sets->write_time_series_sets(base_filename);
        }
    } else {
        time_series_sets = TimeSeriesSets::generate_from_arguments(arguments);
    }



    int32_t time_offset = 1;
    get_argument(arguments, "--time_offset", true, time_offset);

    time_series_sets->export_training_series(time_offset, training_inputs, training_outputs);
    time_series_sets->export_test_series(time_offset, validation_inputs, validation_outputs);

    int number_inputs = time_series_sets->get_number_inputs();
    int number_outputs = time_series_sets->get_number_outputs();

    Log::debug("number_inputs: %d, number_outputs: %d\n", number_inputs, number_outputs);

    int32_t population_size;
    get_argument(arguments, "--population_size", true, population_size);

    int32_t max_genomes;
    get_argument(arguments, "--max_genomes", true, max_genomes);

    int32_t hidden_layers_depth = 0;
    get_argument(arguments, "--hidden_layers_depth", false, hidden_layers_depth);

    int32_t hidden_layer_nodes = 0;
    get_argument(arguments, "--hidden_layer_nodes", false, hidden_layer_nodes);

    int32_t bp_iterations;
    get_argument(arguments, "--bp_iterations", true, bp_iterations);

    double learning_rate = 0.001;
    get_argument(arguments, "--learning_rate", false, learning_rate);

    double high_threshold = 1.0;
    bool use_high_threshold = get_argument(arguments, "--high_threshold", false, high_threshold);

    double low_threshold = 0.05;
    bool use_low_threshold = get_argument(arguments, "--low_threshold", false, low_threshold);

    double dropout_probability = 0.0;
    bool use_dropout = get_argument(arguments, "--dropout_probability", false, dropout_probability);

    string output_directory = "";
    get_argument(arguments, "--output_directory", false, output_directory);

    vector<string> possible_node_types;
    get_argument_vector(arguments, "--possible_node_types", false, possible_node_types);

    int32_t min_recurrent_depth = 1;
    get_argument(arguments, "--min_recurrent_depth", false, min_recurrent_depth);

    int32_t max_recurrent_depth = 10;
    get_argument(arguments, "--max_recurrent_depth", false, max_recurrent_depth);


    bool start_filled = false;
    get_argument(arguments, "--start_filled", false, start_filled);

    Log::clear_rank_restriction();


    colony = new COLONY(30, population_size, max_genomes,
        time_series_sets->get_input_parameter_names(),
        time_series_sets->get_output_parameter_names(),
        time_series_sets->get_normalize_type(),
        time_series_sets->get_normalize_mins(),
        time_series_sets->get_normalize_maxs(),
        time_series_sets->get_normalize_avgs(),
        time_series_sets->get_normalize_std_devs(),
        bp_iterations,
        learning_rate,
        use_high_threshold,
        high_threshold,
        use_low_threshold,
        low_threshold,
        output_directory,
        hidden_layers_depth,
        hidden_layer_nodes,
        max_recurrent_depth);
    colony->training_inputs = training_inputs ;
    colony->training_outputs = training_outputs;
    colony->validation_inputs = validation_inputs;
    colony->validation_outputs = validation_outputs;
    if (rank == 0) {
        if (possible_node_types.size() > 0) colony->set_possible_node_types(possible_node_types);
        colony->startLiving(max_rank);
    } else {
        cout << colony->input_parameter_names.size() << " <-GGGGGG " << endl ;
        colony->worker(rank);
    }
    Log::set_id("main_" + to_string(rank));

    finished = true;

    Log::debug("rank %d completed!\n");
    Log::release_id("main_" + to_string(rank));

    MPI_Finalize();
    delete time_series_sets;
    return 0;

    // COLONY colony1(5, 30) ;
    // colony1.startLiving(max_rank) ;
    // cout << "Number of Ants: " << colony1.getNumberOfAnts() << endl;



}
