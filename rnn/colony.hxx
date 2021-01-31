#ifndef COLONY_HXX
#define COLONY_HXX
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>
#include "point.hxx"
#include "ant.hxx"
#include <chrono>
#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;
#include "cants_dbscan.hxx"
// #include "rnn_genome.hxx"

#include "rnn_genome.hxx"
#include "edge_pheromone.hxx"
#include "node_pheromone.hxx"
#include "rnn_genome.hxx"


#define NUMBER_OF_NODE_TYPES 5
#define EVAPORATION_RATE 0.05
#define INITIAL_PHEROMONE    1.0
#define MAX_PHEROMONE    10.0

#define MINIMUM_POINTS 1     // minimum number of cluster
#define EPSILON (0.05*0.05)  // distance for clustering, metre^2

struct rnn_segment_type {
    map < POINT*, map<int32_t, POINT*> > segments;
};

class COLONY {
    private:
        int timeLag ;
        int32_t numberOfAnts ;
        vector < double > levels_pheromones ;           // lag levels pheromones
        vector < POINT* > inputs ;                      // Inputs pheromones
        vector<vector < double > > outputs ;            // Output <position, pheromone>
        map <int, map<int32_t, POINT*> > levels ;       // dimension pheromones
        minstd_rand0 generator;
        int types[5] = { 1, 2, 3, 4, 5 } ;
        void depositePheromone( map < POINT*, map<int32_t, POINT*> > &segments) ;
        void evaporatePheromone () ;
        void writeColonyToFile (int number) ;
        vector<RNN_Genome*> population;

        int32_t inserted_genomes  ;
        int32_t generated_genomes ;
        int32_t total_bp_epochs   ;
        int32_t edge_innovation_count ;
        int32_t node_innovation_count ;

    public:

        COLONY(int _timeLag, int32_t _numberOfAnts) ;
        vector < ANT > ants ;
        int32_t getNumberOfAnts () ;
        rnn_segment_type createSegments( map<int, map<int, POINT*> > centeroids ) ;
        void generateRNN( ) ;
        void startLiving( ) ;
        map <int32_t, rnn_segment_type> rnns_blueprints ;

    ~COLONY();
};




#endif
