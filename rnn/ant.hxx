#ifndef ANT_HXX
#define ANT_HXX

#include <iostream>
using std::pair;

#include <vector>
using std::vector;

#include <chrono>

#include "point.hxx"

#include <iomanip>
using std::setprecision ;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <map>
using std::map;

#include <math.h>
using std::sqrt;

#include <algorithm>
using std::max ;
using std::min ;

#define PI 3.14159265
#define NEW_POINT_PROBABILITY 0.5
#define DEATH_PROBABILITY 0.1
#define INITIAL_PHEROMONE    1.0

class ANT {
private:
    int id;
    double searchRange ;
    double explorationInstinct ;
    int32_t highestL  ;
    bool didCM = false ;
    int32_t previousL = 0 ;
    int32_t currentL  = 0 ;
    double currentX = 0.5 ;
    double currentY = 0 ;
    minstd_rand0 generator;
    double bestFitness ;
    double w=0.7 ;
    double c1=1.49445 ;
    double c2=1.49445 ;
    double vel_searchRange ;
    double vel_explorationInstinct ;
    double best_searchRange ;
    double best_explorationInstinct ;
    double min_searchRange = 0.01;
    double max_searchRange = 0.98;
    double min_explorationInstinct = 0.25;
    double max_explorationInstinct = 0.75;
    bool verbose ;
    struct PATH {
        POINT* input  ;
        // POINT* output ;
        vector<int> levels ;
        vector<POINT*> points ;
    } path;
    vector<int> favorit_outputs ;

public:
    ANT ( int32_t id_, int32_t _highestL, bool _verbose ) ;
    double get_searchRange() ;
    double get_explorationInstinct() ;
    void set_searchRange(double _searchRange) ;
    void set_explorationInstinct(double _explorationInstinct) ;
    void climb(vector<double> &vertical_pheromones, int jump ) ;
    void pickInput(vector<POINT*> &inputs) ;
    bool move( map <int, map<int32_t, POINT*> > &level_pheromones,  vector<vector < POINT > > &outputs ) ;
    bool centerOfMass( map<int32_t, POINT*> &plane_points, POINT* point,  vector<vector < POINT > > &outputs ) ;
    bool reachedOutput( vector<vector < POINT > > &outputs, POINT* point ) ;
    bool createNewPoint ( map<int32_t, POINT*> &plane_points, POINT* point,  vector<vector < POINT > > &outputs ) ;
    void print_msg (POINT* p) ;
    void reset ( ) ;
    void smartAnt( double fitness ) ;
    void newBorn( ) ;
    void generateSeed () ;



    friend class COLONY;
};


#endif
