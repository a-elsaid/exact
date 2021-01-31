#ifndef POINT_HXX
#define POINT_HXX

#include <fstream>
#include <iostream>
using std::cout;
using std::endl;

#include <map>
using std::map ;

#include <vector>
using std::vector ;

#define INITIAL_PHEROMONE    1.0

class POINT {
    private:
        static int counter;
    public:
        POINT( ) ;
        POINT( double x_, double y_,  int type_, int level_, double pheromone_ ) ;
        std::string parameter_name = "";
        int32_t id ;
        double  x ;
        double  y ;
        int type ;
        int level ;
        double edge_weight = 0.0 ;
        int32_t node_id ;
        vector<double> node_type_pheromones ;
        double pheromone ;
        int clusterID = -1 ;


    ~POINT() ;
};
#endif
