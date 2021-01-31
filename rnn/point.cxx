#include <fstream>
#include <iostream>
#include "point.hxx"

using std::cout;
using std::endl;
POINT::~POINT(){

}
int POINT::counter = 0 ;
POINT::POINT( ) {
    id = counter+=1 ;
}

POINT::POINT( double x_, double y_,  int type_, int level_, double pheromone_ ):
              x(x_),  y(y_),  type(type_), level(level_), pheromone(pheromone_)  {
    id = counter+=1 ;
    for (int i=0; i<6; i++) {
        // vector<double> dum;
        // for (int j=0; j<11; j++) {
        //     dum.push_back(0.0) ;
        // }
        // node_weights[i] = dum ;
        node_type_pheromones.push_back(INITIAL_PHEROMONE) ;
    }
}
