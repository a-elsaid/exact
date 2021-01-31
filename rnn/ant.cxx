#include "ant.hxx"
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
using std::pair;

#include <chrono>

#include <vector>
using std::vector;

#include "point.hxx"

#include <iomanip>
using std::setprecision ;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <math.h>
using std::sqrt;
using std::cos ;
using std::sin ;

#include <algorithm>
using std::max ;
using std::min ;

ANT::ANT( int32_t id_, int32_t _highestL, bool _verbose): id(id_), highestL(_highestL), verbose(_verbose) {
    uniform_real_distribution<double> rng( 0.0, 1.0 ) ;
    generateSeed() ;
    newBorn() ;
    generateSeed() ;
    vel_searchRange         = rng( generator ) ;
    generateSeed() ;
    vel_explorationInstinct = rng( generator ) ;
}

void ANT::newBorn( ) {
    uniform_real_distribution<double> rng_sr( 0.01, 0.98 ) ;
    uniform_real_distribution<double> rng_ei( 0.25, 0.75 ) ;
    generateSeed() ;
    searchRange  = rng_sr( generator ) ;
    // searchRange  = 0.2 ;
    explorationInstinct = rng_ei( generator ) ;
    best_searchRange  = searchRange  ;
    best_explorationInstinct = explorationInstinct ;
}

void ANT::climb(vector<double>& vertical_pheromones, int jump) {
    previousL = currentL ;
    if ( currentL == highestL ) {
        return ;
    }
    if ( verbose ) cout << "Climbing! \n";

    double sum_pheromones = 0 ;
    for ( int i=currentL+jump; i<vertical_pheromones.size(); i++ ) {
        sum_pheromones+=vertical_pheromones[i] ;
    }
    uniform_real_distribution<double> rng( 0.0, 1.0 ) ;
    generateSeed() ;
    double rand_gen = rng( generator ) * sum_pheromones;
    int a;
    for ( a = currentL+jump; a < vertical_pheromones.size(); a++) {      // Do not allow ants to move down
        if ( rand_gen<=vertical_pheromones[a] )
          break;
        else
          rand_gen-= vertical_pheromones[a] ;
    }
    currentL = a ;
    path.levels.push_back( a ) ;
    if ( verbose ) cout << "Climbed! \n";
}

void ANT::pickInput(vector<POINT*> &inputs) {
    if ( verbose ) cout << "Choosing Input!\n" ;
    double sum_pheromones = 0;
    for ( POINT* p: inputs )
        sum_pheromones+= p->pheromone;
    uniform_real_distribution<double> rng( 0.0, 1.0 ) ;
    generateSeed() ;
    double rand_gen = rng(generator) * sum_pheromones;
    int a;
    for ( a = 0; a < inputs.size(); a++) {
        if ( rand_gen<=inputs[a]->pheromone )
          break;
        else
          rand_gen-= inputs[a]->pheromone ;
    }
    path.input = inputs[a] ;
    path.input->level = currentL ;
    if ( verbose ) cout << "Choose Input: " << path.input->id << " at level: " << path.input->level << endl ;
}

bool ANT::reachedOutput( vector<vector < POINT > > &outputs, POINT* point ) {
    if ( verbose ) cout << "Reached Output!\n" ;
    double sum_pheromones = 0;
    for ( POINT o: outputs[currentL] ) {
        sum_pheromones+=o.pheromone ;
    }
    uniform_real_distribution<double> rng( 0.0, 1.0 ) ;
    generateSeed() ;
    double rand_gen = rng(generator) * sum_pheromones;
    int a;

    for ( a = 0; a < outputs[currentL].size(); a++) {
        if ( rand_gen<=outputs[currentL][a].pheromone )
          break;
        else
          rand_gen-= outputs[currentL][a].pheromone ;
    }
    // point->x = float(a)/(outputs[0].size()-1) ;
    // point->y = 1 ;
    // point->type = -2 ;
    // point->level = currentL ;
    // point->pheromone = INITIAL_PHEROMONE ;      //REMOVE THIS
    point = &outputs[currentL][a] ;
    currentY = point->y ;
    currentL = point->level ;

    path.points.push_back( point ) ;
    if ( verbose ) cout<< "Chose Output: " << a << endl ;
    if ( currentL>=outputs.size()-1 ) {
        return false ;
    }
    return true ;
}

bool ANT::centerOfMass( map<int32_t, POINT*> &plane_points, POINT* point,  vector<vector < POINT > > &outputs ) {
    if ( verbose ) cout << "\t\tCalcualting Center of Mass!\n" ;
    vector <POINT*> points_in_range ;
    for (auto const& p : plane_points ) {
        double delta_x  = currentX - p.second->x ;
        double delta_y  = currentY - p.second->y ;
        double distance = sqrt( (delta_x * delta_x ) + ( delta_y * delta_y ) ) ;
        if ( currentL == previousL) {
            if ( distance<=searchRange && currentY>=p.second->y )  {
                points_in_range.push_back(p.second) ;
            }
        }
        else {
            if ( distance<=searchRange ) {
                points_in_range.push_back(p.second) ;
            }
        }
    }
    double mass   = 0.0 ; double x_mass = 0.0 ; double y_mass = 0.0 ;
    for (POINT* p: points_in_range) {
        mass += p->pheromone ;
        x_mass += ( p->x * p->pheromone ) ;
        y_mass += ( p->y * p->pheromone ) ;
    }
    if ( x_mass!=0.0 || y_mass!=0.0 ) {
        didCM = true ;
        double c_x = x_mass / mass ;
        double c_y = y_mass / mass ;
        point->x = c_x ;
        point->y = c_y ;
        point->level = currentL ;
        point->pheromone = INITIAL_PHEROMONE ;      //REMOVE THIS

        if ( currentL == previousL && currentY > c_y ) {        // If same level & Ant's Y > new point's Y: disard point
            delete point ;
            point = NULL ;
            return true  ;
        }

        currentX = c_x ;
        currentY = c_y ;

        bool add_point = true ;
        for ( POINT* p: points_in_range ) {                      // If point exists: dicard point
            if ( p->x==c_x && p->y==c_y && ! add_point ) {
                point = p ;
                add_point = false ;
                break ;
            }
        }
        if ( add_point ) {
            plane_points[point->id] = point ;
            path.points.push_back( point ) ;
            print_msg( point );
            return true ;
        }
        add_point = true ;
        for ( auto p: path.points ) {
            if ( p->x == point->x && p->y==point->y ) {
                add_point = false ;
                break ;
            }
        }
        if ( add_point ) {
            path.points.push_back( point ) ;
        }
        return true ;
    }
    return createNewPoint ( plane_points, point, outputs ) ;

}

void ANT::generateSeed () {
    uint16_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator = minstd_rand0(seed);
}

bool ANT::createNewPoint ( map<int32_t, POINT*> &plane_points, POINT* point, vector<vector < POINT > > &outputs ) {
    if ( verbose ) cout << "\t\tCreating New Point!   currentX: " << currentX << " currentY: " << currentY << endl;
    double ang_rng_srt = 0.0 ;
    if ( currentL!=previousL )
        ang_rng_srt = -1.0 ;
    if ( currentL<previousL ) {
      std::cerr << "ERROR: Current Lavel < Previoius Level In Ants Movement... THIS SHOULD NEVER HAPPEN." << '\n';
      exit(1);
    }

    double oldX = currentX ;
    double oldY = currentY ;


    uniform_real_distribution<double> angle_rng( ang_rng_srt, 1.0 ) ;
    generateSeed() ;
    double angle_dissect = angle_rng( generator ) ;
    currentX+= searchRange * cos( angle_dissect * PI ) ;
    currentY+= searchRange * sin( angle_dissect * PI ) ;

    if ( verbose ) cout << "ANGLE: "<< angle_dissect << "  X: " << currentX << "  Y: " << currentY  << endl ;

    while ( currentY < 0.001 ||  0.0 > currentX || currentX > 1.0 ) {
        currentY = oldY ;
        currentX = oldX ;
        generateSeed() ;
        double angle_dissect = angle_rng( generator ) ;
        currentX+= searchRange * cos( angle_dissect * PI ) ;
        currentY+= searchRange * sin( angle_dissect * PI ) ;
        if ( verbose ) cout << "ANGLE: "<< angle_dissect << "  X: " << currentX << "  Y: " << currentY  << endl ;
    }


    if ( ( currentY ) >= 0.99 ) {
        return reachedOutput ( outputs, point ) ;
    }
    else {
        didCM = false ;
        point->x = currentX ;
        point->y = currentY ;
        point->level = currentL ;
        point->pheromone = INITIAL_PHEROMONE ;      //REMOVE THIS

        bool point_exits = false ;
        // Check if points exits in plane
        for ( auto p: plane_points ) {
            if ( p.second->x == point->x && p.second->y == point->y ) {
                point_exits = true ;
                POINT* temp_point = point ;
                point = p.second ;
                delete temp_point ;
                temp_point = NULL ;
                break ;
            }
        }
        if ( ! point_exits ) {
            plane_points[point->id] = point ;
            path.points.push_back( point ) ;
            if ( verbose ) cout << ("\t\tPoint Exists in Plane:: ") ;
            print_msg( point ) ;
            return true ;
        }
        // Check if points exits in ant's path
        point_exits = false ;
        for ( POINT* p: path.points ) {
            if ( p->x == point->x && p->y == point->y ) {
                point_exits = true ;
                break ;
            }
        }
        if ( ! point_exits ) {
            if ( verbose ) cout << ("\t\tPoint Exists in Ant's Path:: ") ;
            path.points.push_back( point ) ;
        }
    }
    if ( verbose ) cout << "\t\tCreated New Point\n" ;
    return true ;
}

bool ANT::move( map <int, map<int32_t, POINT*> > &levels,  vector<vector < POINT > > &outputs ) {
    POINT* point = new POINT();
    uniform_real_distribution<double> rand_rng( 0.0, 1.0 ) ;
    generateSeed () ;
    if ( ( levels[currentL].empty() || rand_rng( generator ) > NEW_POINT_PROBABILITY ) || didCM ) {
        return createNewPoint( levels[currentL], point, outputs ) ;
    }
    return centerOfMass ( levels[currentL], point, outputs ) ;
}

void ANT::print_msg (POINT* point) {
    if ( verbose ) cout << std::fixed << std::setprecision(4) << "\t\t Level: " << currentL << "(Prv: " << previousL << ")" << " PointId: " << point->id << " Point_X: " << point->x << " Point_Y: " << point->y << " Pheromone: " << point->pheromone  << "  -  X: " << currentX << " Y: " << currentY << " L: " << currentL << " Ant's points size: " << path.points.size() << endl;
}

void ANT::smartAnt( double fitness ) {
    if ( fitness < bestFitness ) {
        bestFitness              = fitness ;
        best_searchRange         = searchRange  ;
        best_explorationInstinct = explorationInstinct ;
    }
    uniform_real_distribution<double> rng( 0.0, 1.0 ) ;
    generateSeed() ;
    double r1 = rng( generator ) ;
    double r2 = rng( generator ) ;
    vel_searchRange         = w * vel_searchRange         + c1 * r1 * (best_searchRange - searchRange) ;
    vel_explorationInstinct = w * vel_explorationInstinct + c1 * r1 * (best_searchRange - searchRange) ;
    searchRange         += vel_searchRange ;
    explorationInstinct += vel_explorationInstinct ;

    double min_searchRange = 0.01;
    double max_searchRange = 0.98;
    double min_explorationInstinct = 0.25;
    double max_explorationInstinct = 0.75;
    if ( ! ( searchRange < min_searchRange
             && searchRange > max_searchRange ) ) {
        uniform_real_distribution<double> rng_sr( 0.01, 0.98 ) ;
        generateSeed() ;
        searchRange  = rng_sr( generator ) ;
    }
    if ( ! ( explorationInstinct < min_explorationInstinct
             && explorationInstinct > max_explorationInstinct ) ) {
        uniform_real_distribution<double> rng_ei( 0.25, 0.75 ) ;
        generateSeed() ;
        explorationInstinct  = rng_ei( generator ) ;
    }

    // Let The Ant Die In Peace and Let A New one To Be Born!
    if ( ( (double)rand() / RAND_MAX ) < DEATH_PROBABILITY ) {
        newBorn() ;
    }
}

void ANT::reset( ) {
    previousL   = 0 ;
    currentL    = 0 ;
    currentX    = 0.5 ;
    currentY    = 0 ;
    path.levels.clear() ;
    path.points.clear() ;
}



double ANT::get_searchRange () {
    return searchRange;
}
double ANT::get_explorationInstinct () {
    return explorationInstinct;
}
void ANT::set_searchRange(double _searchRange) {
    searchRange = _searchRange ;
}
void ANT::set_explorationInstinct(double _explorationInstinct) {
    explorationInstinct = _explorationInstinct ;
}

// int main () {
//     ANT ant;
//     return 1 ;
// }
