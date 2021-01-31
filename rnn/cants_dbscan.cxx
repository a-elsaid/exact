/*
This code is originaly implemented by james-yoo
https://github.com/james-yoo/DBSCAN
Modified by A.ElSaid
*/

#include "cants_dbscan.hxx"

#include <chrono>
#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

int DBSCAN::run()
{
    int clusterID = 1;
    for( POINT* p: m_points ) {
        if ( p->clusterID == UNCLASSIFIED ) {
            if ( expandCluster(p, clusterID) != FAILURE ) {
                clusterID += 1;
            }
        }
    }


    for (int c=1; c<clusterID+1; c++) {
        // std::vector<double> node_weights(10) ;
        double edge_weight = 0.0 ;
        std::vector<double> node_type_pheromones(5) ;
        int count_initialized_points = 0.0 ;
        int num_points = 0 ;
        double c_x = 0.0 ;
        double c_y = 0.0 ;
        for (POINT* p: m_points) {
            if ( c == p->clusterID ) {
                c_x+=p->x ;
                c_y+=p->y ;
                num_points++ ;
                if ( p->edge_weight != 0.0 ) {
                    for ( int i=0; i<p->node_type_pheromones.size(); i++ ) {
                        node_type_pheromones[i]+= p->node_type_pheromones[i] ;
                    }
                    edge_weight+= p->edge_weight ;
                    count_initialized_points++ ;
                }
            }
        }

        if ( num_points == 0 ) {
            // cout << "WARNING (CANTS_DBscan): EMPTY CLUSTER!" << endl;
            continue ;
        }

        for ( int i=0; i<node_type_pheromones.size(); i++ ) {
            node_type_pheromones[i]/= max(1,count_initialized_points) ;
            if ( node_type_pheromones[i] == 0.0 ) {
                node_type_pheromones[i] = INITIAL_PHEROMONE ;
            }
        }

        //choosing node type
        double sum_pheromones = 0;
        for ( double p_value: node_type_pheromones){
            sum_pheromones+=p_value;
        }
        uniform_real_distribution<double> rng(0.0, 1.0);
        double rand_gen = rng(generator) * sum_pheromones;
        int type;
        for ( type = 0; type < node_type_pheromones.size(); type++) {
            if ( rand_gen<=node_type_pheromones[type] )
              break;
            else
              rand_gen-= node_type_pheromones[type] ;
        }

        if ( edge_weight== 0.0 ) {
            uint16_t seed = std::chrono::system_clock::now().time_since_epoch().count();
            uniform_real_distribution<double> rng_5_5( -0.5, 0.5 ) ;
            edge_weight = rng_5_5( generator ) ;
        }
        else {
            edge_weight/=count_initialized_points ;
        }
        // for ( int i=0; i<node_weights.size(); i++ ) {
        //     node_weights[i]/= max(1,count_initialized_points) ;
        //     if ( node_weights[i] == 0.0 ) {
        //         node_weights[i] = rng_5_5( generator ) ;
        //     }
        // }

        c_x/=num_points ;
        c_y/=num_points ;

        POINT* point_added = NULL ;
        for ( POINT* point: m_points ) {
            if ( point->x == c_x && point->y == c_x )
            point_added = point ;
            break ;
        }
        if ( point_added == NULL ) {
            point_added = new POINT(c_x, c_y, m_type, m_level, INITIAL_PHEROMONE ) ;
            m_level_points[point_added->id] = point_added ;
        }

        point_added->edge_weight  = edge_weight  ;
        // point_added->node_weights[type] = node_weights ;
        point_added->type = type ;
        point_added->level = m_points[0]->level ;
        point_added->node_type_pheromones = node_type_pheromones ;
        centroids[c] = point_added ;
    }
    return 0;
}

int DBSCAN::expandCluster(POINT* point, int clusterID)
{
    vector<int> clusterSeeds = calculateCluster(point);

    if ( clusterSeeds.size() < m_minPoints )
    {
        point->clusterID = NOISE;
        return FAILURE;
    }
    else
    {
        int index = 0, indexCorePoint = 0;
        for( int s: clusterSeeds) {
            m_points[s]->clusterID = clusterID ;
            // cout << "Cluster ID INSIDE DBSCAN: " << clusterID << endl;

            if (m_points[s]->x == point->x && m_points[s]->y == point->y )
            {
                indexCorePoint = index;
            }
            ++index;
        }
        clusterSeeds.erase(clusterSeeds.begin()+indexCorePoint);

        for( vector<int>::size_type i = 0, n = clusterSeeds.size(); i < n; ++i )
        {
            vector<int> clusterNeighors = calculateCluster(m_points[clusterSeeds[i]]);

            if ( clusterNeighors.size() >= m_minPoints )
            {
                for ( int m: clusterNeighors )
                {
                    if ( m_points[n]->clusterID == UNCLASSIFIED || m_points[n]->clusterID == NOISE )
                    {
                        if ( m_points[n]->clusterID == UNCLASSIFIED )
                        {
                            clusterSeeds.push_back(m);
                            n = clusterSeeds.size();
                        }
                        m_points[n]->clusterID = clusterID;
                    }
                }
            }
        }
        return SUCCESS_;
    }
}

vector<int> DBSCAN::calculateCluster(POINT* point) {
    int index = 0;
    vector<int> clusterIndex;
    for (POINT* p: m_points)
    {
        if ( calculateDistance(point, p) <= m_epsilon )
        {
            clusterIndex.push_back(index);
        }
        index++;
    }
    return clusterIndex;
}

inline double DBSCAN::calculateDistance( POINT* pointCore, POINT* pointTarget )
{
    // cout << "DISTANCE: " << pow(pointCore->x - pointTarget->x,2)+pow(pointCore->y - pointTarget->y,2) << endl;
    return pow(pointCore->x - pointTarget->x,2)+pow(pointCore->y - pointTarget->y,2);
}
