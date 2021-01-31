/*
This code is originaly implemented by james-yoo
https://github.com/james-yoo/DBSCAN
Modified by A.ElSaid
*/
#ifndef DBSCAN_H
#define DBSCAN_H

#include <chrono>
#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <vector>
#include <cmath>
#include "point.hxx"
#include <iostream>
using std::pair ;
#define UNCLASSIFIED -1
#define CORE_POINT 1
#define BORDER_POINT 2
#define NOISE -2
#define SUCCESS_ 0
#define FAILURE -3

#include <map>
using std::map ;

using namespace std;

#define INITIAL_PHEROMONE    1.0


// typedef struct Point_
// {
//     float x, y, z;  // X, Y, Z position
//     int clusterID;  // clustered ID
// }Point;

class DBSCAN {
public:
    DBSCAN(unsigned int minPts, float eps, vector<POINT*> points, int level, map<int32_t, POINT*> &level_points ){
        m_minPoints = minPts;
        m_epsilon = eps;
        m_points = points;
        m_pointSize = points.size();
        m_level = level ;
        m_level_points = level_points ;
        m_type = points[0]->type ;
        uint16_t seed = std::chrono::system_clock::now().time_since_epoch().count();
        generator = minstd_rand0(seed);

    }
    ~DBSCAN(){}

    int run();
    vector<int> calculateCluster(POINT* point);
    int expandCluster(POINT* point, int clusterID);
    inline double calculateDistance(POINT* pointCore, POINT* pointTarget);

    int getTotalPointSize() {return m_pointSize;}
    int getMinimumClusterSize() {return m_minPoints;}
    int getEpsilonSize() {return m_epsilon;}
    map <int, POINT* > centroids ;

    minstd_rand0 generator;


private:
    vector<POINT*> m_points;
    unsigned int m_pointSize;
    unsigned int m_minPoints;
    float m_epsilon;
    int m_level;
    map<int32_t, POINT*> m_level_points;
    int m_type ;
};

#endif // DBSCAN_H
