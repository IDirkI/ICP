#include <vector>
#include "Eigen/Eigen"

#ifndef STRUCTS_H
#define STRUCTS_H

typedef struct {
    Eigen::Matrix4d transformation;
    std::vector<float> distances;
    int iterations;
} ICP_OUT;

typedef struct{
	std::vector<int> indicies;
	std::vector<float> distances;
	float mean_distance;
} NEIGHBOR;

#endif