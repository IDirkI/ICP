#include <iostream>
#include "Eigen/Eigen"
#include "icp.hpp"

using namespace std;
using namespace Eigen;

int main(int, char**){
    Eigen::MatrixXd srcPc(3,3);
    Eigen::MatrixXd dstPc(3,3);
    
    srcPc <<  1,  0,  0,
              0,  1,  0,
              0,  0,  1;

    dstPc <<  0.965926,  0,  0.258819,
              4.5,  5.5,  4.5,
              -0.258819,  0,  0.965926;

    Eigen::MatrixXd iPc = srcPc.transpose();
    Eigen::MatrixXd fPc = dstPc.transpose();

    Eigen::Matrix4d output = getBestFitTransformation(iPc, fPc);

    cout<< output << endl;
}
