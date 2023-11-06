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

    dstPc <<  1, 0,  0,
              0,  1,  0,
              0,  0,  1;

    Eigen::MatrixXd iPc = srcPc.transpose();
    Eigen::MatrixXd fPc = dstPc.transpose();

    ICP_OUT output = icp(iPc, fPc, 0.000001, 10);

    cout << output.transformation << endl;
}
