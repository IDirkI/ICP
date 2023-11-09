#include <iostream>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/filter_indices.h>
#include <pcl/common/transforms.h>

#include "Eigen/Eigen"
#include "icp.hpp"

using namespace std;
using namespace pcl;

// Function to load files in the data directory
void loadFile(const char* file_name, pcl::PointCloud<pcl::PointXYZ> &pc) {
	pcl::PolygonMesh mesh;

	if(pcl::io::loadPLYFile(file_name, mesh)==-1) {
		PCL_ERROR("File loading faild.");
		return;
	} 
    else{
		pcl::fromPCLPointCloud2<pcl::PointXYZ>(mesh.cloud, pc);
	}

	vector<int> index;
	pcl::removeNaNFromPointCloud(pc, pc, index);
}

int main(int, char**argv){
		
	pcl::PointCloud<pcl::PointXYZ>::Ptr srcPointCloud (new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr dstPointCloud (new pcl::PointCloud<pcl::PointXYZ>());


	loadFile(argv[1], *srcPointCloud);
	loadFile(argv[2], *dstPointCloud);


	pcl::PointCloud<pcl::PointXYZ>::Ptr srcPointCloudTransformation (new pcl::PointCloud<pcl::PointXYZ>());

	{
		Eigen::MatrixXf srcMatrix = cloud_source->getMatrixXfMap(3,4,0).transpose();
		Eigen::MatrixXf dstMatrix = cloud_target->getMatrixXfMap(3,4,0).transpose();
		
		int maxIterationNumber = 10;
		float threshold = 0.000001;

		// call icp
		ICP_OUT result = icp(srcMatrix.cast<double>(), dstMatrix.cast<double>(), maxIterationNumber, threshold);

		int iter = result.iterations;	
		Matrix4f T = result.transformation.cast<float>();	
		vector<float> distances = result.distances;

		Eigen::MatrixXf transformationmatrix = srcMatrix;
		
		int row = srcMatrix.rows();
		MatrixXf transform4d = MatrixXf::Ones(3+1,row);

		for(int i=0;i<row;i++) {
			transform4d.block<3,1>(0,i) = srcMatrix.block<1,3>(i,0).transpose();
		}

		transform4d = T*transform4d;

		for(int i=0;i<row;i++) {
			source_trans_matrix.block<1,3>(i,0) = transform4d.block<3,1>(0,i).transpose();
		}

		pcl::PointCloud<pcl::PointXYZ> newCloud;
		newCloud.width = row;
		newCloud.height = 1;
		newCloud.points.resize(row);
		
		for (size_t n=0; n<row; n++) 
		{
			newCloud[n].x = source_trans_matrix(n,0);
			newCloud[n].y = source_trans_matrix(n,1);
			newCloud[n].z = source_trans_matrix(n,2);	
  		}

  		srcPointCloudTransformation = newCloud.makeShared();

		cout << result.transformation << endl;
  
	}
	return(0);

}
