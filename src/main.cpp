#include <iostream>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/filter_indices.h>
#include <pcl/common/transforms.h>

#include <vtkRenderWindow.h>
#include <vtkRendererCollection.h>
#include <vtkCamera.h>

#include "Eigen/Eigen"
#include "icp.hpp"

using namespace std;
//using namespace pcl;

// Function to load files in the data directory

void loadFile(const char* fileName, pcl::PointCloud<pcl::PointXYZ> &pc) {
	pcl::PolygonMesh mesh;
	if(pcl::io::loadPLYFile(fileName, mesh)==-1) {
		PCL_ERROR("File loading faild.");
		return;
	} 
    else {
		pcl::fromPCLPointCloud2<pcl::PointXYZ>(mesh.cloud, pc);
	}

	vector<int> index;
	pcl::removeNaNFromPointCloud(pc, pc, index);
}

int main(int, char**argv){
	const int MAX_ITERATIONS = 10;
	const float THRESHOLD = 0.000001;
		
	pcl::PointCloud<pcl::PointXYZ>::Ptr srcPointCloud (new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr dstPointCloud (new pcl::PointCloud<pcl::PointXYZ>());


	loadFile(argv[1], *srcPointCloud);
	loadFile(argv[2], *dstPointCloud);


	pcl::PointCloud<pcl::PointXYZ>::Ptr srcPointCloudTransformation (new pcl::PointCloud<pcl::PointXYZ>());


	Eigen::MatrixXf srcMatrix = srcPointCloud->getMatrixXfMap(3,4,0).transpose();
	Eigen::MatrixXf dstMatrix = dstPointCloud->getMatrixXfMap(3,4,0).transpose();

	cout << srcMatrix << endl;

	ICP_OUT result = icp(srcMatrix.cast<double>(), dstMatrix.cast<double>(), THRESHOLD, MAX_ITERATIONS);

	int iter = result.iterations;	
	Matrix4f T = result.transformation.cast<float>();	
	vector<float> distances = result.distances;

	Eigen::MatrixXf transformationMatrix = srcMatrix;
	
	int row = srcMatrix.rows();
	MatrixXf transform4d = MatrixXf::Ones(3+1,row);

	for(int i = 0; i  < row; i++) {
		transform4d.block<3,1>(0,i) = srcMatrix.block<1,3>(i,0).transpose();
	}

	transform4d = T*transform4d;

	for(int i=0;i<row;i++) {
		transformationMatrix.block<1,3>(i,0) = transform4d.block<3,1>(0,i).transpose();
	}

	pcl::PointCloud<pcl::PointXYZ> newCloud;

	newCloud.width = row;
	newCloud.height = 1;

	newCloud.points.resize(row);
	
	for (int n=0; n<row; n++) {
		newCloud[n].x = transformationMatrix(n,0);
		newCloud[n].y = transformationMatrix(n,1);
		newCloud[n].z = transformationMatrix(n,2);	
	}

	srcPointCloudTransformation = newCloud.makeShared();

	cout << result.transformation << endl;
	cout << result.iterations << endl;

		{ // visualization
		boost::shared_ptr <pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
		viewer->setBackgroundColor(255,255,255);

		// black
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color(srcPointCloud,0,0,0);
		viewer->addPointCloud<pcl::PointXYZ>(srcPointCloud,source_color,"source");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,1,"source");

		// blue
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(dstPointCloud,0,0,255);
		viewer->addPointCloud<pcl::PointXYZ>(dstPointCloud,target_color,"target");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,1,"target");

		// red
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_trans_color(srcPointCloudTransformation,255,0,0);
		viewer->addPointCloud<pcl::PointXYZ>(srcPointCloudTransformation,source_trans_color,"source trans");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,1,"source trans");

		viewer->getRenderWindow()->GetRenderers()->GetFirstRenderer()->GetActiveCamera()->SetParallelProjection(1);
		viewer->resetCamera();
		viewer->spin();
	}
	
	return(0);

}
