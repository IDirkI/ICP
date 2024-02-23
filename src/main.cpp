#include <iostream>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
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

void loadFile(const char* fileName, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudIn) {
	pcl::PointCloud<pcl::PointXYZ> cloudOut;
	pcl::PolygonMesh mesh;

	// File load error catch
	if(pcl::io::loadPLYFile(fileName, mesh) == -1) {
		PCL_ERROR("File could not be loaded  due to an error.");
		return;
	} 
    
	// Get point cloud from file mesh. Written to the adress of cloudIn.
	pcl::fromPCLPointCloud2<pcl::PointXYZ>(mesh.cloud, *cloudIn);

	// Manual NaN check and rewrite to the adress of cloudIn. IF point clouds contain NaN values the output will be garbage.
	for(int i = 0; i < cloudIn->points.size(); i++) {
		if((!(isnan(cloudIn->points[i].x)) && !(isnan(cloudIn->points[i].y))) && !(isnan(cloudIn->points[i].z))) {
			cloudOut.push_back(cloudIn->points[i]);
		}
	}

	*cloudIn = cloudOut;
}

void icpStart(char** argv) {
	// Configuration constants.
	/*
	> Higher MAX_ITERATIONS ---> longer wait time, 'possibly' more acctuarte results
	> Lower  THRESHOLD	---> longer wait time, most likely betterr accracy.
	*/
	const int MAX_ITERATIONS = 10;
	const float THRESHOLD = 0.000001;
		
	// Load thhe files  into point cloud instances.
	pcl::PointCloud<pcl::PointXYZ>::Ptr srcPointCloud (new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr dstPointCloud (new pcl::PointCloud<pcl::PointXYZ>());

	loadFile(argv[1], srcPointCloud);
	loadFile(argv[2], dstPointCloud);

	// Produce nx3 matrix containing the points as row vectors.
	Eigen::MatrixXf srcMatrix = srcPointCloud->getMatrixXfMap(3,4,0).transpose();
	Eigen::MatrixXf dstMatrix = dstPointCloud->getMatrixXfMap(3,4,0).transpose();

	// Call the actual ICP loop and stroe the ICP_OUT output
	ICP_OUT result = icp(srcMatrix.cast<double>(), dstMatrix.cast<double>(), THRESHOLD, MAX_ITERATIONS);

	// Useful outputs from the ICP loop
	int iter = result.iterations;	
	Matrix4f T = result.transformation.cast<float>();	
	vector<float> distances = result.distances;

	// Homogenous TRansformation matrix from src to dst outputs
	cout << result.transformation << endl;
	cout << result.iterations << endl;


	// Visualization Code
	pcl::PointCloud<pcl::PointXYZ>::Ptr srcPointCloudTransformation (new pcl::PointCloud<pcl::PointXYZ>());

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
		// Background
		boost::shared_ptr <pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
		viewer->setBackgroundColor(255,255,255);

		// Source cloud [Black]
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color(srcPointCloud,0,0,0);
		viewer->addPointCloud<pcl::PointXYZ>(srcPointCloud,source_color,"source");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,1,"source");

		// Target cloud [Blue
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(dstPointCloud,0,0,255);
		viewer->addPointCloud<pcl::PointXYZ>(dstPointCloud,target_color,"target");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,1,"target");

		// Transformed Source [Red]
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_trans_color(srcPointCloudTransformation,255,0,0);
		viewer->addPointCloud<pcl::PointXYZ>(srcPointCloudTransformation,source_trans_color,"source trans");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,1,"source trans");

		viewer->getRenderWindow()->GetRenderers()->GetFirstRenderer()->GetActiveCamera()->SetParallelProjection(1);
		viewer->resetCamera();
		viewer->spin();
}


int main(int, char** argv) {
	icpStart(argv);
	return 0;
}
