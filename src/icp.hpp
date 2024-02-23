#include <iostream>
#include <numeric>
#include "Eigen/Eigen"
#include "icp_structs.hpp"
#include "nanoflann.hpp"

using namespace std;
using namespace Eigen;
using namespace nanoflann;

// Gets the magnitude of the distance vector between two vectors
float distance(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2) {
    return sqrt((v1[0] - v2[0])*(v1[0] - v2[0]) +
                (v1[1] - v2[1])*(v1[1] - v2[1]) +
                (v1[2] - v2[2])*(v1[2] - v2[2]) );
}

/*
    Uses a k(k=3) dimentionak tree to more efficiently search through the points to match two points as neightbors.
    Two points, p and q, are neighbors if point q, in the dst cloud, is the closest point to p in the dst cloud.

    Output is returned as NEIGHBOR which holds:
        > the matching indicies between the neighbors 
        > the distances between them.
    Ex:
    src[i] and dst[indicies[i]] are neighbor points and the ddistance between them is distances[i]
*/
NEIGHBOR kdNeighbor(const MatrixXd &src, const MatrixXd &dst) {
	int dim = 3;
	int leafSize = 10;
	int K = 1;
	int srcRow = src.rows();
	int dstRow = dst.rows();
	Vector3d srcVec;
	Vector3d dstVec;
	NEIGHBOR neighbor;

	// Setup the kdTree
	Matrix<float,Dynamic,Dynamic> dstMatrix(dstRow, dim);

	for (int i = 0; i < dstRow; i++) {
        for (int d = 0; d < dim; d++) {
		    dstMatrix(i,d) = dst(i,d);
        }
    }
		
	typedef KDTreeEigenMatrixAdaptor<Matrix<float,Dynamic,Dynamic> >kdtree_t;
	kdtree_t dstTree(dim,cref(dstMatrix),leafSize);
	dstTree.index->buildIndex();

    // Search by divisions
	for (int i=0; i < srcRow; i++)
	{
		vector<float> srcPoint(dim);
        vector<size_t> index(K);
		vector<float> distance(K);

		for(int d=0; d<dim; d++) {
			srcPoint[d] = src(i,d);
        }


		nanoflann::KNNResultSet<float> result_set(K);
		result_set.init(&index[0], &distance[0]);
		nanoflann::SearchParams params_ignored;
		dstTree.index->findNeighbors(result_set,&srcPoint[0],params_ignored);
		neighbor.indicies.push_back(index[0]);
		neighbor.distances.push_back(distance[0]);
	}
	return neighbor; 
}

/*
    getBestFitTransformation gives a mathematically ideal/perfect transformation from src to dst, 
    as long as the neighbor points are matched perfectly.

    Mathematically getBestFitTransform relies mainly on the single value decomposition of a matrix COveriance matrx H, 
    that is characterized by a homogenous transformatione from dst to src.

    Covariance of A and B, H, is the prouct of three, in this case, square matricies. H = USVᵀ. U and S vectors are rotation matricies
    while S is a diagonal, scaling matrix.

    Given H, two matricies:
        * HHᵀ = Sₗ , Left Symmetric Matrix of H
        * HᵀH = Sᵣ , Right Symmetric Matrix of H
    can be defined. They are called symmetric matricies becuase they are symmetric with respect to the main diagonal.
    Otherwise Sₗ = Sₗᵀ and Sᵣ = Sᵣᵀ.

    Let,
        * 𝒗₁, 𝒗₂, 𝒗₃ be the eigen vectors of Sₗ that correspond to the eigen values λ₁, λ₂, λ₃
        * 𝒖₁, 𝒖₂, 𝒖₃ be the eigen vectors of Sᵣ that correspond to the eigen values μ₁, μ₂, μ₃
    
    If, λ₁ ≥ λ₂ ≥ λ₃ and μ₁ ≥ μ₂ ≥ μ₃ then 
        *   λ₁ = μ₁
        *   λ₂ = μ₂
        *   λ₃ = μ₃
    
            and

        * √λ₁ = σ₁
        * √λ₂ = σ₂
        * √λ₃ = σ₃
    Where σ₁, σ₂ and σ₃ are the Singular Values of H that should satisfy σ₁ ≥ σ₂ ≥ σ₃.

    In which case svd(H) = USVᵀ, where:
            ╔            ╗         ╔            ╗       ╔            ╗
            ║ |   |   |  ║         ║ σ₁  0   0  ║       ║ |   |   |  ║
        U = ║ 𝒗₁  𝒗₂  𝒗₃  ║     S = ║ 0   σ₂  0  ║   V = ║ 𝒖₁  𝒖₂  𝒖₃ ║    
            ║ |   |   |  ║         ║ 0   0   σ₃ ║       ║ |   |   |  ║
            ╚            ╝         ╚            ╝       ╚            ╝

    Given these U, S and V; the rotation matrix from src to dst will be R = VUᵀ. This, for our sake should be 
    equivelent to H⁻¹. This is because there should be no scaling involved in our transformations, thus σ₁, σ₂, σ₃ = 1 (?)
    and so S = I₃. Making svd(H) = UI₃Vᵀ = UVᵀ. Thus [svd(H)]⁻¹ = [svd(H)]ᵀ = (UVᵀ)ᵀ = VUᵀ = R. 

    Because size change is not present, σ values will always be σ₁ = σ₂ = 1, σ₃ = 0. Because σ₃ = 0, there are an infinite possible decompostions of H.
    This doesn't matter for us as the end result will be the same.
*/
Eigen::Matrix4d getBestFitTransformation(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B) {
    // The sets of points kept as nx3 matricies A and B are short hand for the point clouds src and dst respectively

    int row = min(A.rows(), B.rows()); // TODO: Add code safety with min(A, B)? Check if neede

    /*
        T ∈ ℝ⁴ˣ⁴ is a homogeneuos transformation matrix s.t.
                ╔       ╗   Where:
                ║ R   t ║     * R ∈ ℝ³ˣ³, R = VUᵀ is the rotation matrix from src to dst
            T = ║       ║     * t ∈ ℝ³,   t = y₀ - Rx₀ is the translation vector from x₀ to y₀
                ║ 0   1 ║           > x₀ ∈ ℝ³, x₀ = (∑aₙ)/n is the centroid of all points in src
                ╚       ╝           > y₀ ∈ ℝ³, y₀ = (∑bₙ)/n is the centroid of all points in dst                               

    */ 
    Eigen::Matrix4d T = Eigen::MatrixXd::Identity(4,4);
    Eigen::Matrix3d R;
    Eigen::Vector3d t;

    Eigen::MatrixXd derA = A;
	Eigen::MatrixXd derB = B;

    Eigen::Vector3d centroidA(0,0,0);
	Eigen::Vector3d centroidB(0,0,0);

    Eigen::MatrixXd H;

    // SVD matricies
    Eigen::MatrixXd U;
    Eigen::MatrixXd S;
    Eigen::MatrixXd V;
    Eigen::MatrixXd Vt;

    // Get the centroid of src and dst
    for (int i = 0; i < row; i++) {
		centroidA += A.block<1,3>(i,0).transpose();
		centroidB += B.block<1,3>(i,0).transpose();
	}
    centroidA /= row;
    centroidB /= row;

    // Case when src and dst clouds have different sizes.
    // Keep thhe smallest set to ensure the matrix multiplicaation for the Covariance is defined.
    if(A.rows() > B.rows()) {
        derA = derB;
    }
    else {
        derB = derA;
    }

    // Get the derivations of A & B
    for (int i=0; i<row; i++) {
		derA.block<1,3>(i,0) = A.block<1,3>(i,0) - centroidA.transpose(); 
		derB.block<1,3>(i,0) = B.block<1,3>(i,0) - centroidB.transpose();
	}
    
    // Get covariance of A and B
    H = derA.transpose()*derB;

    Eigen::JacobiSVD<MatrixXd> svd(H, ComputeFullU|ComputeFullV);
    U = svd.matrixU();
    S = svd.singularValues(); // Should be the identity (?)
    V = svd.matrixV();
    Vt = V.transpose();

    // Get the rotation matrix from src to dst
    R = (U*Vt).transpose();

    // Check the determinant of R to make sure we rotated and didn't reflect instead.
    if(R.determinant() < 0) {
        Vt.block<1,3>(2,0) *= -1;
        R = (U*Vt).transpose();
    }

    // Get translation from centroidA to centroidB
    t = centroidB - R*centroidA;

    // Get translation from centroidA to centroidB
    T.block<3,3>(0,0) = R;
    T.block<3,1>(0,3) = t;

    // Transformation clean up. Set very very small entries of the transformation to 0
    for(int i = 0; i < T.rows(); i++) {
        for(int j = 0; j < T.cols(); j++) {
            if(abs(T(i,j)) <= 0.0001) T(i,j) = 0;
        }
    }

    return T;
}

// Main iterative loop algorithm
ICP_OUT icp( const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, double threshold, int maxIterationNumber ) {
    ICP_OUT result; 

    Matrix4d T;

    NEIGHBOR neighbor;

    int row = min(A.rows(), B.rows());

    // These matricies hold column vectors not row
    MatrixXd src = MatrixXd::Ones(4, row);
    MatrixXd src3d = MatrixXd::Ones(3, row);
    MatrixXd dst = MatrixXd::Ones(4, row);
    MatrixXd dstOrd = MatrixXd::Ones(3, row);

    int iter = 0;

    double err = 0;
    double meanErr = 0;

    for(int i = 0; i < row; i++) {
        // src and src3d are the same but src has an extra row so that T can be applied to src. (T is 4x4)
        src.block<3,1>(0,i) = A.block<1,3>(i,0).transpose();     
		src3d.block<3,1>(0,i) = A.block<1,3>(i,0).transpose();  
		dst.block<3,1>(0,i) = B.block<1,3>(i,0).transpose();
    }

    // Start the iteration loop
    for(iter = 0; iter < maxIterationNumber; iter++) {
        neighbor = kdNeighbor(src3d.transpose(), B); // src3d has column vectors while B has row vectors, so src3d has to be transposed.

        for (int j = 0; j < row; j++) {
            dstOrd.block<3,1>(0,j) = dst.block<3,1>(0,neighbor.indicies[j]);
        }

        // Guess a transformation with the semi ordered dstOrd
        T = getBestFitTransformation(src3d.transpose(), dstOrd.transpose());

        // Update src points
        src = T*src;

        // Reconstruct src3d as the first three rows of the new src points
        for (int j=0; j<row; j++) {
			src3d.block<3,1>(0,j) = src.block<3,1>(0,j);
        }

        meanErr = 0.0f;
		for (int i=0; i<neighbor.distances.size();i++) {
            meanErr += neighbor.distances[i];
        }
		meanErr /= neighbor.distances.size();
		if (abs(err - meanErr) < threshold) {
			break;
        }
		err = meanErr;
    }

    /*
        > A is the initial point  cloud configuration.
        > src3d is the best approximation given the constants to the dst
        Best, transformation approxiamtion will be the transformation from A to src3d.
        This gets us the final homogenous transformation.
    */
    T = getBestFitTransformation(A, src3d.transpose());

    // Set up the return variable.
    result.transformation = T;
    result.iterations = iter;
    result.distances = neighbor.distances;

    return result;
}