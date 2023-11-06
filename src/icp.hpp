#include <iostream>
#include <numeric>
#include "Eigen/Eigen"
#include "icp_structs.hpp"

using namespace std;
using namespace Eigen;

/* Gets the magnitude of the distance vector between two vectors */ 
float distance(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2) {
    return sqrt((v1[0] - v2[0])*(v1[0] - v2[0]) +
                (v1[1] - v2[1])*(v1[1] - v2[1]) +
                (v1[2] - v2[2])*(v1[2] - v2[2]) );
}

/*
    Loops through all points in the src point cloud, finding the nearest point to it in the dst point cloud.

    Store the distance between the neighbor src points and dst points along with the matching indicies

    ie. the 0th index of the indicies vector hold the index of the closest neighbor in dst of the 0th vector in src.
*/
NEIGHBOR getClosestNeighbor(const Eigen::MatrixXd &src, const Eigen::MatrixXd &dst) {
    NEIGHBOR neighbor;

    Eigen::Vector3d srcVec;
    Eigen::Vector3d dstVec;

    int srcRow = src.rows();
    int dstRow = dst.rows();

    float minDistance = 100;
    float dist = 0;
    int index = 0;

    for(int i = 0; i < srcRow; i++) {
        srcVec = src.block<1,3>(i,0).transpose();

        float minDistance = 100;
        float dist = 0;
        int index = 0;

        for(int j = 0; j < dstRow; j++) {
            dstVec = dst.block<1,3>(j,0).transpose();
            dist = distance(srcVec, dstVec);
            
            if( dist < minDistance ) {
                minDistance = dist;
                index = j;
            }
        }

        neighbor.distances.push_back(minDistance);
        neighbor.indicies.push_back(index);
    }

    return neighbor;
}

/*
    getBestFitTransformation gives a mathematically ideal/perfect transformation from src to dst, 
    as long as the neighbor points are matched perfectly.

    Mathematically getBestFitTransform relies mainly on the single value decomposition of a matrix H, 
    that is characterized by the transformation of points from dst to src.

    H is the prouct of three, in this case, square matricies. H = USVᵀ. U and S vectors are rotation matricies
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
        U = ║ 𝒗₁  𝒗₂  𝒗₃ ║     S = ║ 0   σ₂  0  ║   V = ║ 𝒖₁  𝒖₂  𝒖₃║    
            ║ |   |   |  ║         ║ 0   0   σ₃ ║       ║ |   |   |  ║
            ╚            ╝         ╚            ╝       ╚            ╝

    Given these U, S and V; the rotation matrix from src to dst will be R = VUᵀ. This, for our sake should be 
    equivelent to H⁻¹. This is because there should be no scaling involved in our transformations, thus σ₁, σ₂, σ₃ = 1 (?)
    and so S = I₃. Making svd(H) = UI₃Vᵀ = UVᵀ. Thus [svd(H)]⁻¹ = [svd(H)]ᵀ = (UVᵀ)ᵀ = VUᵀ = R. 
*/
Eigen::Matrix4d getBestFitTransformation(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B) {
    // The sets of points kept as nx3 matricies A and B are short hand for the point clouds src and dst respectively

    int row = A.rows(); // TODO: Add code safety with min(A, B)? Check if needed

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

    /*
        The only purpose of these matricies are to calculate the matrix H
    */
    Eigen::MatrixXd AA = A; // Assignment here is meaningless as AA is overwritten later. This is just to get AA to the correct size.
	Eigen::MatrixXd BB = B; // Assignment here is meaningless as BB is overwritten later. This is just to get BB to the correct size.

    Eigen::Vector3d centroidA(0,0,0);
	Eigen::Vector3d centroidB(0,0,0);

    // SVD matricies
    Eigen::MatrixXd H;

    Eigen::MatrixXd U;
    Eigen::MatrixXd S;
    Eigen::MatrixXd V;
    Eigen::MatrixXd Vt;

    // Get the centroid of src and dst
    for (int i=0; i<row; i++)
	{
		centroidA += A.block<1,3>(i,0).transpose();
		centroidB += B.block<1,3>(i,0).transpose();
	}
    centroidA/= row;
    centroidB /= row;

    // Calculating H = ∑(bₙ - b₀)(aₙ - a₀)ᵀ
    for (int i=0; i<row; i++)
	{
		AA.block<1,3>(i,0) = A.block<1,3>(i,0) - centroidA.transpose(); 
		BB.block<1,3>(i,0) = B.block<1,3>(i,0) - centroidB.transpose();
	}
    /*
        Here the order of multiplication is not equal to the definition because we hold vectors in AA and BB
        as row vectors while the equation: H = ∑(bₙ - b₀)(aₙ - a₀)ᵀ assumes column vectors.
    */
    H = AA.transpose()*BB;

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

    int row = A.rows();
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
        neighbor = getClosestNeighbor(src3d.transpose(), B); // src3d has column vectors while B has row vectors, so src3d has to be transposed.

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
        cout << abs(err - meanErr) << endl;
		if (abs(err - meanErr) < threshold) {
			break;
        }
		err = meanErr;
    }

    T = getBestFitTransformation(A, src3d.transpose());

    result.transformation = T;
    result.iterations = iter;
    result.distances = neighbor.distances;

    return result;
}