#include <iostream>
#include "matrix_mul.cuh"  // Include the CUDA header file
#include <opencv2/opencv.hpp>
#include <chrono>
#include <numeric>
#define ROUNDS 5

int main(int argc, char *argv[]) {

    // Define the matrix dimensions
    // const int X = 16384, Y = 16384, Z = 16384;
    int X, Y, Z;

    // Initialize matrices A and B
    cv::Mat A = cv::Mat::ones(X, Y, CV_32F);
    cv::Mat B = cv::Mat::ones(Y, Z, CV_32F);
    cv::Mat C = cv::Mat::ones(X, Z, CV_32F);

    // Fill matrices A and B with random values from 0 to 10
    // cv::randu(A, cv::Scalar(0), cv::Scalar(10));
    // cv::randu(B, cv::Scalar(0), cv::Scalar(10));

    std::array<int, 5> sizes = {2048, 4096, 8192, 12288, 128*128};
    std::array<float, ROUNDS> time_per_size;

    for (int i=0; i<sizes.size(); i++){
        

        std::cout << "executing cuda Kernel (" << i << ")..." << std::endl;
        X = sizes[i];
        Y = sizes[i];
        Z = sizes[i];
        std::cout << "A shape: (" << X << ", " << Y << ")" << std::endl;
        std::cout << "B shape: (" << Y << ", " << Z << ")" << std::endl;
        
        // create time
        float time = 0.0f;
        bool print = true;
        for (int j = 0; j < ROUNDS; ++j){

            MatMul(A.ptr<float>(), B.ptr<float>(), C.ptr<float>(), X, Y, Z, &time, print);
            time_per_size[j] = time;
            print = false;

        }

        float sum = std::accumulate(time_per_size.begin(), time_per_size.end(), 0.0f);

        std::cout << "\nCUDA kernels execution complete in : " << sum / time_per_size.size() << std::endl;
        /*
        // Print a portion of the result matrix
        std::cout << "Result matrix C (first [4, 4]):" << std::endl;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
            std::cout << C.at<float>(i, j) << " ";
        }
        std::cout << std::endl;
    }*/
        
    }
    


    // Print a portion of the result matrix
    return 0;
}
