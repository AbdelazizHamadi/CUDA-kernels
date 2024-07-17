cmake_minimum_required(VERSION 3.18)
project(MyCudaProject LANGUAGES CXX CUDA)

# Specify the CUDA standard
set(CMAKE_CUDA_STANDARD 11)

# Find OpenCV package
find_package(PkgConfig REQUIRED)
pkg_check_modules(OpenCV REQUIRED opencv)

# Add the source file(s)
add_executable(cuda_program src/matrix_mul.cu  src/matrix_mul_outside.cpp)

# Link OpenCV libraries
target_link_libraries(cuda_program PRIVATE ${OpenCV_LIBRARIES})
target_include_directories(cuda_program PRIVATE ${OpenCV_INCLUDE_DIRS})

# Enable CUDA support
enable_language(CUDA)

