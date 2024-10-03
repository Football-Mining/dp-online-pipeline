cmake_minimum_required(VERSION 3.10)
project(EquirectangularProject CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Specify the path to your Conda environment
set(CONDA_ENV_PATH "$ENV{HOME}/miniforge3/envs/details")

# Set the OpenCV_DIR to the OpenCV installation in your Conda environment
set(OpenCV_DIR "${CONDA_ENV_PATH}/lib/cmake/opencv4")

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the Conda environment's lib directory to the runtime path
set(CMAKE_INSTALL_RPATH "${CONDA_ENV_PATH}/lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# Option for CUDA support
option(USE_CUDA "Enable CUDA support" OFF)

if(USE_CUDA)
    # Find CUDA
    find_package(CUDA)
    if(CUDA_FOUND)
        enable_language(CUDA)
        include_directories(${CUDA_INCLUDE_DIRS})
        add_definitions(-DUSE_CUDA)
    else()
        message(WARNING "CUDA requested but not found. Building without CUDA support.")
    endif()
endif()

# Add executable
add_executable(equirectangular_app src/equirectangular.cpp)

# Link libraries
target_link_libraries(equirectangular_app ${OpenCV_LIBS})

# Explicitly link against libstdc++ from the Conda environment
target_link_libraries(equirectangular_app "${CONDA_ENV_PATH}/lib/libstdc++.so")

if(USE_CUDA AND CUDA_FOUND)
    target_link_libraries(equirectangular_app ${CUDA_LIBRARIES})
    set_target_properties(equirectangular_app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()