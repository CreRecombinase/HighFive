project(HighFive)
cmake_minimum_required(VERSION 3.0)

enable_testing()

set(HIGHFIVE_VERSION_MAJOR 1)
set(HIGHFIVE_VERSION_MINOR 2)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake
  ${PROJECT_SOURCE_DIR}/CMake/portability
  ${PROJECT_SOURCE_DIR}/CMake/packages
  ${PROJECT_SOURCE_DIR}/CMake/config)

include(DefineInstallationPaths)
include(ReleaseDebugAutoFlags)
include(CheckCXX11Portability)

include(BlueGenePortability)

option(UNIT_TESTS "enable unit tests" TRUE)
option(USE_BOOST "enable Boost Support" TRUE)
option(HIGHFIVE_EXAMPLES "Compile examples" TRUE)
option(HIGHFIVE_PARALLEL_HDF5 "Enable Parallel HDF5 support" FALSE)
option(USE_EIGEN "Enable Eigen support" TRUE)

if(HIGHFIVE_PARALLEL_HDF5)
set(HDF5_PREFER_PARALLEL TRUE)
endif()

find_package(HDF5 QUIET REQUIRED)
if(USE_EIGEN)
  
  find_package (Eigen3 3.3 REQUIRED NO_MODULE)
  add_definitions(-DH5_USE_EIGEN)
  include_directories ( ${EIGEN3_INCLUDE_DIR} )
#  include_directories(SYSTEM ${Eigen3_INCLUDE_DIRS})
endif()

set(Boost_NO_BOOST_CMAKE TRUE)

if(USE_BOOST)
  find_package(Boost QUIET REQUIRED COMPONENTS system unit_test_framework)
  add_definitions(-DH5_USE_BOOST )
  include_directories(SYSTEM ${Boost_INCLUDE_DIR})
endif()

if(HIGHFIVE_PARALLEL_HDF5)
  find_package(MPI)
  include_directories(MPI_C_INCLUDE_PATH)
endif()

# Absolute include used to avoid interaction with downstream projects including
# HighFive as a subproject and using the the same filename
include(${PROJECT_SOURCE_DIR}/CMake/PackageConfig.cmake)

if(HIGHFIVE_EXAMPLES)
  add_subdirectory(src/examples)
endif()

add_subdirectory(tests/unit)
