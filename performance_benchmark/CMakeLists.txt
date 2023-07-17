cmake_minimum_required(VERSION 3.1)
include($ENV{BOB_ROBOTICS_PATH}/cmake/bob_robotics.cmake)

BoB_project(IS_EXPERIMENT TRUE
            EXECUTABLE performance_benchmark
            SOURCES performance_benchmark.cc
            BOB_MODULES common hid imgproc navigation viz)
