cmake_minimum_required(VERSION 3.25)
project(cadre_power LANGUAGES CXX)

find_package(amigo REQUIRED CONFIG)

if(AMIGO_ENABLE_CUDA)
    enable_language(CUDA) 
endif()

amigo_add_python_module(
    NAME cadre_power
    SOURCES cadre_power.cpp
)
