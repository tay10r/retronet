cmake_minimum_required(VERSION 3.14.7)

project(retronet)

add_library(retronet
  retronet.h
  retronet.cpp)
target_include_directories(retronet
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}")

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  add_executable(example
    example/main.cpp
    example/stb_image_write.h
    example/stb_image_write.c)
  target_link_libraries(example
    PUBLIC
      retronet)
endif()
