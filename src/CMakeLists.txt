# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

set(CMAKE_CXX_STANDARD 14)

if(MSVC)
	add_subdirectory(dll)
else()
	#file(GLOB CPP_SOURCES *.cpp)
	set(CPP_SOURCES ann_exception.cpp recall_utils.cpp index.cpp graph_index.cpp
        math_utils.cpp 
        partition_and_pq.cpp utils.cpp vamana.cpp relative_ng.cpp hnsw.cpp ivf.cpp lsh.cpp)
	add_library(${PROJECT_NAME} ${CPP_SOURCES})
	add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})
endif()
install()
