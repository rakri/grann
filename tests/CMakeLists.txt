# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

set(CMAKE_CXX_STANDARD 14)

add_executable(build_vamana build_vamana.cpp )
if(MSVC)
	target_link_options(build_vamana PRIVATE /MACHINE:x64 /DEBUG:FULL "/INCLUDE:_tcmalloc")
	target_link_libraries(build_vamana debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/grann_dll.lib ${PROJECT_SOURCE_DIR}/dependencies/windows/tcmalloc/libtcmalloc_minimal.lib)
	target_link_libraries(build_vamana optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/grann_dll.lib ${PROJECT_SOURCE_DIR}/dependencies/windows/tcmalloc/libtcmalloc_minimal.lib)
else() 
	target_link_libraries(build_vamana ${PROJECT_NAME} -ltcmalloc)
endif()


add_executable(build_hnsw build_hnsw.cpp )
if(MSVC)
	target_link_options(build_hnsw PRIVATE /MACHINE:x64 /DEBUG:FULL "/INCLUDE:_tcmalloc")
	target_link_libraries(build_hnsw debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/grann_dll.lib ${PROJECT_SOURCE_DIR}/dependencies/windows/tcmalloc/libtcmalloc_minimal.lib)
	target_link_libraries(build_hnsw optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/grann_dll.lib ${PROJECT_SOURCE_DIR}/dependencies/windows/tcmalloc/libtcmalloc_minimal.lib)
else() 
	target_link_libraries(build_hnsw ${PROJECT_NAME} -ltcmalloc)
endif()


add_executable(build_relng build_relng.cpp )
if(MSVC)
	target_link_options(build_relng PRIVATE /MACHINE:x64 /DEBUG:FULL "/INCLUDE:_tcmalloc")
	target_link_libraries(build_relng debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/grann_dll.lib ${PROJECT_SOURCE_DIR}/dependencies/windows/tcmalloc/libtcmalloc_minimal.lib)
	target_link_libraries(build_relng optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/grann_dll.lib ${PROJECT_SOURCE_DIR}/dependencies/windows/tcmalloc/libtcmalloc_minimal.lib)
else() 
	target_link_libraries(build_relng ${PROJECT_NAME} -ltcmalloc)
endif()


add_executable(build_ivf build_ivf.cpp )
if(MSVC)
	target_link_options(build_ivf PRIVATE /MACHINE:x64 /DEBUG:FULL "/INCLUDE:_tcmalloc")
	target_link_libraries(build_ivf debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/grann_dll.lib ${PROJECT_SOURCE_DIR}/dependencies/windows/tcmalloc/libtcmalloc_minimal.lib)
	target_link_libraries(build_ivf optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/grann_dll.lib ${PROJECT_SOURCE_DIR}/dependencies/windows/tcmalloc/libtcmalloc_minimal.lib)
else() 
	target_link_libraries(build_ivf ${PROJECT_NAME} -ltcmalloc)
endif()


add_executable(search_vamana search_vamana.cpp )
if(MSVC)
	target_link_options(search_vamana PRIVATE /MACHINE:x64 /DEBUG:FULL)
	target_link_libraries(search_vamana debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/grann_dll.lib)
	target_link_libraries(search_vamana optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/grann_dll.lib)
else() 
		target_link_libraries(search_vamana ${PROJECT_NAME} aio -ltcmalloc)
endif()

add_executable(search_hnsw search_hnsw.cpp )
if(MSVC)
	target_link_options(search_hnsw PRIVATE /MACHINE:x64 /DEBUG:FULL)
	target_link_libraries(search_hnsw debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/grann_dll.lib)
	target_link_libraries(search_hnsw optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/grann_dll.lib)
else() 
		target_link_libraries(search_hnsw ${PROJECT_NAME} aio -ltcmalloc)
endif()


add_executable(search_relng search_relng.cpp )
if(MSVC)
	target_link_options(search_relng PRIVATE /MACHINE:x64 /DEBUG:FULL)
	target_link_libraries(search_relng debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/grann_dll.lib)
	target_link_libraries(search_relng optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/grann_dll.lib)
else() 
		target_link_libraries(search_relng ${PROJECT_NAME} aio -ltcmalloc)
endif()


add_executable(search_ivf search_ivf.cpp )
if(MSVC)
	target_link_options(search_ivf PRIVATE /MACHINE:x64 /DEBUG:FULL)
	target_link_libraries(search_ivf debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/grann_dll.lib)
	target_link_libraries(search_ivf optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/grann_dll.lib)
else() 
		target_link_libraries(search_ivf ${PROJECT_NAME} aio -ltcmalloc)
endif()
