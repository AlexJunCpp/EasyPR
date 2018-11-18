#CMakeLists.txt由命令、注释和空格组成，其中命令是不区分大小写。符号#后的内容被认为是注释。命令由命令名称、小括号和参数组成，参数之间使用空格进行间隔。
#CMake是一个比Make工具更高级的编译配置工具，是一个跨平台的、开源的构建系统（BuildSystem）


cmake_minimum_required(VERSION 3.0.0)
project(easypr)
#*leijun 这里大小写都可以

# c++11 required
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/local/opt/opencv-3.2.0")
endif ()

# OpenCV3 required
find_package(OpenCV 3.2.0 REQUIRED)

# where to find header files
include_directories(.)
include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})
#?leijun  这个路径如何找到的？

# sub directories
add_subdirectory(thirdparty)

# sources to be compiled
set(SOURCE_FILES
        src/core/core_func.cpp
        src/core/chars_identify.cpp
        src/core/chars_recognise.cpp
        src/core/chars_segment.cpp
        src/core/feature.cpp
        src/core/plate_detect.cpp
        src/core/plate_judge.cpp
        src/core/plate_locate.cpp
        src/core/plate_recognize.cpp
        src/core/params.cpp

        src/train/ann_train.cpp
        src/train/annCh_train.cpp
        src/train/svm_train.cpp
        src/train/train.cpp
        src/train/create_data.cpp

        src/util/util.cpp
        src/util/program_options.cpp
        src/util/kv.cpp
        )

# pack objects to static library
add_library(easypr STATIC ${SOURCE_FILES})

if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(EXECUTABLE_NAME "demo")
elseif (CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(EXECUTABLE_NAME "demo")
endif ()

# set to be releas mode
# set(CMAKE_BUILD_TYPE Release)

#*leijun
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

#添加要编译的可执行文件
# test cases
add_executable(${EXECUTABLE_NAME} test/main.cpp)
# link opencv libs
target_link_libraries(${EXECUTABLE_NAME} easypr thirdparty ${OpenCV_LIBS})
# MESSAGE(${CMAKE_BINARY_DIR}/../)
SET_TARGET_PROPERTIES(${EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY  "${CMAKE_BINARY_DIR}/../")
