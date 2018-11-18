#!/usr/bin/env bash

rm -rf build
mkdir build
cd build
cmake ..
make -j8
#make(gmake,gnumake)的-j参数，优化多核、多线程的编译过程, 提升编译速度

