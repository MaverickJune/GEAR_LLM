#!/bin/bash
set -e -x
gcc -c -fPIC sys_perf.c -I/home/nxc/anaconda3/envs/rime/include/python3.11
gcc -shared -o sys_perf.so sys_perf.o \
  -L/home/nxc/anaconda3/envs/rime/lib -lpython3.11 \
  -Wl,-rpath,/home/nxc/anaconda3/envs/rime/lib
rm -f sys_perf.o