#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/github/FaceDetection/libmtcnn/


taskset -c 5 ./camera -t caffe
