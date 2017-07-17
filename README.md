# MTCNN C++ Implementation

This is a C++ project to implement MTCNN, a perfect face detect algorithm, on different DL frameworks.<br>
The most popular frameworks: caffe/mxnet/tensorflow, are all suppported now.

# Build

* Bulid caffe,  mxnet  or tensorflow first
   Please  edit makefile.mk (set xxx_ON flags to enable corresponding dp framework) to select one or more to be supported
	* Build CaffeOnACL, refer to [CaffeOnACL Release notes](https://github.com/OAID/caffeOnACL/tree/master/acl_openailab/README.md)
	* Build MxnetOnACL, refer to [MxnetOnACL release notes](https://github.com/OAID/mxnetOnACL/tree/master/acl_openailab/README.md)
	* Build tensorflow, to generate libtensorflow.so, please use:

	>> bazel build --config=opt //tensorflow/tools/lib_package:libtensorflow
	>> the tarball, bazel-bin/tensorflow/tools/lib_package/libtensorflow.tar.gz, includes the libtensorflow.so and c header files  

* Install opencv package 

* Edit Makefile to set `CAFFE_ROOT`, `MXNET_ROOT`  or `TENSORFLOW_ROOT` to the right path in your machine. For example : CAFFE_ROOT=~/oaid/caffeOnACL.

* make -j4

# Run
If the basic work is ready (build caffe/Mxnet/Tensorflow sucessfully) followed by above steps. You can run the test now.
### 1. Test on single picture:

	./test -f photo_fname [ -t DL_type] [-s] 
	  -f photo_fname  picture to be  detected
	  -t DL_type      DL frame: "caffe" , "mxnet"(default) or "tensorflow"
	  -s              Save face chop into jpg files

The new picture, which boxed face and 5 landmark points will be created and saved as "new.jpg"

### 2. Test on camera (DL Framework is caffe)

 	./camera -t caffe


# Credit

### MTCNN algorithm

https://github.com/kpzhang93/MTCNN_face_detection_alignment

### MTCNN C++ on Caffe

https://github.com/wowo200/MTCNN

### MTCNN python on Mxnet

https://github.com/pangyupo/mxnet_mtcnn_face_detection

### MTCNN python on Tensorflow

FaceNet uses MTCNN to align face

https://github.com/davidsandberg/facenet
From this directory:
  facenet/src/align
