include $(TOPDIR)/makefile.mk

LIBMTCNN = libmtcnn
LIB_STATIC = $(addsuffix .a, $(LIBMTCNN))
LIB_SHARED = $(addsuffix .so, $(LIBMTCNN))

objs-y =
objs-y += mtcnn.o
objs-y += comm_lib.o
objs-y += utils.o

ifeq ($(CAFFE_ON), 1)
objs-y += caffe_mtcnn.o
endif

ifeq ($(MXNET_ON), 1)
objs-y += mxnet_mtcnn.o
endif

ifeq ($(TF_ON), 1)
objs-y += tensorflow_mtcnn.o
endif

objs-os := $(patsubst %.o, %.os, $(objs-y))

all : $(LIB_SHARED) $(LIB_STATIC)

$(LIB_STATIC) : $(objs-y)
	$(AR) $(ARFLAGS) $@ $^

$(LIB_SHARED) : $(objs-os)
	$(CXX) -fpic -shared -o $@ $^

clean :
	find . \( -name '*.[oad]' \
		-o -name '*.os' \
		-o -name '*.so' \) -type f -print \
		| xargs rm -f
