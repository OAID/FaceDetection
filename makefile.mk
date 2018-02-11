CC = $(PREFIX)gcc
CXX = $(PREFIX)g++
AR = $(PREFIX)ar

export PKG_CONFIG_PATH=/usr/local/AID/pkgconfig

CFLAGS += -Wall -O2
CFLAGS += -I$(TOPDIR)/include

CXXFLAGS += -Wall -O2 -ggdb -std=c++11
CXXFLAGS += -I$(TOPDIR)/include

ARFLAGS = -rcv

CAFFE_ON = 1
MXNET_ON = 0
TF_ON = 0

#MTCNN_LDFLAGS = -L$(MTCNNLIBDIR) -Wl,--whole-archive -lmtcnn -Wl,--no-whole-archive
LDFLAGS += -L$(MTCNNLIBDIR) -lmtcnn

# opencv headers and libraries
CXXFLAGS += `pkg-config --cflags opencv`
LDFLAGS += `pkg-config --libs opencv`

# mxnet settings
ifeq ($(MXNET_ON), 1)
   MXNET_INCS += `pkg-config --cflags mxnet-hrt`

   LDFLAGS += `pkg-config --libs mxnet-hrt`
   CXXFLAGS += $(MXNET_INCS) -Wno-sign-compare
endif

# caffe settings
ifeq ($(CAFFE_ON), 1)
   CAFFE_INCS := `pkg-config --cflags caffe-hrt`

   LDFLAGS += `pkg-config --libs caffe-hrt`
   LDFLAGS += -lprotobuf -lboost_system -lglog
   CXXFLAGS += $(CAFFE_INCS) -DCPU_ONLY=1
endif

#  tensorflow settings
ifeq ($(TF_ON), 1)
   TENSORFLOW_INCS += -I$(TENSORFLOW_ROOT)/include
   LIBS+=-Wl,-rpath,$(TENSORFLOW_ROOT)/lib -L$(TENSORFLOW_ROOT)/lib -ltensorflow

   CXXFLAGS +=$(TENSORFLOW_INCS)
endif

# arm compute library setting
LDFLAGS += `pkg-config --libs computelibrary`

%.os : %.cpp
	$(CXX) -fpic -shared $(CXXFLAGS) -c $< -o $@

%.i : %.c
	$(CC) $(CFLAGS) -E $< -o $@

%.o : %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.d : %.c
	@$(CC) -MM $(CFLAGS) $< > $@.$$$$; \
	sed -e 's/\($(@F)\.o\)[ :]*/$(<D)\/\1 $(@D)\/$(@F) : /g' $@.$$$$ > $@ ; \
	$(RM) $@.$$$$;

%.i : %.cpp
	$(CXX) $(CXXFLAGS) -E $< -o $@

%.o : %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.d : %.cpp
	@$(CXX) -MM $(CXXFLAGS) $< > $@.$$$$; \
	sed -e 's/\($(@F)\.o\)[ :]*/$(<D)\/\1 $(@D)\/$(@F) : /g' $@.$$$$ > $@ ; \
	$(RM) $@.$$$$;
#.depens := $(patsubst %.o, %.d, $(lib_objs))
sinclude $(.depens)
