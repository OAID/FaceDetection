TOPDIR := $(shell pwd)
MTCNNLIBDIR = $(TOPDIR)/libmtcnn

include makefile.mk

EXES = test camera

SUBDIRS += $(MTCNNLIBDIR)

export TOPDIR

all : $(EXES)

$(EXES) : libs

test : test.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

camera : camera.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

libs : force
	@for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir; \
	done

force :

libs_clean:
	@for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done

clean : libs_clean
	@find . -name '*.[od]' | xargs rm -f
	$(RM) $(EXES)

debug :
	@echo "LDFLAGS = $(LDFLAGS)"
.PHONY : all clean force libs
