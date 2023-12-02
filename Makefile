.PHONY:clean all

LIB_SRC= ./lib/basic.cpp  \
	 ./lib/pool.cpp \
	 ./lib/conv.cpp \
	 ./lib/FPGA_DDR.cpp
 
Compile_CMD= 	-O3

all:VGG16
 
clean:
	@rm -f vgg16

VGG16:
	g++ vgg16_conv0.c ${LIB_SRC} -I ./lib/ -I ./ ${Compile_CMD} -o vgg16

