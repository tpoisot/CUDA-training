# Makefile

#DEBUG = -DDEBUG -g -O0

#ARCH = -arch=compute_11
#ARCH = -arch=compute_13
#VERB = -Xptxas=-v

NVCC = nvcc ${DEBUG} ${ARCH} -D__STDC_FORMAT_MACROS -I../ \
	-D__STDC_CONSTANT_MACROS ${VERB}

MTGP32_INC = ../mtgp32-fast.h ../mtgp32-fast.c mtgp-util.cuh mtgp-util.cu
MTGP64_INC = ../mtgp64-fast.h ../mtgp64-fast.c mtgp-util.cuh mtgp-util.cu
MTGP32_SRC = ../mtgp32-fast.c mtgp-util.cu
MTGP64_SRC = ../mtgp64-fast.c mtgp-util.cu
CUDALINK = -lcuda

.SUFFIXES:	.cu .cubin .ptx .bin

samples:sample32-cuda sample32-cuda-tex sample32-cuda-minmem \
	sample32-cuda23209-tex sample64-cuda sample64-cuda-tex

sample32-cuda: mtgp32-cuda.cu ${MTGP32_INC} mtgp32dc-param-11213.c
	${NVCC} -o $@ mtgp32-cuda.cu mtgp32dc-param-11213.c \
	${MTGP32_SRC} ${CUDALINK}

sample32-cuda-tex: mtgp32-cuda-tex.cu ${MTGP32_INC} mtgp32dc-param-11213.c
	${NVCC} -o $@ mtgp32-cuda-tex.cu mtgp32dc-param-11213.c \
	${MTGP32_SRC} ${CUDALINK}

sample32-cuda-minmem: mtgp32-cuda-minmem.cu ${MTGP32_INC} \
	mtgp32dc-param-11213.c
	${NVCC} -o $@ mtgp32-cuda-minmem.cu mtgp32dc-param-11213.c \
	${MTGP32_SRC} ${CUDALINK}

sample32-cuda23209-tex: mtgp32-cuda23209-tex.cu ${MTGP32_INC} \
	../mtgp32-param-fast.c
	${NVCC} -o $@ mtgp32-cuda23209-tex.cu ../mtgp32-param-fast.c \
	${MTGP32_SRC} ${CUDALINK}

sample64-cuda: mtgp64-cuda.cu ${MTGP64_INC} mtgp64dc-param-11213.c
	${NVCC} -o $@ mtgp64-cuda.cu mtgp64dc-param-11213.c \
	${MTGP64_SRC} ${CUDALINK}

sample64-cuda-tex: mtgp64-cuda-tex.cu ${MTGP64_INC} mtgp64dc-param-11213.c
	${NVCC} -o $@ mtgp64-cuda-tex.cu mtgp64dc-param-11213.c \
	${MTGP64_SRC} ${CUDALINK}

# This sample needs compute capability 1.3 or greater than 1.3
sample64-cuda13-tex: mtgp64-cuda13-tex.cu ${MTGP64_INC} mtgp64dc-param-11213.c
	${NVCC} -arch=compute_13 -o $@ mtgp64-cuda13-tex.cu \
	mtgp64dc-param-11213.c ${MTGP64_SRC} ${CUDALINK}

.cu.ptx:
	${NVCC} -ptx $<

.cu.cubin:
	${NVCC} -cubin $<

.cu.bin:
	${NVCC} -o $@ $< ${CUDALINK}

clean:
	rm -f *.o *~ *.cubin *.ptx
