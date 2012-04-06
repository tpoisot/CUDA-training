# Makefile
#
# Mutsuo Saito (Hiroshima University)
#
# To make bigcrush and bigcrush-param, you need TESTU01
# http://www.iro.umontreal.ca/~simardr/testu01/tu01.html
#
# To make check-poly, you need NTL
# http://www.shoup.net/ntl/

#DEBUG = -O0 -g -ggdb
WARN = -Wmissing-prototypes -Wall #-Winline
#WARN = -Wmissing-prototypes -Wall -W
OPTI = -O3 -finline-functions -fomit-frame-pointer -DNDEBUG \
-fno-strict-aliasing
#--param inline-unit-growth=500 --param large-function-growth=900 #for gcc 4
#STD =
#STD = -std=c89 -pedantic
#STD = -std=c99 -pedantic
#DEBUG = -DDEBUG -g -O0
STD = -std=c99
CC = gcc
CPP = g++

CPPFLAGS = -Wall -Wextra -O3 -msse3 -I../ $(DEBUG)

CCFLAGS = -I../ ${TESTU01LIB} $(OPTI) $(WARN) $(STD) $(DEBUG)
OBJS = ../mtgp32-fast.o ../mtgp64-fast.o ../mtgp32-param-fast.o \
	../mtgp64-param-fast.o

cmp64-cuda: test64.c
	${CC} ${CCFLAGS} -o $@ test64.c ${OBJS}

cmp32-cuda: test32.c
	${CC} ${CCFLAGS} -o $@ test32.c ${OBJS}

cmp32-cuda2: test32-2.c
	${CC} ${CCFLAGS} -o $@ test32-2.c ${OBJS}

check-poly: check-poly.cpp
	${CPP} ${CPPFLAGS} -o $@ check-poly.cpp ${OBJS} -lntl -lgmp -lcrypto

zero-count: zero-count.c
	${CC} ${CCFLAGS} -o $@ zero-count.c $(OBJS)

bigcrush: bigcrush.c
	${CC} ${CCFLAGS} -o $@ bigcrush.c $(OBJS) -ltestu01

bigcrush-param: bigcrush-param.c
	${CC} ${CCFLAGS} -o $@ bigcrush-param.c $(OBJS) -ltestu01

#bigcrush-param-nonlinear: bigcrush-param-nonlinear.c
#	${CC} ${CCFLAGS} -o $@ bigcrush-param-nonlinear.c $(OBJS) -ltestu01
#
#bigcrush-param-nonlinear2: bigcrush-param-nonlinear2.c
#	${CC} ${CCFLAGS} -o $@ bigcrush-param-nonlinear2.c $(OBJS) -ltestu01

.c.o:
	${CC} ${CCFLAGS} -c $<

clean:
	rm -rf *.o *~ *.dSYM
