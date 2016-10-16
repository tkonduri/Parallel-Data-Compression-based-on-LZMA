#
# File:		Makefile
#
# Authors: 	John Carpinelli   (johnfc@ecr.mu.oz.au)
#	 	Wayne Salamonsen  (wbs@mundil.cs.mu.oz.au)
#		Lang Stuiver      (langs@cs.mu.oz.au)
#		Andrew Turpin     (aht@cs.mu.oz.au)
#
# Purpose:	This is the makefile for the revised arithmetic coding 
#		implementation
#
# Based on:	A. Moffat, R. Neal, I.H. Witten, "Arithmetic Coding Revisted",
#		ACM Transactions on Information Systems, 16(3):256-294, 1998
#
#       	A. Moffat, "An improved data structure for cummulative 
#       	probability tables", Software-Practice and Experience,
#		29(7):647-659, 1999.
#
# Any comments or suggestions to: alistair@cs.mu.oz.au
#
CC = gcc

# Change the fixed B_BITS and F_BITS parameters here if required.

FIXED_SHIFT = -DB_BITS=32 -DF_BITS=27
VARY_SHIFT  = 					 -DVARY_NBITS
FIXED_MULT  = -DB_BITS=32 -DF_BITS=27 -DMULT_DIV 
VARY_MULT   = 			      -DMULT_DIV -DVARY_NBITS

# Uncomment one of the setup flag assignments below to choose the type
# of coder.

# SETUP_FLAGS = $(FIXED_SHIFT)
# SETUP_FLAGS = $(VARY_SHIFT)
SETUP_FLAGS = $(FIXED_MULT)
# SETUP_FLAGS = $(VARY_MULT)

# comment this line out if your system doesn't have the "times" system call
SYSTYPE = -DSYSV

# Setup compile flags
CFLAGS = -Wall -O3 -ansi -pedantic $(SYSTYPE) $(SETUP_FLAGS)

LINKFLAGS = 

CODER	= arith
BITIO	= bitio

ARITHOBJ = main.o bits.o char.o word.o hashtable.o uint.o stats.o \
		$(CODER).o $(BITIO).o

all 	:	arith_coder word char bits uint

arith_coder	:	$(ARITHOBJ)
		$(CC) $(LINKFLAGS) -o arith_coder $(ARITHOBJ)

word 	:	arith_coder ; test '[ ! -L word ]'
		ln -s arith_coder word

char 	:	arith_coder ; test '[ ! -L char ]'
		ln -s arith_coder char

bits 	:	arith_coder ; test '[ ! -L bits ]'
		ln -s arith_coder bits

uint 	:	arith_coder ; test '[ ! -L uint ]'
		ln -s arith_coder uint

clean	:	
	/bin/rm -f $(ARITHOBJ)

clobber	:	
	/bin/rm -f $(ARITHOBJ)
	/bin/rm -f word char bits uint arith_coder

package:
	tar cf - Makefile README WHATSNEW *.c *.h *.i *.1 | gzip -9 > ../arith_coder-3.tar.gz

# dependencies for header files
main.o		:	$(BITIO).h $(CODER).h stats.h main.h Makefile
char.o 		:	$(BITIO).h $(CODER).h stats.h main.h Makefile
bits.o		:	$(BITIO).h $(CODER).h stats.h main.h Makefile
uint.o		:	$(BITIO).h $(CODER).h stats.h main.h Makefile
word.o 		: 	$(BITIO).h $(CODER).h stats.h main.h Makefile hashtable.h
hashtable.o	: 	stats.h $(CODER).h main.h hashtable.h  Makefile
stats.o 	:	stats.h $(CODER).h Makefile
$(BITIO).o	:	$(BITIO).h Makefile
$(CODER).o 	:	$(CODER).h $(BITIO).h Makefile
