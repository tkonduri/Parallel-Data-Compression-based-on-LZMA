/******************************************************************************
File: 		word.c

Authors: 	John Carpinelli   (johnfc@ecr.mu.oz.au)
	 	Wayne Salamonsen  (wbs@mundil.cs.mu.oz.au)
		Lang Stuiver      (langs@cs.mu.oz.au)

Purpose:	Data compression with a word-based model using
		arithmetic coding.


Hacked by aht to read file of uints.  Fri Aug 22 08:33:51 EST 1997

Copyright 1995 John Carpinelli and Wayne Salamonsen, All Rights Reserved.
Copyright 1996 Lang Stuiver.  All Rights Reserved.

These programs are supplied free of charge for research purposes only,
and may not sold or incorporated into any commercial product.  There is
ABSOLUTELY NO WARRANTY of any sort, nor any undertaking that they are
fit for ANY PURPOSE WHATSOEVER.  Use them at your own risk.  If you do
happen to find a bug, or have modifications to suggest, please report
the same to Alistair Moffat, alistair@cs.mu.oz.au.  The copyright
notice above and this statement of conditions must remain an integral
part of each and every copy made of these files.

******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bitio.h"
#include "arith.h"
#include "stats.h"
#include "main.h"

#define INIT_CONTEXT	1	    /* initial size of uint contexts */

#define BUFFER_SIZE	4096		/* size of file input buffer */

#define END_OF_MESSAGE  0       /* end of message symbol */


/* function prototypes */
static void install_symbol_safe(context *pContext, int symbol);

/* global variables */
static unsigned int nWords; 	    /* counts number of nums */
static unsigned int nDistinctWords;	/* counts number of distinct nums */

#ifdef RCSID
static char
   rcsid[] = "$Id: word.c,v 1.1 1996/08/07 01:34:11 langs Exp langs $";
#endif


#if !defined(__alpha__)
   typedef unsigned int uint;
   typedef unsigned long ulong;
   typedef unsigned char uchar;
   typedef unsigned short ushort;
#endif

/*
 *
 * print the results of compressing/decompressing a file
 *
 */
void print_results_uints(int operation)
{
	fprintf(stderr, "\n                              uint\n");
	fprintf(stderr, "Words read             : %10u\n", nWords);
	fprintf(stderr, "Distinct uints         : %10u\n",nDistinctWords);
}

/*
 * Installs a symbol, if it can't, it halts the program with an error
 * message.  Makes sure initial symbols are always added.
 */
static void install_symbol_safe(context *pContext, int symbol)
{
  if (install_symbol(pContext, symbol) == TOO_MANY_SYMBOLS)
	{
	  fprintf(stderr,"TOO_MANY_SYMBOLS error installing symbols\n");
	  fprintf(stderr,"(Perhaps F_bits is too small?)\n");
	  exit(1);
	}
}

/*
 *
 * Compress with uint based model using i/o in bitio.c
 *
 */
void encode_uints(void)
{
    uint	buffer[BUFFER_SIZE], *buffer_pos;
    int	    buffer_len;
    context	*nums;

    /* initialize the word and non-word contexts */
    nums = create_context(INIT_CONTEXT, DYNAMIC);
    install_symbol_safe(nums, END_OF_MESSAGE); /* add end of message symbol */
    install_symbol_safe(nums, 1);

    buffer_len = 0;

    startoutputtingbits();
    start_encode();

        /* just a little check on the input */
    *buffer = 0;
    BITIO_FREAD(buffer, sizeof(uint), 1);
    if (*buffer != 1) {
        fprintf(stderr,"Input must start with a 1\n");
        exit(-1);
    }
    nWords = 1; 
    nDistinctWords = 1;

    buffer_len = BITIO_FREAD(buffer, sizeof(uint), BUFFER_SIZE);
    buffer_pos = buffer;
    nWords += buffer_len;
    while (buffer_len > 0)
    {
        if (encode(nums, *buffer_pos) == NOT_KNOWN) {
            /* install new symbol */
            install_symbol_safe(nums, *buffer_pos);
            nDistinctWords++;
        }
        buffer_pos++;
        buffer_len--;
        if (buffer_len == 0) {
            buffer_len = BITIO_FREAD(buffer, sizeof(uint), BUFFER_SIZE);
            buffer_pos = buffer;
            nWords    += buffer_len;
        }
    } 

    encode(nums, END_OF_MESSAGE);	/* encode end of message */
    finish_encode();
    doneoutputtingbits();
}


/*
 *
 * uncompress with a uint based model using bitio.c for i/o
 *
 */
void decode_uints(void)
{
    int symbol;
    context *nums;
    uint next_word = 2;

    #define OUT_BUFFER_SIZE 4096
    uint out_buffer[OUT_BUFFER_SIZE];
    uint *buff = out_buffer;
    
    /* initialize word/non-word contexts and hash tables */
    nums = create_context(INIT_CONTEXT, DYNAMIC);
    install_symbol_safe(nums, END_OF_MESSAGE); /* add end of message symbol */
    install_symbol_safe(nums, 1);

    startinputtingbits();
    start_decode();

        /* first number is always 1  ASSUMES OUT_BUFFER_SIZE > 1 */
    symbol = 1;
    nWords = nDistinctWords = 1;
    *buff = 1;
    buff++;

    for (;;)
    {
	    symbol = decode(nums);
	    if (symbol == END_OF_MESSAGE)
	        break;
	    nWords++;

	    if (symbol == NOT_KNOWN)
	    {      
	        nDistinctWords++;
            symbol = next_word;
            next_word++;

	        install_symbol_safe(nums, symbol); /* install new symbol */
	    }

	    /* output the uint to standard out */
        if (buff == out_buffer + OUT_BUFFER_SIZE) {
	        BITIO_FWRITE(out_buffer, sizeof(uint),OUT_BUFFER_SIZE);
            buff = out_buffer;
        }
        *buff = symbol; buff++;
    } 
	BITIO_FWRITE(out_buffer, sizeof(uint),buff - out_buffer);
    finish_decode();
    doneinputtingbits();
}
