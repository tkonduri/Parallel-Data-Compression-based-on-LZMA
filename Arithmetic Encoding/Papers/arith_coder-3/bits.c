/******************************************************************************
File:		bits.c

Authors: 	John Carpinelli   (johnfc@ecr.mu.oz.au)
	 	Wayne Salamonsen  (wbs@mundil.cs.mu.oz.au)
		Lang Stuiver	  (langs@cs.mu.oz.au)

Purpose:	Data compression using an nth order binary model and revised 
		arithmetic coding method.

Based on: 	A. Moffat, R. Neal, I.H. Witten, "Arithmetic Coding Revisited",
		Proc. IEEE Data Compression Conference, Snowbird, Utah, 
		March 1995.


Copyright 1995 John Carpinelli and Wayne Salamonsen, All Rights Reserved.

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
#include "bitio.h"
#include "arith.h"
#include "stats.h"
#include "main.h"

/* global variables */
int bits_context=DEFAULT_BITS_CONTEXT;	    /* no. bits in binary contexts */
static unsigned int non_null_contexts = 0;  /* count of contexts used */


#ifdef RCSID
static char
   rcsid[] = "$Id: bits.c,v 1.1 1996/08/07 01:34:11 langs Exp $";
#endif

/*
 *
 * print the results of compressing/decompressing a file
 *
 */
void print_results_bits(int decode)
{
    fprintf(stderr, "Context bits           : %10d\n", bits_context);
    fprintf(stderr, "Number of contexts     : %10u\n", 
	    non_null_contexts);
}



/*
 *
 * compress data using i/o in bitio.c, compressing
 * each bit with probabilities derived from the previous bits
 *
 */
void encode_bits(void)
{
    unsigned int	i, j, cur_context, closest_context;
    unsigned int 	mask, bit;
    int			bits_to_go, buffer;
    binary_context	**contexts;
    binary_context	*still_coding;		

    OUTPUT_BYTE(bits_context);

    /* initialise context array */
    contexts = (binary_context **)do_malloc(sizeof(binary_context *) * 
					  (1 << bits_context));
    if (contexts == NULL)
    {
	fprintf(stderr, "bits: not enough memory to allocate context array\n");
	exit(1);
    }
    still_coding = create_binary_context();

    /* initialise contexts to NULL */
    for (i=0; i < 1 << bits_context; i++)
	contexts[i] = NULL;

    /* initalise variables */
    cur_context = 0;
    mask = (1 << bits_context) - 1;
    
		/* Ensure context '0' is created, as it is implicitly
	  	 * assumed to exist in the main loop that follows
		 */

    if (get_memory(sizeof(binary_context)) != NOMEMLEFT)
    {
         contexts[0] = create_binary_context();
         non_null_contexts++;
    }
    else
    {
	fprintf(stderr,"bits: not enough memory to allocate initial context\n");
	exit(1);
    }

    startoutputtingbits();
    start_encode();

    while ((buffer = INPUT_BYTE()) != EOF)
    {
	binary_encode(still_coding, 1);
	   
	for (bits_to_go = 7; bits_to_go >= 0; bits_to_go--)
	{
	    if (contexts[cur_context] == NULL)
	    {
		if (get_memory(sizeof(binary_context)) != NOMEMLEFT)
		{
		    contexts[cur_context] = create_binary_context();
		    non_null_contexts++;
		}
		else 
		    /* 
		     * determine closest existing context to current one
		     * by stripping off the leading bits of the context
		     * guaranteed to get contexts[0] if nothing closer
		     */
		{
		    closest_context = cur_context;
		    j = 1;
		    do
		    {
			closest_context &= (mask >> j);
			j++;
		    } while (contexts[closest_context] == NULL);
		    contexts[cur_context] = contexts[closest_context];
		}
	    }
	    bit = (buffer >> bits_to_go) & 1;
	    binary_encode(contexts[cur_context], bit);
	    cur_context = ((cur_context << 1) | bit) & mask;
	}
    
    }
    /* encode end of message flag */
    binary_encode(still_coding, 0);
    finish_encode();
    doneoutputtingbits();
}




/*
 *
 * decode a compressed file using a bit context model,
 * using bitio.c i/o
 *
 */
void decode_bits(void)
{
    int			i, j, cur_context, closest_context, buffer, bits_to_go;
    int 		mask, bit;
    binary_context	**contexts;
    binary_context	*still_coding;		

    bits_context = INPUT_BYTE();

    /* initialise context array */
    contexts = (binary_context **)do_malloc(sizeof(binary_context *) * 
					  (1 << bits_context));
    if (contexts == NULL)
    {
	fprintf(stderr, "bits: unable to malloc %d bytes\n",
		(int)sizeof(binary_context *) * (1 << bits_context)); 
	exit(1);
    }
    still_coding = create_binary_context();
    
    /* initialise contexts to NULL */
    for (i=0; i < 1 << bits_context; i++)
	contexts[i] = (binary_context *)NULL;

    /* initalise variables */
    cur_context = 0;
    mask = (1 << bits_context) - 1;

    startinputtingbits();
    start_decode();

    /* decode the file */
    while (binary_decode(still_coding))
    {
	buffer = 0;
	for (bits_to_go = 7; bits_to_go >= 0; bits_to_go--)
	{
	    if (contexts[cur_context] == (binary_context *)NULL)
	    {
		if (get_memory(sizeof(binary_context)) != NOMEMLEFT)
		{
		    contexts[cur_context] = create_binary_context();
		    non_null_contexts++;
		}
		else 
		{   /* 
		     * determine closest existing context to current one
		     * by stripping off the leading bits of the context
		     * guaranteed to get contexts[0] if nothing closer
		     */
		    closest_context = cur_context;
		    j = 1;
		    do
		    {
			closest_context &= (mask >> j);
			j++;
		    } while (contexts[closest_context] == NULL);
		    contexts[cur_context] = contexts[closest_context];
		}
	    }
	    bit = binary_decode(contexts[cur_context]);
	    buffer = (buffer << 1) | bit;
	    cur_context = ((cur_context << 1) | bit) & mask;
	}	    
	OUTPUT_BYTE(buffer);
    }
    finish_decode();
    doneinputtingbits();
}
