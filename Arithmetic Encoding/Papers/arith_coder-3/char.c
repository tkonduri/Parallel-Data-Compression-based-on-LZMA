/******************************************************************************
File:		char.c

Authors: 	John Carpinelli   (johnfc@ecr.mu.oz.au)
	 	Wayne Salamonsen  (wbs@mundil.cs.mu.oz.au)
		Lang Stuiver	  (langs@cs.mu.oz.au)

Purpose:	Data compression using a characater-based model and revised 
		arithmetic coding method.

Based on: 	A. Moffat, R. Neal, I.H. Witten, "Arithmetic Coding Revisted",
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

#define END_OF_MESSAGE	256		/* end of message symbol */
#define CHAR_CONTEXT	256  		/* length of character contexts */

#ifdef RCSID
static char
   rcsid[] = "$Id: char.c,v 1.1 1996/08/07 01:34:11 langs Exp $";
#endif


/*
 *
 * compress a file using zero-order character model
 *
 */
void encode_char()
{
    int		i, cur_char;
    context	*characters;

    /* initialise character context */
    characters = create_context(CHAR_CONTEXT+1, STATIC);
    for (i=0; i < CHAR_CONTEXT; i++)
	install_symbol(characters, i);
    if (install_symbol(characters, END_OF_MESSAGE) == TOO_MANY_SYMBOLS)
	  {
		fprintf(stderr,"TOO_MANY_SYMBOLS: "
			       "Couldn't install initial symbols\n");
		fprintf(stderr,"(Perhaps F_bits (-f option) is too small?)\n");
		exit(1);
	  }
    startoutputtingbits();
    start_encode();


    while ((cur_char = INPUT_BYTE()) != EOF)
    {
	encode(characters, cur_char);
    }

    encode(characters, END_OF_MESSAGE);	/* encode end of message */
    finish_encode();
    doneoutputtingbits();

    return;
}


/*
 *
 * decode a compressed file using zero-order character model 
 *
 */

void decode_char()
{
    int i, symbol;
    context *characters;

    /* initialise character context */
    characters = create_context(CHAR_CONTEXT+1, STATIC);
    for (i=0; i < CHAR_CONTEXT; i++)
	install_symbol(characters, i);
    if (install_symbol(characters, END_OF_MESSAGE) == TOO_MANY_SYMBOLS)
	   {
		fprintf(stderr, "TOO_MANY_SYMBOLS: "
				"Couldn't install initial symbols\n");
		exit(1);
	   }

    startinputtingbits();
    start_decode();
 
    for (;;)
    {
	symbol=decode(characters);
	if (symbol == END_OF_MESSAGE)
	    break;

	OUTPUT_BYTE(symbol);
    }
    finish_decode();
    doneinputtingbits();

    return;
}
