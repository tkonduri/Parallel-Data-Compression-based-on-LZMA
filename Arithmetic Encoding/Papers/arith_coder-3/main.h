/******************************************************************************
File:		main.h

Author:		Lang Stuiver      (langs@cs.mu.oz.au)

Purpose:	Data compression using revised arithmetic coding method.

Based on:	A. Moffat, R. Neal, I.H. Witten, "Arithmetic Coding Revisted",
		Proc. IEEE Data Compression Conference, Snowbird, Utah,
		March 1995.

Copyright 1996 Lang Stuiver, All Rights Reserved.

These programs are supplied free of charge for research purposes only,
and may not sold or incorporated into any commercial product.  There is
ABSOLUTELY NO WARRANTY of any sort, nor any undertaking that they are
fit for ANY PURPOSE WHATSOEVER.  Use them at your own risk.  If you do
happen to find a bug, or have modifications to suggest, please report
the same to Alistair Moffat, alistair@cs.mu.oz.au.  The copyright
notice above and this statement of conditions must remain an integral
part of each and every copy made of these files.

******************************************************************************/
#ifndef MAIN_H

#define VERSION " 2.0"
#define VERSION_LEN	4		/* Length of version str */

#define ENCODE	0
#define DECODE  1

#define DEFAULT_MODEL	"char"		/* "char", "word", or "bits" */

#define NOMEMLEFT       (-1)            /* flag set when mem runs out */

#define MEGABYTE        (1 << 20)       /* size of one megabyte */

#define DEFAULT_MEM     8               /* default 8 megabyte limit */
#define MIN_MBYTES      1               /* minimum allowable memory size */
#define MAX_MBYTES      255             /* maximum no for 8 bit int */
#define MAGICNO_LENGTH  4               /* length of magic number */


extern int verbose;

void purge_memory(void);
void *do_malloc(size_t size);
void *do_realloc(void *ptr, size_t size);
int get_memory(size_t size);


/* In bits.c */
void encode_bits(void);
void decode_bits(void);
void print_results_bits(int);

extern int bits_context;	/* May be set by main.c */

#define MAX_CONTEXT_BITS        20      /* max. number of bits for context */
#define MIN_CONTEXT_BITS        0       /* min. number of bits for context */
#define DEFAULT_BITS_CONTEXT    16      /* default value for bits_context */


/* In word.c */
void encode_word(void);
void decode_word(void);
void print_results_word(int);

/* In char.c */
void encode_char(void);
void decode_char(void);

/* In num.c */
void encode_uints(void);
void decode_uints(void);

/* In dyn_chars.c */
void encode_dchars(void);
void decode_dchars(void);

#endif					/* #ifndef MAIN_H */
