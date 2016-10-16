/******************************************************************************
File: 		word.c

Authors: 	John Carpinelli   (johnfc@ecr.mu.oz.au)
	 	Wayne Salamonsen  (wbs@mundil.cs.mu.oz.au)
		Lang Stuiver      (langs@cs.mu.oz.au)

Purpose:	Data compression with a word-based model using
		arithmetic coding.


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
#include "hashtable.h"

#define	WORD		0		/* flag to process a word */
#define NON_WORD	1		/* flag to process a non-word */

#define INIT_CONTEXT	1023		/* initial size of word contexts */
#define CHAR_CONTEXT	256		/* length of character contexts */

#define BUFFER_SIZE	512		/* size of file input buffer */

#define END_OF_MESSAGE  0               /* end of message symbol */


/* Macro to specify what a word is */
#define ISWORD(c) (((c >= 'A') && (c <= 'Z')) || \
		   ((c >= 'a') && (c <= 'z')) || \
		   ((c >= '0') && (c <= '9')))



/* function prototypes */
static void install_symbol_safe(context *pContext, int symbol);
static void init_word_model(hash_table *tables[], context *words[]);
static void purge_word_model(hash_table *tables[], context *words[]);
static void init_char_model(context *characters[], context *lengths[]);
static void read_word(char buffer[], int *buffer_length, int *curr_pos, 
	       string *pWord, int type);

/* global variables */
static int base_memory;	       	/* memory used by character model */
static unsigned int nWords[2]; 	/* counts number of words */
static unsigned int nDistinctWords[2];	/* counts number of distinct words */

#ifdef RCSID
static char
   rcsid[] = "$Id: word.c,v 1.1 1996/08/07 01:34:11 langs Exp $";
#endif



/*
 *
 * print the results of compressing/decompressing a file
 *
 */
void print_results_word(int operation)
{
	fprintf(stderr, "\n" 
		"                              words           non-words\n");
	fprintf(stderr, "Words read             : %10u          %10u\n", 
		nWords[0], nWords[1]);
	fprintf(stderr, "Distinct words         : %10u          %10u\n",
		nDistinctWords[0], nDistinctWords[1]);
}

/*
 * Installs a symbol, if it can't, it halts the program with an error
 * message.  Makes sure initial symbols are always added.
 */
static void install_symbol_safe(context *pContext, int symbol)
{
  if (install_symbol(pContext, symbol) == TOO_MANY_SYMBOLS)
	{
	  fprintf(stderr,"TOO_MANY_SYMBOLS error installing initial symbols\n");
	  fprintf(stderr,"(Perhaps F_bits is too small?)\n");
	  exit(1);
	}
}

/*
 *
 * initialize the word/non-word context and hash tables
 *
 */
static void init_word_model(hash_table *tables[], context *words[])
{
    tables[WORD] = create_table();
    tables[NON_WORD] = create_table();
    words[WORD] = create_context(INIT_CONTEXT, DYNAMIC);
    words[NON_WORD] = create_context(INIT_CONTEXT, DYNAMIC);

    if (tables[WORD]==NULL || tables[NON_WORD]==NULL)
	{ fprintf(stderr,"init_word_model(): Unable to create word tables!\n");
	  exit(1);
 	}
    
    /* add end of message symbol to word contexts */
    install_symbol_safe(words[WORD], END_OF_MESSAGE);
    install_symbol_safe(words[NON_WORD], END_OF_MESSAGE);

    get_memory(2 * MEM_PER_SYMBOL);		/* record memory used */
}


/*
 *
 * free all memory associated with the word and non-word models
 * then create empty models.
 *
 */
static void purge_word_model(hash_table *tables[], context *words[])
{
    /* free the memory used by the word models */
    purge_context(words[WORD]);
    purge_context(words[NON_WORD]);
    purge_table(tables[WORD]);
    purge_table(tables[NON_WORD]);

    /* rebuild the hash tables with no entries */
    purge_memory();			/* set memory count back to zero */
    get_memory(base_memory);

    tables[WORD] = create_table();
    tables[NON_WORD] = create_table();

    if (tables[WORD]==NULL || tables[NON_WORD]==NULL)
	{ fprintf(stderr,
		  "purge_word_model(): Unable to recreate word tables!\n");
	  exit(1);
 	}

    /* add end of message symbol to word contexts */
    install_symbol_safe(words[WORD], END_OF_MESSAGE);
    install_symbol_safe(words[NON_WORD], END_OF_MESSAGE);
}



/*
 *
 * initialize the character and length contexts
 *
 */
static void init_char_model(context *characters[], context *lengths[])
{
    int i;

    /* initialize the character and length contexts */
    characters[WORD] = create_context(CHAR_CONTEXT, STATIC);
    characters[NON_WORD] = create_context(CHAR_CONTEXT, STATIC);
    lengths[WORD] = create_context(MAX_WORD_LEN+1, STATIC);
    lengths[NON_WORD] = create_context(MAX_WORD_LEN+1, STATIC);

    /* initialise char contexts with all chars having a frequency of 1 */ 
    for (i = 0; i < CHAR_CONTEXT; i++)
    {
	if (ISWORD(i)) 
	    install_symbol_safe(characters[WORD], i);
	else
	    install_symbol_safe(characters[NON_WORD], i);
    }

    for (i = 0; i <= MAX_WORD_LEN; i++)
    {
	install_symbol_safe(lengths[WORD], i);
	install_symbol_safe(lengths[NON_WORD], i);
    }

    /* record memory used by character and length contexts */
    get_memory(2 * MAX_WORD_LEN * MEM_PER_SYMBOL);
    get_memory(2 * CHAR_CONTEXT * MEM_PER_SYMBOL);
}



/*
 *
 * compress with word based model using i/o in bitio.c
 *
 */
void encode_word(void)
{
    char	buffer[BUFFER_SIZE];
    int		buffer_len, buffer_pos = 0, word_no, i, type;
    string	curr_word;
    context	*words[2], *characters[2], *lengths[2];
    hash_table	*tables[2];

    /* set up the character and length contexts */
    init_char_model(characters, lengths);

    /* initialize the word and non-word contexts and hash tables */
    init_word_model(tables, words);
    base_memory = get_memory(0);		/* record base memory level */

    buffer_len = 0;

    startoutputtingbits();
    start_encode();
    
    /* start processing with a word */
    type = WORD;
    for (;;)
    {
	read_word(buffer, &buffer_len, &buffer_pos, &curr_word, type);
	if ((buffer_len == 0) && (curr_word.length == 0))
	    break;
	nWords[type]++;
	word_no = lookup_word(&curr_word, tables[type]);
	if (encode(words[type], word_no) == NOT_KNOWN)
	{
	    /* spell out new word before adding to list of words */
	    encode(lengths[type], curr_word.length);
	    
	    for (i = 0; i<curr_word.length; i++)
		encode(characters[type], curr_word.text[i]);
	    
	    /* add word to hash table, and install new symbol */
	    if ((word_no = add_word(&curr_word, tables[type])) == NOMEMLEFT ||
		(install_symbol(words[type], word_no) != 0))
		/* purge word model if memory or symbol limit is exceeded */
		{
		    if (verbose)
			fprintf(stderr, "Reached %s limit "
					"adding new word...purging\n",
				word_no == NOMEMLEFT ? "memory" : "symbol");
		    purge_word_model(tables, words);
		}
	    nDistinctWords[type]++;
	}
 	type = !type;				/* toggle WORD/NON_WORD type */
    } 

    encode(words[type], END_OF_MESSAGE);	/* encode end of message */
    finish_encode();
    doneoutputtingbits();
}


/*
 *
 * uncompress with a word based model using bitio.c for i/o
 *
 */
void decode_word(void)
{
    int i, symbol, type, length;
    hash_table *tables[2];
    context *words[2], *characters[2], *lengths[2];
    string word;
    unsigned char *pWord;
    
    /* set up the character and length contexts */
    init_char_model(characters, lengths);

    /* initialize word/non-word contexts and hash tables */
    init_word_model(tables, words);
    base_memory = get_memory(0);		/* record base memory level */

    startinputtingbits();
    start_decode();
    type = WORD;				/* first symbol is a WORD */

    for (;;)
    {
	symbol = decode(words[type]);
	if (symbol == END_OF_MESSAGE)
	    break;
	nWords[type]++;
	if (symbol == NOT_KNOWN)
	{      
	    /* read in the length, then the spelling of a new word */
	    word.length = decode(lengths[type]);
	    for (i = 0; i<word.length; i++)
		word.text[i] = decode(characters[type]);
	    pWord = word.text;
	    length = word.length;
	    nDistinctWords[type]++;

	    /* add new word to hash table, and install new symbol */
	    if (((symbol = add_word(&word, tables[type])) == NOMEMLEFT) || 
		(install_symbol(words[type], symbol) != 0))
		{
		    /* purge word model if memory limit exceeded */
		    if (verbose)
			fprintf(stderr, "Reached %s limit "
					"adding new word...purging\n",
				symbol == NOMEMLEFT ? "memory" : "symbol");
		    purge_word_model(tables, words);
		}
	}
	else
	    get_word(tables[type], symbol, &pWord, &length);

	/* output the word to standard out */
	BITIO_FWRITE(pWord, length, 1);

	type = !type;			/* toggle between WORD/NON_WORD */
    } 
    finish_decode();
    doneinputtingbits();
}


/*
 *
 * read word or non-word from stdin and update the buffer_length 
 * and buffer_position variables
 *
 */
static void
read_word(char buffer[], int *buffer_length, int *curr_pos, string *pWord,
	  int type)
{
    pWord->length = 0;
    while (pWord->length < MAX_WORD_LEN)
    {
	if (*buffer_length == 0)
	{
	    /* 
	     * if buffer is empty then fill it, using fread. If file to be
             * encoded is empty then return current word
	     */ 
	    if ((*buffer_length = BITIO_FREAD(buffer, 1, BUFFER_SIZE)) == 0)
		return;
	    *curr_pos = 0;
	}
	
	/* 
	 * terminate on non-word character if type = WORD (0)
	 * or word character if type = NON_WORD (1)
	 */
	if ((!ISWORD(buffer[*curr_pos])) ^ type)
	    return;
	else
	{
	    pWord->text[pWord->length] = buffer[*curr_pos];
	    pWord->length += 1;
	    *curr_pos += 1;
	    *buffer_length -= 1;
	}
    }
}

