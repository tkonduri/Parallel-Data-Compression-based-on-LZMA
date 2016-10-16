/******************************************************************************
File:		hashtable.h

Authors: 	John Carpinelli   (johnfc@ecr.mu.oz.au)
	 	Wayne Salamonsen  (wbs@mundil.cs.mu.oz.au)

Purpose:	Data compression using a word-based model and revised 
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
#ifndef HASHTABLE_H
#define HASHTABLE_H

#define	MAXBUCKET	10		/* no. of words in each bucket */
#define MAX_WORD_LEN 	32		/* Maximum Word Length */
#define STRINGBLOCK	1024		/* size of each block of words */
#define GROWTH_RATE	2		/* growth on a bucket split */
#define EXTRA_SYMBOLS	1 		/* no. of extra symbols like ESCAPE */
#define	ALLOCATE_BLOCK	100		/* no. of buckets allocated at once */

#define HASH_MULT	613
#define HASH_MOD	1111151		/* a prime number */

/*
*
* data structure to hold a string with a length byte.
* Null terminated strings aren't used because non-word strings
* may include multiple \0's
*
*/
typedef struct {
  unsigned char length;			/* length of string */
  unsigned char text[MAX_WORD_LEN];	/* characters of string */
} string;


/*
*
* data structure for holding blocks of words
*
*/
typedef struct string_block string_block;
struct string_block {
    int length;
    unsigned char strings[STRINGBLOCK];
    string_block *prev;
};


/*
 * 
 * dictionary data structures
 *
 */
typedef struct {
    unsigned char *pWord;		/* pointer to word */
    int wordNumber;			/* ordinal word number */
} word_rec;

typedef struct {
    int nBits;				/* number of hash bits used */
    int nWords;				/* number of words in this bucket */
    word_rec words[MAXBUCKET];		/* word records */
} bucket;

/*
 * data structure for holding blocks of buckets
 */
typedef struct bucket_block bucket_block;
struct bucket_block {
    int nFreeBuckets;
    bucket *pBuckets;
    bucket_block *prev;
};


typedef struct {
    int nBuckets;			/* number of buckets in table */
    int nBits;				/* number of bits of key used */
    int next_number;			/* next available ordinal word no. */
    bucket **pIndex;			/* index to buckets */
    string_block *pStrings;		/* block of word/non-words */
    unsigned char **pNumToWords;	/* maps word nos to words */
    int numToWordsLimit;		/* current size of array */
    bucket_block *pFreeBuckets;		/* array of preallocated buckets */
} hash_table;


/*
 *
 * model interface functions
 *
 */
hash_table *create_table(void);
int lookup_word(string *pItem, hash_table *pTable);
void get_word(hash_table *pTable, int symbol,
	unsigned char **pText, int *pLength);
int add_word(string *pItem, hash_table *pTable);
void purge_table(hash_table *pTable);

#endif

