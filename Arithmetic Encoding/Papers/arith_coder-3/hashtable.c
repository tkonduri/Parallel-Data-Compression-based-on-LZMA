/******************************************************************************
File:	       	hashtable.c

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

 $Log: hashtable.c,v $
 Revision 1.1  1996/08/07 01:34:11  langs
 Initial revision

******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "arith.h"
#include "stats.h"
#include "main.h"
#include "hashtable.h"

#ifdef RCSID
static char rcsid[] = "$Id: hashtable.c,v 1.1 1996/08/07 01:34:11 langs Exp $";
#endif

/* Local function declarations */

static bucket *get_bucket(hash_table *pTable);
static unsigned char *add_string(string *pItem, hash_table *pTable);


/*
 *
 * Extendible hashing is used to store both words and non-words. The
 * initial size of the hash table is determined by INITIAL_BITS.
 *
 * Converting word numbers to words in the decoder is done using an
 * array of word pointers. Each hash table holds this array and updates
 * it as words are added.
 *
 */

/*
 *
 * create a new hash table
 * returns pointer to new table, or NULL if memory exhausted
 *
 */
hash_table 
*create_table(void)
{
    hash_table *pTable;

    /* allocate the hash table structure */
    if ((pTable = (hash_table *) do_malloc(sizeof(hash_table))) == NULL)
	return NULL;
    
    /* initially allocate index of length one */
    pTable->nBits = 0;
    pTable->nBuckets = 0;
    if ((pTable->pIndex = (bucket **) do_malloc(sizeof(bucket *))) == NULL)
	return NULL;

    /* point index to first bucket */
    pTable->pFreeBuckets = NULL;
    pTable->pIndex[0] = get_bucket(pTable);
    pTable->pIndex[0]->nBits = 0;
    pTable->pIndex[0]->nWords = 0;

    pTable->next_number = EXTRA_SYMBOLS;

    /* start string tables */
    if ((pTable->pStrings = (string_block *) do_malloc(sizeof(string_block)))
	== NULL)
	return NULL;
    pTable->pStrings->prev = NULL;
    pTable->pStrings->length = 0;

    /* allocate memory for Number-to-Wordptr array */
    if ((pTable->pNumToWords = (unsigned char **) 
	do_malloc(ALLOCATE_BLOCK * sizeof(unsigned char **))) == NULL)
	return NULL;
    pTable->numToWordsLimit = ALLOCATE_BLOCK;
    return pTable;
}



/*
 *
 * get an empty bucket from the pool of preallocated buckets
 * returns NULL if memory exceeded
 *
 */
static bucket *get_bucket(hash_table *pTable)
{
    bucket_block *pBlock, *pNew;

    pBlock=pTable->pFreeBuckets;
    
    if ((pBlock == NULL) || (pBlock->nFreeBuckets == 0)) 
    {
	if ((pNew=(bucket_block *) do_malloc(sizeof(bucket_block))) == NULL)
	    return NULL;
	pNew->prev = pBlock;
	pTable->pFreeBuckets = pNew;
	if ((pNew->pBuckets = (bucket *) 
	    do_malloc(sizeof(bucket) * ALLOCATE_BLOCK)) == NULL)
	    return NULL;
	pNew->nFreeBuckets = ALLOCATE_BLOCK;
	pBlock = pNew;
    	pNew->pBuckets->nBits = 0;
        pNew->pBuckets->nWords = 0;
    }
    pTable->nBuckets++;
    pBlock->nFreeBuckets--;
    return (pBlock->pBuckets + pBlock->nFreeBuckets);
}


/*
 * 
 * hash function which takes a length and a pointer to the string
 *
 */
int 
hash(int length, unsigned char *pText)
{
    int hash, i;

    hash = 0;
    for (i = 0; i < length; i++) 
	hash = HASH_MULT*hash + pText[i];
    hash += length;
    hash = hash % HASH_MOD;
    return hash;
}


/*
 *
 * look up a word in the hash table.
 * Returns the ordinal word number if word found, 
 * or the next unused word number if word is unknown
 *
 */
int 
lookup_word(string *pItem, hash_table *pTable)
{
    int i, key;
    bucket *pBucket;

    key = hash(pItem->length, pItem->text);
    /* strip off the unused bits of the key */
    key = key & ((1 << pTable->nBits) -1);
    pBucket = pTable->pIndex[key];

    /* search the bucket for the string */
    for (i = 0; i<pBucket->nWords; i++)
    {
	/* compare the lengths */
	if (pItem->length == (int) *(pBucket->words[i].pWord))
	{
	    /* compare the text */
	    if (memcmp(pItem->text,pBucket->words[i].pWord+1,
			pItem->length) == 0) 
		return pBucket->words[i].wordNumber;
	}
    }
    return pTable->next_number;		/* return next available number */
}



/*
 *
 * add a word to the hash table
 * if the bucket overflows, double the size of the table and split
 * all buckets
 * returns word number if successful, or NOMEMLEFT if memory limit reached
 *
 */
int 
add_word(string *pItem, hash_table *pTable)
{
    int i, key, tail, length, nWord, nWordsOld, nWordsNew, word_no;
    bucket *pBucket, *pNewBucket;

    /* note new memory required by statistics to store symbol */
    if (get_memory(MEM_PER_SYMBOL) == NOMEMLEFT)
	return NOMEMLEFT;

    key = hash(pItem->length, pItem->text);

    /* strip off the unused bits of the key */
    key = key & ((1 << pTable->nBits) -1);
    pBucket = pTable->pIndex[key];

    /* add the item to the bucket */
    nWord = pBucket->nWords;
    if (nWord < MAXBUCKET) 
    {
	pBucket->words[nWord].wordNumber = (word_no = pTable->next_number);
	pTable->next_number++;
	if ((pBucket->words[nWord].pWord = add_string(pItem, pTable)) == NULL)
	    return NOMEMLEFT;
	
	pTable->pNumToWords[word_no-EXTRA_SYMBOLS] = 
	    pBucket->words[nWord].pWord;
	pBucket->nWords++;
    }
    else {
	/* split bucket on pBucket->nBits+1 bit */
	tail = key & ((1 << pBucket->nBits) -1);	/* save for later */
	pBucket->nBits++;
	if ((pNewBucket = get_bucket(pTable))==NULL)
	    return NOMEMLEFT;
	pNewBucket->nBits = pBucket->nBits;
	
	nWordsOld = 0;
	nWordsNew = 0;
	for (nWord = 0; nWord < pBucket->nWords; nWord++) 
	{
	    /*
	     * move each word depending on the leftmost 
	     * significant bit
	     */
	    key = hash(*(pBucket->words[nWord].pWord),
		       (pBucket->words[nWord].pWord)+1);
	    key = key & (1 << (pBucket->nBits - 1));
	    if (key>0) 
	    {	/* move word to new bucket */
		pNewBucket->words[nWordsNew] = pBucket->words[nWord];
		nWordsNew++;
	    }
	    else 
	    {	/* put word in old bucket */
		pBucket->words[nWordsOld] = pBucket->words[nWord];
		nWordsOld++;
	    }
	}
	pNewBucket->nWords = nWordsNew;
	pBucket->nWords = nWordsOld;

	/* check if we need to double the index, or rearrange pointers */
	if (pBucket->nBits <= pTable->nBits) 
	{  
	    /* add leading one to key to make new bucket key */
	    tail = tail | (1 << (pBucket->nBits-1));
	    /* point index entries ending with tail to new bucket */
	    for (i=0; i < (1 << (pTable->nBits - pBucket->nBits)); i++)
		pTable->pIndex[(i << pBucket->nBits) | tail] = pNewBucket;
	}
	else
	{
	    /* must double size of table */
	    length = 1 << pTable->nBits;
	    pTable->nBits++;
	    if ((pTable->pIndex = (bucket **) 
		 do_realloc(pTable->pIndex, length * 2 * sizeof(bucket *)))
							  == NULL)
		return NOMEMLEFT;
	    
	    /* copy old half of index into new half */
	    memcpy(pTable->pIndex + length, pTable->pIndex, 
		   length*sizeof(bucket *));
	    
	    /* pointer in second half points to new bucket */
	    pTable->pIndex[(1 << (pTable->nBits-1)) | tail] = pNewBucket;
	}
	return (add_word(pItem, pTable));
    }
    if (pTable->next_number-EXTRA_SYMBOLS == pTable->numToWordsLimit)
    {
	pTable->numToWordsLimit *= GROWTH_RATE;
	/* resize NumToWords array to new size */
	pTable->pNumToWords = (unsigned char **) 
	    do_realloc(pTable->pNumToWords, pTable->numToWordsLimit * 
		       sizeof(char *));
    	if (pTable->pNumToWords == NULL)
	    return NOMEMLEFT;
    }
	
    return word_no;
}



/*
*
* look up a word given its ordinal word number,
* returning a pointer to the text and updating the length
*
*/
void 
get_word(hash_table *pTable, int symbol, unsigned char **pText, int *pLength)
{
    *pText = pTable->pNumToWords[symbol-EXTRA_SYMBOLS];
    *pLength = (int) *((unsigned char *)*pText);
    *pText += 1;			/* move pointer past length bytes */
}


/*
 *
 * function to add a string to a hash table's string block
 * if the string_block is full, create a new one and link it to the old one
 * returns a pointer to the string in the string block
 * Returns NULL if memory limit exceeded
 *
 */
static unsigned char 
*add_string(string *pItem, hash_table *pTable)
{
    unsigned char *pWord;
    string_block *pBlock, *pNew;

    pBlock = pTable->pStrings;

    /* check that there is enough room in the string block */
    if (STRINGBLOCK - pBlock->length > pItem->length+1) 
    {
	/* copy the length then the text into the string block */
	pWord = pBlock->strings+pBlock->length;
	*pWord = pItem->length;
	memcpy(pWord+1, pItem->text, pItem->length);
	pBlock->length += pItem->length+1;
	return (pWord);
    }
    else {
	if ((pNew = (string_block *) do_malloc(sizeof(string_block))) == NULL)
	{
		/* Reached memory limit adding new word */
	    return NULL;
	}
	pNew->prev = pBlock;
	pNew->length = 0;
	pTable->pStrings = pNew;
	return (add_string(pItem, pTable));
    }
}


/*
*
* free all memory associated with a hash table
*
*/
void 
purge_table(hash_table *pTable)
{
    string_block *pThis, *pPrev;
    bucket_block *pBlock, *prev;

    free(pTable->pNumToWords);
    free(pTable->pIndex);

    /* free the linked list of bucket blocks */
    pBlock = pTable->pFreeBuckets;
    while (pBlock != NULL)
    {
	prev = pBlock->prev;
	free(pBlock);
	pBlock=prev;
    }

    /* free the linked list of string blocks */
    pThis = pTable->pStrings;
    while (pThis != NULL)
    {
	pPrev = pThis->prev;
	free(pThis);
	pThis = pPrev;
    }
    free(pTable);
}

