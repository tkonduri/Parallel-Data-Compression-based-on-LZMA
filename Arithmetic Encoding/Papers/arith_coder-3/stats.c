/******************************************************************************
File:         stats.c

Authors:     John Carpinelli   (johnfc@ecr.mu.oz.au)
             Wayne Salamonsen  (wbs@mundil.cs.mu.oz.au)
             Lang Stuiver      (langs@cs.mu.oz.au)
             Andrew Turpin     (aht@cs.mu.oz.au)

Purpose:    Data compression and revised arithmetic coding method.
            Including modified Fenwick structure so coding is on-line.

Based on: 
        A. Moffat, R. Neal, I.H. Witten, "Arithmetic Coding Revisited",
        Proc. IEEE Data Compression Conference, Snowbird, Utah, 
        March 1995.
        
        A. Moffat, "An improved data structure for cummulative 
        probability tables", Software-Practice and Experience, 1998.  
        To appear.

Copyright 1995 John Carpinelli and Wayne Salamonsen, All Rights Reserved.
Copyright 1996, Lang Stuiver.  All Rights Reserved.
Copyright 1998, Andrew Turpin.  All Rights Reserved.

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
#include "arith.h"
#include "stats.h"

#ifdef RCSID
static char rcsid[] = "$Id: stats.new.c,v 1.1 1998/09/18 00:36:26 aht Exp aht $";
#endif


#ifdef VARY_NBITS
    static freq_value    Max_frequency;
#else
#   define    Max_frequency   ((freq_value) 1 << F_bits)
#endif

/* MOST_PROB_AT_END:
 * Put most probable symbol at end of range (for more accurate approximations)
 * This option influences the compressed data, but is not recorded in the
 * data stream.  It is for experimental purposes and it is recommended
 * that it is left #defined.
 */
#define MOST_PROB_AT_END

#ifdef MOST_PROB_AT_END
    char *stats_desc = "Cumulative stats with Moffat tree (MPS at front)";
#else
    char *stats_desc = "Cumulative stats with Moffat tree";
#endif

static void get_interval(context *pContext,
                 freq_value *pLow, freq_value *pHigh, int symbol);
static void halve_context(context *pContext);


/* INCR_SYMBOL_PROB increments the specified symbol probability by the 'incr'
 * amount.  If the most probable symbol is maintined at the end of the coding
 * range (MOST_PROB_AT_END #defined), then both INCR_SYMBOL_PROB_ACTUAL and
 * INCR_SYMBOL_PROB_MPS are used.  Otherwise, just INCR_SYMBOL_PROB_ACTUAL
 * is used.
 */
/* INCR_SYMBOL_PROB_ACTUAL:  Increment 'symbol' in 'pContext' by inc1' */

#define INCR_SYMBOL_PROB_ACTUAL(pContext, symbol, inc1) \
   freq_value _inc = (inc1);						    \
   freq_value *_tree = pContext->tree;                  \
   int p=symbol;                                        \
                    /* Increment stats */               \
   while (p > 0) {                                      \
        _tree[p] += _inc;                               \
        p = BACK(p);                                    \
   }                                                    \
   pContext->total += _inc;                             \

/* 
 * INCR_SYMBOL_PROB_MPS: Update most frequent symbol, assuming 'symbol'
 * in 'pContext' was just incremented.
 */

/* Assumes _inc already set by macro above */
/* And _low and _high set before also */
#define INCR_SYMBOL_PROB_MPS(pContext, symbol)                    \
 {                                                               \
   if (symbol == pContext->most_freq_symbol)                      \
    pContext->most_freq_count += _inc;                            \
   else if ((_high)-(_low)+(_inc) > pContext->most_freq_count)    \
    { pContext->most_freq_symbol = symbol;                        \
      pContext->most_freq_count = (_high) - (_low) + (_inc);      \
      pContext->most_freq_pos   = _low;                           \
    }                                                             \
   else if (symbol < pContext->most_freq_symbol)                  \
    pContext->most_freq_pos += _inc;                              \
  }

/* 
 * Define INCR_SYMBOL_PROB.  Definition depends on whether most probable
 * symbol needs to be remembered
 */
#ifdef MOST_PROB_AT_END
#  define INCR_SYMBOL_PROB(pContext, symbol, low1, high1, inc1)		\
   do {                                            			\
	freq_value _low = low1;						\
	freq_value _high = high1;					\
	INCR_SYMBOL_PROB_ACTUAL(pContext, symbol, inc1)			\
	INCR_SYMBOL_PROB_MPS   (pContext, symbol)			\
      } while (0)

#else
#  define INCR_SYMBOL_PROB(pContext, symbol, low1, high1, inc1)		\
   do {                                            			\
	INCR_SYMBOL_PROB_ACTUAL(pContext, symbol, inc1)			\
      } while (0)
#endif

#define MIN(a,b) ((a) < (b) ? (a) : (b))

#define GET_COUNT(pContext, symbol, c)                          \
   do {                                                         \
       if ((symbol) & 1) c = pContext->tree[symbol];            \
       else {                                                   \
           int q = symbol + 1;                                  \
           int z = MIN(FORW(symbol), pContext->max_length + 1); \
           c     = pContext->tree[symbol];                      \
           while (q < z) {                                      \
               c -= pContext->tree[q];                          \
               q  = FORW(q);                                    \
           }                                                    \
       }                                                        \
   } while (0)

/* 
 * Zero frequency probability is specified as a count out of the total
 * frequency count.  It is stored as the first item in the tree (item
 * 1).  Luckily, a Moffat tree is defined such that we can read the
 * value of item 1 directly (pContext->tree[1]), although it cannot be
 * updated directly.  After each symbol is coded, adjust_zero_freq() is
 * called to ensure that the zero frequency probability stored in the
 * tree is still accurate (and changes it if it has changed).
 */
#define adjust_zero_freq(pContext)					     \
do { freq_value diff;							     \
	diff = ZERO_FREQ_PROB(pContext) - pContext->tree[1];		     \
	if (diff != 0)							     \
		INCR_SYMBOL_PROB(pContext, 1, 0, pContext->tree[1], diff);   \
   } while (0)

/* 
 * ZERO_FREQ_PROB defines the count for the escape symbol.  We
 * implemented a variation of the XC method (which we call AX).  We
 * create a special escape symbol, which we keep up to date with the
 * count of "number of singletons + 1".  To achieve this, but still be
 * efficient with static contexts, we falsely increment the number of
 * singletons at the start of modelling for dynamic contexts, and keep
 * it at 0 for static contexts.  This way, nSingletons is always our
 * zero frequency probability, without the need to check if the context
 * is static or dynamic (remember that this operation would be done for
 * each symbol coded).
 */

/* Zero frequency symbol probability for context `ctx' */
#define ZERO_FREQ_PROB(ctx)   ((freq_value)ctx->nSingletons)

void init_zero_freq(context *pContext)
{
    /* nSingletons is now actually nSingletons + 1, but this means
       we can use nSingletons directly as zero_freq_prob (see above) */
    if (pContext->type  == DYNAMIC)
        pContext->nSingletons += pContext->incr;
    else
        pContext->nSingletons = 0;
}

/*
 *
 * Create a new frequency table using a binary index tree.
 * Table may be STATIC or DYNAMIC depending on the type parameter.
 * DYNAMIC tables may grow above their intial length as new symbols
 * are installed.
 *
 * max_length is set to 2^ceil(log_2 length).
 * Valid tree indicies are 1..max_length-1.
 * (max_length, refers to the maximum length of the structure before it
 *  needs to expand)
 */

context *create_context(int length, int type)
{
    context    *pContext;
    int        i;
    int        size = 1;

#ifdef VARY_NBITS
    /* Ensure max frequency set up.  */
    Max_frequency = ((freq_value) 1 << F_bits);
#endif

    /*
     * increment length to accommodate the fact 
     * that symbol 0 is stored at pos 2 in the array.
     * (Escape symbol is at pos 1, pos 0 is not used).
     */
    length+=2;

    /* round length up to next power of two */
    while (size < length-1)
    size = size << 1;

    /* malloc context structure and array for frequencies */
    if (((pContext = (context *) malloc(sizeof(context))) == NULL) ||
    ((pContext->tree = (freq_value *) malloc((size+1)*sizeof(freq_value)))
                                == NULL))
    {
        fprintf(stderr, "stats: not enough memory to create context\n");
        exit(1);
    }
    pContext->initial_size = size;    /* save for purging later */
    pContext->length = 1;             /* current no. of symbols */
    pContext->total = 0;              /* total frequency */
    pContext->nSymbols = 1;           /* count of symbols */
    pContext->type = type;            /* is context DYNAMIC or STATIC */
    pContext->max_length = size;      /* no. symbols before growing */

    pContext->most_freq_symbol = -1;  /* Initially no most_freq_symbol */
    pContext->most_freq_count = 0;
    pContext->most_freq_pos = 0;
    
    /* initialise contents of tree array to zero */
    for (i = 0; i <= pContext->max_length; i++)
    pContext->tree[i] = 0;
                    /* increment is initially 2 ^ f */
    pContext->incr = (freq_value) 1 << F_bits; 
    pContext->nSingletons = 0;

    init_zero_freq(pContext);
    adjust_zero_freq(pContext);

    return pContext;                /* return a pointer to the context */
}


/*
 *
 * install a new symbol in a context's frequency table
 * returns 0 if successful, TOO_MANY_SYMBOLS or NO_MEMORY if install fails
 *
 */

int install_symbol(context *pContext, int symbol)
{
    int i;
    freq_value low, high;

        /* Increment because first user symbol (symbol 0)
            is stored at array position 2 */
    symbol+=2;

    /* 
     * if new symbol is greater than current array length then double length 
     * of array 
     */    
    while (symbol > pContext->max_length) 
    {
        pContext->tree = (freq_value *) 
            realloc(pContext->tree, 
                    (2*pContext->max_length+1) * sizeof(freq_value));
        if (pContext->tree == NULL)
        {
            fprintf(stderr, "stats: not enough memory to expand context\n");
            return NO_MEMORY;
        }

        /* clear new part of table to zero */
        for (i = pContext->max_length+1; i <= 2*pContext->max_length; i++)
            pContext->tree[i] = 0;
        
        pContext->max_length <<= 1;
    }

    /* check that we are not installing too many symbols */
    if (((pContext->nSymbols + 1) << 1) >= Max_frequency)
    /* 
     * cannot install another symbol as all frequencies will 
     * halve to one and an infinite loop will result
     */
    return TOO_MANY_SYMBOLS;           
    
    if (symbol > pContext->length)    /* update length if necessary */
        pContext->length = symbol;
    pContext->nSymbols++;        /* increment count of symbols */

    get_interval(pContext, &low, &high, symbol);

    /* update the number of singletons if context is DYNAMIC */
    INCR_SYMBOL_PROB(pContext, symbol, low, high, pContext->incr);
    if (pContext->type == DYNAMIC)
        pContext->nSingletons += pContext->incr;

    adjust_zero_freq(pContext);

    /* halve frequency counts if total greater than Max_frequency */
    while (pContext->total > Max_frequency)
        halve_context(pContext);

    return 0;
}/* install_symbol() */



/*
 *
 * encode a symbol given its context
 * the lower and upper bounds are determined using the frequency table,
 * and then passed on to the coder
 * if the symbol has zero frequency, code an escape symbol and
 * return NOT_KNOWN otherwise returns 0
 *
 */
int encode(context *pContext, int symbol)
{
    freq_value low, high, low_w, high_w;

    symbol+=2;
    if ((symbol > 0) && (symbol <= pContext->max_length))
    {
    if (pContext->most_freq_symbol == symbol)
        {
          low  = pContext->most_freq_pos;
          high = low + pContext->most_freq_count;
        }
        else
          get_interval(pContext, &low, &high, symbol);
    }
    else
    low = high = 0;
    
    if (low == high)
      {
    if (ZERO_FREQ_PROB(pContext) == 0) 
    {
        fprintf(stderr,"stats: cannot code zero-probability novel symbol");
        abort();
        exit(1);
    }
    /* encode the escape symbol if unknown symbol */
    symbol = 1;
    if (pContext->most_freq_symbol == 1)
        {
        low = pContext->most_freq_pos;
        high = low + pContext->most_freq_count;
        }
       else
        get_interval(pContext, &low, &high, symbol);
     }

    /* Shift high and low if required so that most probable symbol
     * is at the end of the range
     * (Shifted values are low_w, and high_w, as original values
     * are needed when updating the stats)
     */

#ifdef MOST_PROB_AT_END
    if (symbol > pContext->most_freq_symbol) {
	    low_w  = low  - pContext->most_freq_count;
	    high_w = high - pContext->most_freq_count;
	}
    else if (symbol == pContext->most_freq_symbol) {
	    low_w  = pContext->total - pContext->most_freq_count;
	    high_w = pContext->total;
	} else {
	    low_w  = low;
	    high_w = high;
	}
#else
    low_w  = low;
    high_w = high;
#endif

    /* call the coder with the low, high and total for this symbol
     * (with low_w, high_w:  Most probable symbol moved to end of range)
     */

    arithmetic_encode(low_w, high_w, pContext->total);

    if (symbol != 1)        /* If not the special ESC / NOT_KNOWN symbol */
    {
            /*update the singleton count if symbol was previously a singleton */
        if (pContext->type == DYNAMIC && high-low == pContext->incr)
            pContext->nSingletons -= pContext->incr;

            /* increment the symbol's frequency count */
        INCR_SYMBOL_PROB(pContext, symbol, low, high, pContext->incr);
    }

    adjust_zero_freq(pContext);

    while (pContext->total > Max_frequency)
        halve_context(pContext);

    if (symbol == 1) return NOT_KNOWN;
    return 0;
}/* encode() */




/*
 *
 * decode function is passed a context, and returns a symbol
 *
 */
int 
decode(context *pContext)
{
    int    symbol;
    int p, m, e;
    freq_value low, high, target;

    int            n = pContext->max_length;
    freq_value total = pContext->total;
    freq_value    *M = pContext->tree;
 
    target = arithmetic_decode_target(total);

#ifdef MOST_PROB_AT_END
	/* Check if most probable symbol (shortcut decode)
	 */
    if (target >= total - pContext->most_freq_count) {
	  arithmetic_decode( total - pContext->most_freq_count, total, total);
	  symbol = pContext->most_freq_symbol;
	  low    = pContext->most_freq_pos;
	  high   = low + pContext->most_freq_count;

      INCR_SYMBOL_PROB(pContext, symbol, low, high, pContext->incr);

      if (symbol != 1) 
        if (pContext->type == DYNAMIC && high-low == pContext->incr)
            pContext->nSingletons -= pContext->incr;
	}
    else
	/* Not MPS, have to decode slowly */
  {
	if (target >= pContext->most_freq_pos)
	    target += pContext->most_freq_count;
#endif
    p = 1; low = 0;
    while ( ((p << 1) <= pContext->max_length ) && (M[p] <= target)) {
        target -= M[p];
        low    += M[p];
        p      <<= 1;
    }

    symbol = p;
    m = p >> 1;
    e = 0;

    while (m >= 1) {
        if (symbol + m <= n) {
            e += M[symbol + m];
            if (M[symbol] - e <= target) {
                target    -= M[symbol] - e;
                low       += M[symbol] - e;
                if (symbol != 1) M[symbol] += pContext->incr;
                symbol    += m;
                e         =  0;
            }
        }
        m >>= 1;
    }
    if (symbol!= 1) M[symbol] += pContext->incr;

    if (symbol & 1)
        high = low + pContext->tree[symbol];
    else {
        GET_COUNT(pContext, symbol, high);
        high += low;
    }
    if (symbol != 1) high -= pContext->incr;

#ifdef MOST_PROB_AT_END
    if (low >= pContext->most_freq_pos)   /* Ie: Was moved */
	    arithmetic_decode(low  - pContext->most_freq_count,
			  high - pContext->most_freq_count,
			  total);
    else
#endif
        arithmetic_decode(low, high, total);

    /* update the singleton count if symbol was previously a singleton */
    if (symbol != 1)
    {
      if (pContext->type == DYNAMIC && high-low == pContext->incr)
        pContext->nSingletons -= pContext->incr;

        pContext->total += pContext->incr;

        if (symbol == pContext->most_freq_symbol)
	        pContext->most_freq_count += pContext->incr;
        else if (high-low+pContext->incr > pContext->most_freq_count) { 
            pContext->most_freq_symbol = symbol;
	        pContext->most_freq_count  = high - low + pContext->incr;
	        pContext->most_freq_pos    = low;
	    } else if (symbol < pContext->most_freq_symbol)
	        pContext->most_freq_pos += pContext->incr;
    }

#ifdef MOST_PROB_AT_END
  }  /* If not MPS */
#endif

    adjust_zero_freq(pContext);

    /* halve all frequencies if necessary */
    while (pContext->total > Max_frequency)
        halve_context(pContext);

    if (symbol == 1) return NOT_KNOWN;

    return symbol-2;
}/* decode() */



/*
 *
 * Get the low and high limits of the frequency interval
 * occupied by a symbol.
 *
 */
static void 
get_interval(context *pContext, freq_value *pLow, freq_value *pHigh, int symbol)
{
    freq_value low, count;
    int p, q;
    freq_value *tree = pContext->tree;

        /* go too far */
    for(p = 1, low = 0 ; p < symbol ; ) {
        low  += tree[p], 
        p   <<= 1;
    }

        /* subtract off the extra freqs from low */
    q = symbol;
    while ((q != p) && (q <= pContext->max_length)) {
        low -= tree[q], 
        q   = FORW(q);
    }

    GET_COUNT(pContext, symbol, count);

    *pLow = low;
    *pHigh = low + count;
}
 

/*
 *
 * Halve_context is responsible for halving all the frequency counts in a 
 * context.
 * Halves context in linear time by converting tree to list of freqs
 * then back again.
 *
 * It ensures the most probable symbol size and range stay updated.
 * If halving the context gives rise to a sudden drop in the
 * ZERO_FREQ_PROB(), and if it was the MPS, it will stay recorded as the most
 * probable symbol even if it isn't.  This may cause slight compression
 * inefficiency.  (The ZERO_FREQ_PROB() as implemented does not have this
 * characteristic, and in this case the inefficiency cannot occur)
 */

static void
halve_context(context *pContext)
{
    int  shifts, p, symbol;
    freq_value incr;

    pContext->incr = (pContext->incr + MIN_INCR) >> 1;    /* halve increment */
    if (pContext->incr < MIN_INCR) 
            pContext->incr = MIN_INCR;
    pContext->nSingletons = incr = pContext->incr;
    
        /*
        ** Convert Moffat tree to array of freqs
        */
    for (shifts=0 , p = pContext->max_length ; p > 1 ; shifts++ ) p >>= 1;
    p  = 1 << shifts;      /* p is now to 2^floor(log_2 pContext->max_length) */
    while( p > 1 ) {
        symbol = p;
        while (symbol + (p >> 1) <= pContext->max_length ) {
            pContext->tree[symbol] -= pContext->tree[symbol + (p >> 1)];
            symbol                 += p;
        }
        p >>= 1;
    }

        /*
        ** Halve the counts (ignore tree[1] as it will be changed soon)
        */
    pContext->total = 0;
    for (p = 2; p <= pContext->max_length; p++) {
        pContext->tree[p] = (pContext->tree[p] + 1) >> 1;
        pContext->total  += pContext->tree[p];
        if (pContext->tree[p] == incr)
            pContext->nSingletons += incr;
    }

        /*
        ** Convert array of freqs to Moffat tree
        */
    for (p = 2; p <= pContext->max_length; ) {
        symbol = p;
        while (symbol + (p >> 1) <= pContext->max_length ) {
            pContext->tree[symbol] += pContext->tree[symbol + (p >> 1)];
            symbol                 += p;
        }
        p <<= 1;
    }

    if (pContext->type == STATIC)
        pContext->nSingletons = 0;

    pContext->tree[1] = ZERO_FREQ_PROB(pContext);
    pContext->total  += ZERO_FREQ_PROB(pContext);

    /* Recalc new most_freq_symbol info if it exists
     * (since roundoff may mean not exactly half of previous value)
     */

    if (pContext->most_freq_symbol != -1) 
      { freq_value low, high;
        get_interval(pContext, &low, &high, pContext->most_freq_symbol);
        pContext->most_freq_count = high-low;
        pContext->most_freq_pos = low;
      }

    adjust_zero_freq(pContext);
}/* halve_context() */


/*
 *
 * free memory allocated for a context and initialize empty context
 * of original size
 *
 */
void 
purge_context(context *pContext)
{
    int i;

    free(pContext->tree);
    
    /* malloc new tree of original size */
    if ((pContext->tree = (freq_value *)malloc((pContext->initial_size + 1)
                        * sizeof(freq_value))) == NULL)
    {
    fprintf(stderr, "stats: not enough memory to create context\n");
    exit(1);
    }
    pContext->length = 1;
    pContext->total = 0;
    pContext->nSymbols = 1;        /* Start with escape symbol */

    pContext->most_freq_symbol = -1;    /* Indicates no such symbol */
    pContext->most_freq_count = 0;
    pContext->most_freq_pos = 0;

    pContext->max_length = pContext->initial_size;
    for (i = 0; i <= pContext->initial_size; i++)
	    pContext->tree[i] = 0;
                      /* increment is initially 2 ^ f */
    pContext->incr = (freq_value) 1 << F_bits;
    pContext->nSingletons = 0;

    init_zero_freq(pContext);
    adjust_zero_freq(pContext);
}

/******************************************************************************
*
* functions for binary contexts
*
******************************************************************************/

/*
 *
 * create a binary_context
 * binary contexts consist of two counts and an increment which
 * is normalized
 *
 */
binary_context *create_binary_context(void)
{
    binary_context *pContext;

#ifdef VARY_NBITS
    Max_frequency = ((freq_value) 1 << F_bits);
#endif

    pContext = (binary_context *) malloc(sizeof(binary_context));
    if (pContext == NULL)
    {
    fprintf(stderr, "stats: not enough memory to create context\n");
    exit(1);
    }
                        /* start with incr=2^(f-1) */
    pContext->incr = (freq_value) 1 << (F_bits - 1);
    pContext->c0 = pContext->incr;
    pContext->c1 = pContext->incr;
    return pContext;
}



/*
 *
 * encode a binary symbol using special binary arithmetic
 * coding functions
 * returns 0 if successful
 *
 */
int
binary_encode(binary_context *pContext, int bit)
{
    binary_arithmetic_encode(pContext->c0, pContext->c1, bit);

    /* increment symbol count */
    if (bit == 0)
    pContext->c0 += pContext->incr;
    else
    pContext->c1 += pContext->incr;

    /* halve frequencies if necessary */
    if (pContext->c0 + pContext->c1 > Max_frequency)
    {
    pContext->c0 = (pContext->c0 + 1) >> 1;
    pContext->c1 = (pContext->c1 + 1) >> 1;
    pContext->incr = (pContext->incr + MIN_INCR) >> 1;
    }
    return 0;
}    



/*
 *
 * decode a binary symbol using specialised binary arithmetic
 * coding functions
 *
 */
int
binary_decode(binary_context *pContext)
{
    int bit;

    bit = binary_arithmetic_decode(pContext->c0, pContext->c1);

    /* increment symbol count */
    if (bit == 0)
    pContext->c0 += pContext->incr;
    else
    pContext->c1 += pContext->incr;

    /* halve frequencies if necessary */
    if (pContext->c0 + pContext->c1 > Max_frequency)
    {
    pContext->c0 = (pContext->c0 + 1) >> 1;
    pContext->c1 = (pContext->c1 + 1) >> 1;
    pContext->incr = (pContext->incr + MIN_INCR) >> 1;
    }    
    return bit;
}
