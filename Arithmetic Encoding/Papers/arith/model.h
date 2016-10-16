/*INTERFACE TO THE MODEL. */
/*THE SET OF SYMBOLS THAT MAY BE ENCODED.*/

#define No_of_chars 256 /* Number of character symbols*/
#define EOF-symbol (No_of_chars+l) /* Index of EOF symbol*/

#define No_of_symbols (No_of_chars+l) /*Total number of symbols*/

/* TRANSLATION TABLES BETWEEN CHARACTERS AND SYMBOL INDEXES.*/
int char-to-index[No-of-chars]; /* To index from character */
unsigned char index_to_char[No_of_symbols+l]: /* To character from index */

/* CUMULATIVE FREQUENCY TABLE. */
#define Max_frequency 16383				/* Maximum allowed frequency count */
										/* 2^14 - 1*/
int cum_frsq[No_of_symbols+l];		 	/* Cumulative symbol frequencies l /