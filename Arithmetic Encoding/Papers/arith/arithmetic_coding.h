/* DECLARATIONS USED FOR ARITHMETIC ENCODING AND DECODING l /
/* SIZE OF ARITHMETIC CODE VALUES. l /

 #define Code-value-bits 16 /* Number of bits in a code value l /
 typedef long code-value: /* Type of an arithmetic code value l /

#define Top-value (((long)l<<Code_value_blts)-1) /* Largest code value l /


 /' HALF AND QUARTER POINTS IN THE CODE VALUE RANGE. l /

 #define First-qtr (Top-value/ltl) /* Point after first quarter l /
 # define Half (Z'First-qtr) /* Point after first half "/
 # define Third-qtr (3’Firat-qtr) /* Point after third quarter l 

mode1.h
/' INTERFACE TO THE MODEL. '/
/' THE SET OF SYMBOLS THAT MAY BE ENCODED. l /
#define No-of-chars 256 /* Number of character symbols '/
#define EOF-symbol (No-of-charetl) /* Index of EOF symbol '/
#define No-of-symbols (No-of-charstll /* Total number of symbols */
/' TRANSLATION TABLES BETWEEN CHARACTERS AND SYMBOL INDEXES. l /
int char-to-index[No-of-chars]; /* To index from character '/
unsigned char index_to_char[No_of_symbols+l]: /* To character from index l /
/* CUMULATIVE FREQUENCY TABLE. */
Idefine Max-frequency 16383
int cum_frsq[No_of_symbols+l];
/* Maximum allowed frequency count l /
/* 2a14 - 1 l /
/* Cumulative symbol frequencies l /