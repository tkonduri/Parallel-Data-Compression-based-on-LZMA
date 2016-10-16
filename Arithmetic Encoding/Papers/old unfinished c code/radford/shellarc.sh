#! /bin/sh
# This is a shell archive.  Remove anything before this line, then unpack
# it by saving it into a file and typing "sh file".  To overwrite existing
# files, use "sh file -c".
# If this archive is complete, you will see the following message at the end:
#		"End of shell archive."
#
# Contents:
#   README Makefile code.h model.h bit_io.c code_mul.c
#   code_sft.c decode.c decpic.c encode.c encpic.c model.c
#   redundancy.c tstpic
#
# Wrapped by radford@ai.utoronto.ca on Thu May 20 19:54:12 1993
#
PATH=/bin:/usr/bin:/usr/ucb ; export PATH
if test -f README -a "${1}" != "-c" ; then 
  echo shar: Will not over-write existing file \"README\"
else
echo shar: Extracting \"README\" \(6323 characters\)
sed "s/^X//" >README <<'END_OF_README'
X
X
X            LOW-PRECISION ARITHMETIC CODING IMPLEMENTATION
X
X                          Radford M. Neal
X
X
X                    Initial release:  8 July 1991
X               Documentation update: 16 July 1991
X                            Bug fix: 25 July 1991
X                            Bug fix: 16 Sept 1992
X   Changes for ANSI C compatibility: 29 October 1992
X                            Bug fix: 19 May 1993
X
X
XThis directory contains C source for an implementation of arithmetic
Xcoding using low-precision division. This division can be performed
Xusing shift/add operations, with potential savings in time on any
Xmachine without parallel multiply/divide hardware.
X
XThe implementation is based on that in the paper of Witten, Neal, and
XCleary published in the June 1987 Communications of the ACM. Anyone
Xwishing to understand this program is urged to first read that paper.
XDifferences in this version are as follows:
X
X    1) The arithmetic coding operations have been fiddled so that
X       the division involved can be done to very low precision.
X       There is a tradeoff between precision and compression performance,
X       but nearly optimal results are obtained with a precision of
X       six bits, and precisions of as low as three bits give reasonable 
X       results. A precision of at least two bits is required for
X       correct operation.
X
X    2) In order for (1) to be possible, the model is now required
X       to produce "partially normalized" frequencies, in which the
X       total for all symbols is always more than half the maximum 
X       allowed total. This is not onerous, at least for the models
X       used here.
X
X    3) The model must also now arrainge for the most probable symbol
X       to have index 1. This was always the case, but previously
X       this was solely a matter of time efficiency. Now, failure
X       to do this would impact compression efficiency, though not 
X       correct operation.
X
X    4) The precision to which symbol frequencies may be held is much
X       higher in this implementation - 27 bits with the default
X       parameter settings. The CACM implementation was restricted
X       to 14 bit frequencies. This is of significance in applications
X       where the number of symbols is large, such as with word-based
X       models.
X
X    5) Encode/decode procedures specialized for use with a two-symbol
X       alphabet have been added. These are demonstrated by a simple
X       adaptive image compression program.
X
X    6) Various minor modifications and structural changes to the
X       program have been made.
X
XTwo versions of the coding procedures are provided - one using C
Xmultiply and divide operators, the other using shifts and adds. These
Xversions, and the resulting encode/decode programs, are distinguished
Xby the suffixes "_mul" and "_sft". Which version is fastest will
Xdepend on the particular machine and compiler used. All encode/decode
Xprograms simply read from standard input and write to standard output.
X
XThe file 'tstpic' contains a test picture for the image
Xencoding/decoding programs. The format of such pictures may be
Xdiscerned by examination of this example, and of the program code.
X
XA program for calculating a bound on maximum expected redundancy 
Xresulting from low-precision division is included. Typical redundancy
Xis much less than this bound.
X
XFor the multiply/divide version, the requirement that the model
Xproduce partially normalized frequencies is not really necessary.
X
XThe program is intended for demonstation purposes. It is not optimized
Xfor time efficiency, and provides little in the way of error checking.
X
XThe method used in this program has some resemblences to that presented
Xby Rissanen and Mohiuddin in "A Multiplication-Free Multialphabet Arithmetic
XCode", IEEE Transactions on Communications, February, 1989. The main
Xsimilarities are the following:
X
X     1) The idea of constraining the size of the coding region and the
X        range of the occurrence counts so as to allow an approximation.
X
X     2) The placement of the most-probable symbol at the top of the
X        coding region.
X
XThere are a number of significant dissimilarities, however. The details
Xof the constraints mentioned above are different. The low-precision
Xmethod implemented here is more general, giving a smooth trade-off between
Xcompression performance and speed through choice of precision for the
Xmultiplication and division. Other unique features of this code include:
X
X     1) Incremental maintenance of partialy-normalized occurrence counts,
X        eliminating the need for such normalization in the coding process,
X        as is the case with the Rissanen and Mohiuddin method.
X
X     2) Merging of multiply and divide operations for faster operation
X        with serial arithmetic (not relevant in the less-general Rissanen 
X        and Mohiuddin method).
X
X     3) A variable-precision computation in order to locate the next
X        symbol in the non-binary decode procedure (also not relevant in
X        the Rissanen and Mohiuddin method).
X
XThe Rissanen and Mohiuddin method should be somewhat faster than that
Xused here. Its coding efficiency appears similar to that which would be
Xobtained with this method if divisions are performed to a precision of 
Xtwo bits.
X
XThe detailed algorithm presented in the paper by Rissanen and Mohiuddin
Xuses the supposedly patented "bit stuffing" procedure. This procedure
Xis _not_ used in this code. 
X
XThis code is public domain, and may be used by anyone for any purpose.
XI do, however, expect that distributions utilizing this code will
Xinclude an acknowledgement of its source. The program is provided
Xwithout any warranty as to correct operation. The Rissanen and
XMohiuddin method is said to be patented, and I can offer no guarantees
Xas to whether use of the code presented here might infringe those
Xpatents (off hand, this would seem to be a complex question with no
Xdefinitive answer). My amateur understanding of patent law leads me to
Xbelieve that use for research purposes would be permitted under any
Xcircumstances, but I could well be deluded in this regard.
X
XAddress comments to:    
X   
X       Radford Neal
X       Dept. of Computer Science
X       University of Toronto
X       10 King's College Road
X       Toronto, Ontario, CANADA
X       M5S 1A4
X
X       e-mail: radford@ai.toronto.edu
END_OF_README
if test 6323 -ne `wc -c <README`; then
    echo shar: \"README\" unpacked with wrong size!
fi
# end of overwriting check
fi
if test -f Makefile -a "${1}" != "-c" ; then 
  echo shar: Will not over-write existing file \"Makefile\"
else
echo shar: Extracting \"Makefile\" \(1530 characters\)
sed "s/^X//" >Makefile <<'END_OF_Makefile'
X# MAKEFILE FOR LOW-PRECISION ARITHMETIC CODING PROGRAMS.
X
X
XCFLAGS = -O
X
Xall:			encode_mul decode_mul encode_sft decode_sft \
X			encpic_mul decpic_mul encpic_sft decpic_sft
X
Xlowp_ac.shar:		Makefile tstpic model.h code.h \
X			encode.c decode.c model.c encpic.c decpic.c \
X			code_mul.c code_sft.c bit_io.c 
X	shar -o lowp_ac.shar Makefile tstpic model.h code.h \
X		encode.c decode.c model.c encpic.c decpic.c \
X		code_mul.c code_sft.c bit_io.c redundancy.c
X
X
Xencode_mul:		encode.o model.o bit_io.o code_mul.o
X	cc -O encode.o model.o bit_io.o code_mul.o -o encode_mul
X
Xdecode_mul:		decode.o model.o bit_io.o code_mul.o
X	cc -O decode.o model.o bit_io.o code_mul.o -o decode_mul
X
Xencode_sft:		encode.o model.o bit_io.o code_sft.o
X	cc -O encode.o model.o bit_io.o code_sft.o -o encode_sft
X
Xdecode_sft:		decode.o model.o bit_io.o code_sft.o
X	cc -O decode.o model.o bit_io.o code_sft.o -o decode_sft
X
X
Xencpic_mul:		encpic.o bit_io.o code_mul.o
X	cc -O encpic.o bit_io.o code_mul.o -o encpic_mul
X
Xdecpic_mul:		decpic.o bit_io.o code_mul.o
X	cc -O decpic.o bit_io.o code_mul.o -o decpic_mul
X
Xencpic_sft:		encpic.o bit_io.o code_sft.o
X	cc -O encpic.o bit_io.o code_sft.o -o encpic_sft
X
Xdecpic_sft:		decpic.o bit_io.o code_sft.o
X	cc -O decpic.o bit_io.o code_sft.o -o decpic_sft
X
X
Xencode.o:		encode.c model.h code.h
Xdecode.o:		decode.c model.h code.h
Xmodel.o:		model.c model.h code.h
X
Xencpic.o:		encpic.c model.h code.h
Xdecpic.o:		decpic.c model.h code.h
X
Xcode_mul.o:		code_mul.c code.h
Xcode_sft.o:		code_sft.c code.h
Xbit_io.o:		bit_io.c code.h
END_OF_Makefile
if test 1530 -ne `wc -c <Makefile`; then
    echo shar: \"Makefile\" unpacked with wrong size!
fi
# end of overwriting check
fi
if test -f code.h -a "${1}" != "-c" ; then 
  echo shar: Will not over-write existing file \"code.h\"
else
echo shar: Extracting \"code.h\" \(1609 characters\)
sed "s/^X//" >code.h <<'END_OF_code.h'
X/* CODE.H - DECLARATIONS USED FOR ARITHMETIC ENCODING AND DECODING */
X
X
X/* PRECISION OF CODING VALUES. Coding values are fixed-point binary 
X   fractions in the range [0,1), represented as integers scaled by 
X   2^Code_bits. */
X
X#define Code_bits 32			/* Number of bits for code values   */
X#define Code_quarter (1<<(Code_bits-2))	/* Quarter point for code values    */
X#define Code_half (1<<(Code_bits-1))	/* Halfway point for code values    */
X
Xtypedef unsigned code_value;		/* Data type holding code values    */
X
X
X/* PRECISION OF SYMBOL PROBABILITIES. Symbol probabilities must be partially
X   normalized, so that the total for all symbols is in the range (1/2,1].
X   These probabilities are represented as integers scaled by 2^Freq_bits. */
X
X#define Freq_bits 27			/* Number of bits for frequencies   */
X#define Freq_half (1<<(Freq_bits-1))	/* Halfway point for frequencies    */
X#define Freq_full (1<<Freq_bits)	/* Largest frequency value          */
X
Xtypedef unsigned freq_value;		/* Data type holding frequencies    */
X
X
X/* PRECISION OF DIVISION. Division of a code value by a frequency value 
X   gives a result of precision (Code_bits-Freq_bits), which must be at
X   least two for correct operation (larger differences give smaller code
X   size). */
X
Xtypedef unsigned div_value;		/* Data type for result of division */
X
X
X/* PROCEDURES. */
X
Xvoid start_encoding (void);
Xvoid start_decoding (void);
Xvoid encode_symbol  (int, freq_value []);
Xvoid encode_bit     (int, freq_value, freq_value);
Xint  decode_symbol  (freq_value []);
Xint  decode_bit     (freq_value, freq_value);
Xvoid done_encoding  (void);
END_OF_code.h
if test 1609 -ne `wc -c <code.h`; then
    echo shar: \"code.h\" unpacked with wrong size!
fi
# end of overwriting check
fi
if test -f model.h -a "${1}" != "-c" ; then 
  echo shar: Will not over-write existing file \"model.h\"
else
echo shar: Extracting \"model.h\" \(930 characters\)
sed "s/^X//" >model.h <<'END_OF_model.h'
X/* MODEL.H - INTERFACE TO THE MODEL. */
X
X
X/* THE SET OF SYMBOLS THAT MAY BE ENCODED. Symbols are indexed by integers
X   from 1 to No_of_symbols. */
X
X#define No_of_chars 256			/* Number of character symbols      */
X#define EOF_symbol (No_of_chars+1)	/* Index of EOF symbol              */
X
X#define No_of_symbols (No_of_chars+1)	/* Total number of symbols          */
X
X
X/* TRANSLATION TABLES BETWEEN CHARACTERS AND SYMBOL INDEXES. */
X
Xglobal int char_to_index[No_of_chars];	/* To index from character          */
Xglobal unsigned char index_to_char[No_of_symbols+1]; /* To char from index  */
X
X
X/* CUMULATIVE FREQUENCY TABLE. Cumulative frequencies are stored as
X   partially normalized counts. The normalization factor is cum_freq[0],
X   which must lie in the range (1/2,1]. */
X
Xglobal freq_value cum_freq[No_of_symbols+1]; /* Cumulative symbol frequencies */
X
X
X/* PROCEDURES. */
X
Xvoid start_model  (void);
Xvoid update_model (int);
END_OF_model.h
if test 930 -ne `wc -c <model.h`; then
    echo shar: \"model.h\" unpacked with wrong size!
fi
# end of overwriting check
fi
if test -f bit_io.c -a "${1}" != "-c" ; then 
  echo shar: Will not over-write existing file \"bit_io.c\"
else
echo shar: Extracting \"bit_io.c\" \(1713 characters\)
sed "s/^X//" >bit_io.c <<'END_OF_bit_io.c'
X/* BIT_IO.C - BIT INPUT/OUTPUT ROUTINES. */
X
X#include <stdio.h>
X
X#define global extern
X
X#include "code.h"
X
X
X/* THE BIT BUFFER. */
X
Xstatic int buffer;		/* Bits waiting to be input                 */
Xstatic int bits_to_go;		/* Number of bits still in buffer           */
X
Xstatic int garbage;		/* Number of garbage bytes after EOF        */
X
X
X/* INITIALIZE FOR BIT OUTPUT. */
X
Xstart_outputing_bits()
X{
X    buffer = 0;					/* Buffer is empty to start */
X    bits_to_go = 8;				/* with.                    */
X}
X
X
X/* INITIALIZE FOR BIT INPUT. */
X
Xstart_inputing_bits()
X{
X    bits_to_go = 0;				/* Buffer starts out with   */
X    garbage = 0;				/* no bits in it.           */
X}
X
X
X/* OUTPUT A BIT. */
X
Xoutput_bit(bit)
X    int bit;
X{
X    if (bits_to_go==0) {			/* Output buffer if it is   */
X        putc(buffer,stdout);			/* full.                    */
X        bits_to_go = 8;
X    }
X
X    buffer >>= 1;		 		/* Put bit in top of buffer.*/
X    if (bit) buffer |= 0x80;
X    bits_to_go -= 1;
X}
X
X
X/* INPUT A BIT. */
X
Xint input_bit()
X{
X    int t;
X
X    if (bits_to_go==0) {			/* Read next byte if no     */
X        buffer = getc(stdin);			/* bits left in the buffer. */
X        bits_to_go = 8;
X        if (buffer==EOF) {  			/* Return anything after    */
X            if (garbage*8>=Code_bits) {		/* end-of-file, but not too */
X                fprintf(stderr,"Bad input file (2)\n"); /* much of anything.*/
X                exit(1);
X            }
X            garbage += 1;
X        }
X    }
X
X    t = buffer&1;				/* Return the next bit from */
X    buffer >>= 1;				/* the bottom of the byte.  */
X    bits_to_go -= 1;
X    return t;	
X}
X
X
X/* FLUSH OUT THE LAST BITS. */
X
Xdone_outputing_bits()
X{
X    putc(buffer>>bits_to_go,stdout);
X}
END_OF_bit_io.c
if test 1713 -ne `wc -c <bit_io.c`; then
    echo shar: \"bit_io.c\" unpacked with wrong size!
fi
# end of overwriting check
fi
if test -f code_mul.c -a "${1}" != "-c" ; then 
  echo shar: Will not over-write existing file \"code_mul.c\"
else
echo shar: Extracting \"code_mul.c\" \(8327 characters\)
sed "s/^X//" >code_mul.c <<'END_OF_code_mul.c'
X/* CODE_MUL.C - ARITHMETIC ENCODING/DECODING, USING MULTIPLY/DIVIDE. */
X
X#include <stdio.h>
X
X#define global extern
X
X#include "code.h"
X
X
X/* CURRENT STATE OF THE ENCODING/DECODING. */
X
Xstatic code_value low;		/* Start of current coding region, in [0,1) */
X
Xstatic code_value range;	/* Size of region, normally kept in the     */
X				/* interval [1/4,1/2]                       */
X
Xstatic code_value value;	/* Currently-seen code value, in [0,1)      */
X
Xstatic int bits_to_follow;	/* Number of opposite bits to output after  */
X				/* the next bit                             */
X
Xstatic div_value F;		/* Common factor used, in interval [1/4,1)  */
X
Xstatic code_value split_region();
X
X
X/* LOCAL PROCEDURES. */
X
Xstatic int        find_symbol   (freq_value []);
Xstatic void       narrow_region (freq_value [], int, int);
Xstatic code_value split_region  (freq_value, freq_value);
Xstatic void       push_out_bits (void);
Xstatic void       discard_bits  (void);
X
X
X/* OUTPUT BIT PLUS FOLLOWING OPPOSITE BITS. */
X
X#define bit_plus_follow(bit) \
Xdo { \
X    output_bit(bit);				/* Output the bit.          */\
X    while (bits_to_follow>0) {			/* Output bits_to_follow    */\
X        output_bit(!(bit));			/* opposite bits. Set       */\
X        bits_to_follow -= 1;			/* bits_to_follow to zero.  */\
X    } \
X} while (0) 
X
X
X/* START ENCODING A STREAM OF SYMBOLS. */
X
Xvoid start_encoding(void)
X{   
X    low = Code_half;				/* Use half the code region */
X    range = Code_half;				/* (wastes first bit as 1). */
X    bits_to_follow = 0;				/* No opposite bits follow. */
X}
X
X
X/* START DECODING A STREAM OF SYMBOLS. */
X
Xvoid start_decoding(void)
X{   
X    int i;
X
X    value = input_bit();			/* Check that first bit     */
X    if (value!=1) {				/* is a 1.                  */
X        fprintf(stderr,"Bad input file (1)\n");
X        exit(1);
X    }
X
X    for (i = 1; i<Code_bits; i++) { 		/* Fill the code value with */
X        value += value;				/* input bits.              */
X        value += input_bit();
X    }
X
X    low = Code_half;				/* Use half the code region */
X    range = Code_half;				/* (first bit must be 1).   */
X}
X
X
X/* ENCODE A SYMBOL. */
X
Xvoid encode_symbol(symbol,cum_freq)
X    int symbol;			/* Symbol to encode                         */
X    freq_value cum_freq[];	/* Cumulative symbol frequencies            */
X{
X    narrow_region(cum_freq,symbol,0);		/* Narrow coding region.    */
X    push_out_bits();				/* Output determined bits.  */
X}
X
X
X/* ENCODE A BIT. */
X
Xvoid encode_bit(bit,freq0,freq1)
X    int bit;			/* Bit to encode (0 or 1)                   */
X    freq_value freq0;           /* Frequency for 0 bit                      */
X    freq_value freq1;           /* Frequency for 1 bit                      */
X{
X    code_value split;
X
X    if (freq1>freq0) {				/* Encode bit when most     */
X        split = split_region(freq0,freq1);	/* probable symbol is a 1.  */
X        if (bit) { low += split; range -= split; }
X        else     { range = split;  }
X    }
X
X    else {					/* Encode bit when most     */
X        split = split_region(freq1,freq0);	/* probable symbol is a 0.  */
X        if (bit) { range = split;  }
X        else     { low += split; range -= split; }
X    }
X
X    push_out_bits();				/* Output determined bits.  */
X}
X
X
X/* DECODE THE NEXT SYMBOL. */
X
Xint decode_symbol(cum_freq)
X    freq_value cum_freq[];	/* Cumulative symbol frequencies            */
X{
X    int symbol;			/* Symbol decoded                           */
X
X    symbol = find_symbol(cum_freq);		/* Find decoded symbol.     */
X    narrow_region(cum_freq,symbol,1);		/* Narrow coding region.    */
X    discard_bits();				/* Discard output bits.     */
X
X    return symbol;
X}
X
X
X/* DECODE A BIT. */
X
Xint decode_bit(freq0,freq1)
X    freq_value freq0;           /* Frequency for 0 bit                      */
X    freq_value freq1;           /* Frequency for 1 bit                      */
X{
X    code_value split;
X    int bit;
X
X    if (freq1>freq0) {				/* Decode bit when most     */
X        split = split_region(freq0,freq1);	/* probable symbol is a 1.  */
X        bit = value-low >= split;
X        if (bit) { low += split; range -= split; }
X        else     { range = split;  }
X    }
X
X    else {					/* Decode bit when most     */
X        split = split_region(freq1,freq0);	/* probable symbol is a 0.  */
X        bit = value-low < split;
X        if (bit) { range = split;  }
X        else     { low += split; range -= split; }
X    }
X
X    discard_bits();				/* Discard output bits.     */
X
X    return bit;
X}
X
X
X/* DETERMINE DECODED SYMBOL FROM INPUT VALUE. */
X
Xstatic int find_symbol(cum_freq)
X    freq_value cum_freq[];	/* Cumulative symbol frequencies            */
X{
X    freq_value cum;		/* Cumulative frequency calculated          */
X    int symbol;			/* Symbol decoded                           */
X
X    F = range / cum_freq[0];			/* Compute common factor.   */
X
X    cum = (value-low) / F;			/* Compute target cum freq. */
X    for (symbol = 1; cum_freq[symbol]>cum; symbol++) ; /* Then find symbol. */
X
X    return symbol;
X}
X
X
X/* NARROW CODING REGION TO THAT ALLOTTED TO SYMBOL. */
X
Xstatic void narrow_region(cum_freq,symbol,have_F)
X    freq_value cum_freq[];	/* Cumulative symbol frequencies            */
X    int symbol;			/* Symbol decoded                           */
X    int have_F;			/* Is F already computed?                   */
X{
X    code_value T;		/* Temporary value                          */
X
X    if (!have_F) F = range / cum_freq[0];	/* Compute common factor.   */
X
X    if (symbol==1) {				/* Narrow range for symbol  */
X        T = F * cum_freq[symbol];		/* at the top.              */
X        low += T;
X        range -= T;
X    }
X
X    else {					/* Narrow range for symbol  */
X        T = F * cum_freq[symbol];               /* not at the top.          */
X        low += T;
X        range = F * cum_freq[symbol-1] - T; 
X    }
X}
X
X
X/* FIND BINARY SPLIT FOR CODING REGION. */
X
Xstatic code_value split_region(freq_b,freq_t)
X    freq_value freq_b;		/* Frequency for symbol in bottom half      */
X    freq_value freq_t;		/* Frequency for symbol in top half         */
X{
X    return freq_b * (range / (freq_b+freq_t)); 
X}
X
X
X/* OUTPUT BITS THAT ARE NOW DETERMINED. */
X
Xstatic void push_out_bits(void)
X{
X    while (range<=Code_quarter) {
X
X        if (low>=Code_half) {			/* Output 1 if in top half.*/
X            bit_plus_follow(1);
X            low -= Code_half;			/* Subtract offset to top.  */
X        }
X
X        else if (low+range<=Code_half) {	/* Output 0 in bottom half. */
X            bit_plus_follow(0);		
X	} 
X
X        else {			 		/* Output an opposite bit   */
X            bits_to_follow += 1;		/* later if in middle half. */
X            low -= Code_quarter;		/* Subtract offset to middle*/
X        } 
X
X        low += low;				/* Scale up code region.    */
X        range += range;
X    }
X}
X
X
X/* DISCARD BITS THE ENCODER WOULD HAVE OUTPUT. */
X
Xstatic void discard_bits(void)
X{
X    while (range<=Code_quarter) {
X
X        if (low>=Code_half) {			/* Expand top half.         */
X            low -= Code_half;			/* Subtract offset to top.  */
X            value -= Code_half;
X        }
X
X        else if (low+range<=Code_half) {	/* Expand bottom half.      */
X            /* nothing */
X	} 
X
X        else {			 		/* Expand middle half.      */
X            low -= Code_quarter;		/* Subtract offset to middle*/
X            value -= Code_quarter;
X        } 
X
X        low += low;				/* Scale up code region.    */
X        range += range;
X
X        value += value;				/* Move in next input bit.  */
X        value += input_bit();
X    }
X}
X
X
X/* FINISH ENCODING THE STREAM. */
X
Xvoid done_encoding(void)
X{   
X    for (;;) {
X
X        if (low+(range>>1)>=Code_half) {	/* Output a 1 if mostly in  */
X            bit_plus_follow(1);			/* top half.                */
X            if (low<Code_half) {
X                range -= Code_half-low;
X                low = 0;
X            }
X            else {
X                low -= Code_half;
X            }
X        }
X
X        else {					/* Output a 0 if mostly in  */
X            bit_plus_follow(0);			/* bottom half.             */
X            if (low+range>Code_half) {
X                range = Code_half-low;
X            }
X        }
X
X        if (range==Code_half) break;		/* Quit when coding region  */
X						/* becomes entire interval. */
X        low += low;
X        range += range;				/* Scale up code region.    */
X    }
X}
END_OF_code_mul.c
if test 8327 -ne `wc -c <code_mul.c`; then
    echo shar: \"code_mul.c\" unpacked with wrong size!
fi
# end of overwriting check
fi
if test -f code_sft.c -a "${1}" != "-c" ; then 
  echo shar: Will not over-write existing file \"code_sft.c\"
else
echo shar: Extracting \"code_sft.c\" \(12376 characters\)
sed "s/^X//" >code_sft.c <<'END_OF_code_sft.c'
X/* CODE_SFT.C - ARITHMETIC ENCODING/DECODING, USING SHIFT/ADD. */
X
X#include <stdio.h>
X
X#define global extern
X
X#include "code.h"
X
X
X/* CURRENT STATE OF THE ENCODING/DECODING. */
X
Xstatic code_value low;		/* Start of current coding region, in [0,1) */
X
Xstatic code_value range;	/* Size of region, normally kept in the     */
X				/* interval [1/4,1/2]                       */
X
Xstatic code_value value;	/* Currently-seen code value, in [0,1)      */
X
Xstatic int bits_to_follow;	/* Number of opposite bits to output after  */
X				/* the next bit                             */
X
Xstatic code_value split_region();
X
X
X/* LOCAL PROCEDURES. */
X
Xstatic int        find_symbol   (freq_value []);
Xstatic void       narrow_region (freq_value [], int);
Xstatic code_value split_region  (freq_value, freq_value);
Xstatic void       push_out_bits (void);
Xstatic void       discard_bits  (void);
X
X
X/* OUTPUT BIT PLUS FOLLOWING OPPOSITE BITS. */
X
X#define bit_plus_follow(bit) \
Xdo { \
X    output_bit(bit);				/* Output the bit.          */\
X    while (bits_to_follow>0) {			/* Output bits_to_follow    */\
X        output_bit(!(bit));			/* opposite bits. Set       */\
X        bits_to_follow -= 1;			/* bits_to_follow to zero.  */\
X    } \
X} while (0) 
X
X
X/* START ENCODING A STREAM OF SYMBOLS. */
X
Xvoid start_encoding(void)
X{   
X    low = Code_half;				/* Use half the code region */
X    range = Code_half;				/* (wastes first bit as 1). */
X    bits_to_follow = 0;				/* No opposite bits follow. */
X}
X
X
X/* START DECODING A STREAM OF SYMBOLS. */
X
Xvoid start_decoding(void)
X{   
X    int i;
X
X    value = input_bit();			/* Check that first bit     */
X    if (value!=1) {				/* is a 1.                  */
X        fprintf(stderr,"Bad input file (1)\n");
X        exit(1);
X    }
X
X    for (i = 1; i<Code_bits; i++) { 		/* Fill the code value with */
X        value += value;				/* input bits.              */
X        value += input_bit();
X    }
X
X    low = Code_half;				/* Use half the code region */
X    range = Code_half;				/* (first bit must be 1).   */
X}
X
X
X/* ENCODE A SYMBOL. */
X
Xvoid encode_symbol(symbol,cum_freq)
X    int symbol;			/* Symbol to encode                         */
X    freq_value cum_freq[];	/* Cumulative symbol frequencies            */
X{
X    narrow_region(cum_freq,symbol);		/* Narrow coding region.    */
X    push_out_bits();				/* Output determined bits.  */
X}
X
X
X/* ENCODE A BIT. */
X
Xvoid encode_bit(bit,freq0,freq1)
X    int bit;			/* Bit to encode (0 or 1)                   */
X    freq_value freq0;           /* Frequency for 0 bit                      */
X    freq_value freq1;           /* Frequency for 1 bit                      */
X{
X    code_value split;
X
X    if (freq1>freq0) {				/* Encode bit when most     */
X        split = split_region(freq0,freq1);	/* probable symbol is a 1.  */
X        if (bit) { low += split; range -= split; }
X        else     { range = split;  }
X    }
X
X    else {					/* Encode bit when most     */
X        split = split_region(freq1,freq0);	/* probable symbol is a 0.  */
X        if (bit) { range = split;  }
X        else     { low += split; range -= split; }
X    }
X
X    push_out_bits();				/* Output determined bits.  */
X}
X
X
X/* DECODE THE NEXT SYMBOL. */
X
Xint decode_symbol(cum_freq)
X    freq_value cum_freq[];	/* Cumulative symbol frequencies            */
X{
X    int symbol;			/* Symbol decoded                           */
X
X    symbol = find_symbol(cum_freq);		/* Find decoded symbol.     */
X    narrow_region(cum_freq,symbol);		/* Narrow coding region.    */
X    discard_bits();				/* Discard output bits.     */
X
X    return symbol;
X}
X
X
X/* DECODE A BIT. */
X
Xint decode_bit(freq0,freq1)
X    freq_value freq0;           /* Frequency for 0 bit                      */
X    freq_value freq1;           /* Frequency for 1 bit                      */
X{
X    code_value split;
X    int bit;
X
X    if (freq1>freq0) {				/* Decode bit when most     */
X        split = split_region(freq0,freq1);	/* probable symbol is a 1.  */
X        bit = value-low >= split;
X        if (bit) { low += split; range -= split; }
X        else     { range = split;  }
X    }
X
X    else {					/* Decode bit when most     */
X        split = split_region(freq1,freq0);	/* probable symbol is a 0.  */
X        bit = value-low < split;
X        if (bit) { range = split;  }
X        else     { low += split; range -= split; }
X    }
X
X    discard_bits();				/* Discard output bits.     */
X
X    return bit;
X}
X
X
X/* DETERMINE DECODED SYMBOL FROM INPUT VALUE. */
X
Xstatic int find_symbol(cum_freq)
X    freq_value cum_freq[];	/* Cumulative symbol frequencies            */
X{
X    int symbol;			/* Symbol decoded                           */
X    freq_value M, P, Q, B;	/* Temporary values                         */
X    div_value F, G;
X    code_value A;	
X    int i;
X
X    A = range; 					/* Compute the value of     */
X    M = cum_freq[0] << (Code_bits-Freq_bits-1); /*   F = range/cum_freq[0]. */
X    F = 0;
X
X    if (A>=M) { A -= M; F += 1; }
X#   if Code_bits-Freq_bits>2
X    A <<= 1; F <<= 1;
X    if (A>=M) { A -= M; F += 1; }
X#   endif
X#   if Code_bits-Freq_bits>3
X    A <<= 1; F <<= 1;
X    if (A>=M) { A -= M; F += 1; }
X#   endif
X#   if Code_bits-Freq_bits>4
X    A <<= 1; F <<= 1;
X    if (A>=M) { A -= M; F += 1; }
X#   endif
X#   if Code_bits-Freq_bits>5
X    A <<= 1; F <<= 1;
X    if (A>=M) { A -= M; F += 1; }
X#   endif
X#   if Code_bits-Freq_bits>6
X    for (i = Code_bits-Freq_bits-6; i>0; i--) {
X        A <<= 1; F <<= 1;
X        if (A>=M) { A -= M; F += 1; }
X    }
X#   endif
X    A <<= 1; F <<= 1;
X    if (A>=M) { F += 1; }
X
X    A = value - low;			  	/* Find the symbol that     */
X    B = Freq_half;				/* fits the input. To do    */
X    G = F << (Freq_bits-1);			/* so compute (value-low)/F */
X    P = 0;                                      /* to as many bits as is    */
X    Q = cum_freq[0];     			/* necessary.               */
X    symbol = 1;
X
X    while (cum_freq[symbol]>P) {
X        if (A>=G) {
X            A -= G; 
X            P = P+B; 
X        }
X        else {
X            Q = P+B;
X            while (cum_freq[symbol]>=Q) symbol += 1;
X        }
X        B >>= 1; G >>= 1;
X    }
X
X    return symbol;
X}
X
X
X/* NARROW CODING REGION TO THAT ALLOTTED TO SYMBOL. */
X
Xstatic void narrow_region(cum_freq,symbol)
X    freq_value cum_freq[];	/* Cumulative symbol frequencies            */
X    int symbol;			/* Symbol decoded                           */
X{
X    code_value A, Ta, Tb;	/* Temporary values                         */
X    freq_value M, Ba, Bb;
X    int i;
X
X    A = range; 
X    M = cum_freq[0] << (Code_bits-Freq_bits-1);
X
X    if (symbol==1) {				/* Narrow range for symbol  */
X                                                /* at the top.              */
X        Ba = cum_freq[symbol];
X        Ta = 0;                                 /* Compute the value of     */
X						/*   Ta = cum_freq[symbol]  */
X        if (A>=M) { A -= M; Ta += Ba; }		/*     * (range/cum_freq[0])*/
X#       if Code_bits-Freq_bits>2
X        A <<= 1; Ta <<= 1;
X        if (A>=M) { A -= M; Ta += Ba; }
X#       endif
X#       if Code_bits-Freq_bits>3
X        A <<= 1; Ta <<= 1;
X        if (A>=M) { A -= M; Ta += Ba; }
X#       endif
X#       if Code_bits-Freq_bits>4
X        A <<= 1; Ta <<= 1;
X        if (A>=M) { A -= M; Ta += Ba; }
X#       endif
X#       if Code_bits-Freq_bits>5
X        A <<= 1; Ta <<= 1;
X        if (A>=M) { A -= M; Ta += Ba; }
X#       endif
X#       if Code_bits-Freq_bits>6
X        for (i = Code_bits-Freq_bits-6; i>0; i--) {
X            A <<= 1; Ta <<= 1;
X            if (A>=M) { A -= M; Ta += Ba; }
X        }
X#       endif
X        A <<= 1; Ta <<= 1;
X        if (A>=M) { Ta += Ba; }
X
X        low += Ta;
X        range -= Ta;
X    }
X
X    else {					/* Narrow range for symbol  */
X                                                /* not at the top.          */
X        Ba = cum_freq[symbol];
X        Bb = cum_freq[symbol-1];	        /* Compute the value of     */
X        Ta = Tb = 0;				/*   Ta = cum_freq[symbol]  */
X						/*     * (range/cum_freq[0])*/
X        if (A>=M) { A -= M; Ta += Ba; Tb += Bb; }/*and of                   */
X#       if Code_bits-Freq_bits>2		/*   Tb = cum_freq[symbol-1]*/
X        A <<= 1; Ta <<= 1; Tb <<= 1;		/*     * (range/cum_freq[0] */
X        if (A>=M) { A -= M; Ta += Ba; Tb += Bb; }
X#       endif
X#       if Code_bits-Freq_bits>3
X        A <<= 1; Ta <<= 1; Tb <<= 1;
X        if (A>=M) { A -= M; Ta += Ba; Tb += Bb; }
X#       endif
X#       if Code_bits-Freq_bits>4
X        A <<= 1; Ta <<= 1; Tb <<= 1;
X        if (A>=M) { A -= M; Ta += Ba; Tb += Bb; }
X#       endif
X#       if Code_bits-Freq_bits>5
X        A <<= 1; Ta <<= 1; Tb <<= 1;
X        if (A>=M) { A -= M; Ta += Ba; Tb += Bb; }
X#       endif
X#       if Code_bits-Freq_bits>6
X        for (i = Code_bits-Freq_bits-6; i>0; i--) {
X            A <<= 1; Ta <<= 1; Tb <<= 1;
X            if (A>=M) { A -= M; Ta += Ba; Tb += Bb; }
X        }
X#       endif
X        A <<= 1; Ta <<= 1; Tb <<= 1;
X        if (A>=M) { Ta += Ba; Tb += Bb; }
X
X        low += Ta;
X        range = Tb - Ta;
X    }
X}
X
X
X/* FIND BINARY SPLIT FOR CODING REGION. */
X
Xstatic code_value split_region(freq_b,freq_t)
X    freq_value freq_b;		/* Frequency for symbol in bottom half      */
X    freq_value freq_t;		/* Frequency for symbol in top half         */
X{
X    code_value A, T;		/* Temporary values                         */
X    freq_value M, B;
X    int i;
X
X    A = range; 
X    M = (freq_b+freq_t) << (Code_bits-Freq_bits-1);
X    B = freq_b;
X    T = 0;           	                        /* Compute the value of     */
X						/* T = freq_b * (range      */
X    if (A>=M) { A -= M; T += B; }		/*       / (freq_b+freq_t)) */
X#   if Code_bits-Freq_bits>2
X    A <<= 1; T <<= 1;
X    if (A>=M) { A -= M; T += B; }
X#   endif
X#   if Code_bits-Freq_bits>3
X    A <<= 1; T <<= 1;
X    if (A>=M) { A -= M; T += B; }
X#   endif
X#   if Code_bits-Freq_bits>4
X    A <<= 1; T <<= 1;
X    if (A>=M) { A -= M; T += B; }
X#   endif
X#   if Code_bits-Freq_bits>5
X    A <<= 1; T <<= 1;
X    if (A>=M) { A -= M; T += B; }
X#   endif
X#   if Code_bits-Freq_bits>6
X    for (i = Code_bits-Freq_bits-6; i>0; i--) {
X        A <<= 1; T <<= 1;
X        if (A>=M) { A -= M; T += B; }
X    }
X#   endif
X    A <<= 1; T <<= 1;
X    if (A>=M) { T += B; }
X
X    return T;
X}
X
X
X/* OUTPUT BITS THAT ARE NOW DETERMINED. */
X
Xstatic void push_out_bits(void)
X{
X    while (range<=Code_quarter) {
X
X        if (low>=Code_half) {			/* Output 1 if in top half.*/
X            bit_plus_follow(1);
X            low -= Code_half;			/* Subtract offset to top.  */
X        }
X
X        else if (low+range<=Code_half) {	/* Output 0 in bottom half. */
X            bit_plus_follow(0);		
X	} 
X
X        else {			 		/* Output an opposite bit   */
X            bits_to_follow += 1;		/* later if in middle half. */
X            low -= Code_quarter;		/* Subtract offset to middle*/
X        } 
X
X        low += low;				/* Scale up code region.    */
X        range += range;
X    }
X}
X
X
X/* DISCARD BITS THE ENCODER WOULD HAVE OUTPUT. */
X
Xstatic void discard_bits(void)
X{
X    while (range<=Code_quarter) {
X
X        if (low>=Code_half) {			/* Expand top half.         */
X            low -= Code_half;			/* Subtract offset to top.  */
X            value -= Code_half;
X        }
X
X        else if (low+range<=Code_half) {	/* Expand bottom half.      */
X            /* nothing */
X	} 
X
X        else {			 		/* Expand middle half.      */
X            low -= Code_quarter;		/* Subtract offset to middle*/
X            value -= Code_quarter;
X        } 
X
X        low += low;				/* Scale up code region.    */
X        range += range;
X
X        value += value;				/* Move in next input bit.  */
X        value += input_bit();
X    }
X}
X
X
X/* FINISH ENCODING THE STREAM. */
X
Xvoid done_encoding(void)
X{   
X    for (;;) {
X
X        if (low+(range>>1)>=Code_half) {	/* Output a 1 if mostly in  */
X            bit_plus_follow(1);			/* top half.                */
X            if (low<Code_half) {
X                range -= Code_half-low;
X                low = 0;
X            }
X            else {
X                low -= Code_half;
X            }
X        }
X
X        else {					/* Output a 0 if mostly in  */
X            bit_plus_follow(0);			/* bottom half.             */
X            if (low+range>Code_half) {
X                range = Code_half-low;
X            }
X        }
X
X        if (range==Code_half) break;		/* Quit when coding region  */
X						/* becomes entire interval. */
X        low += low;
X        range += range;				/* Scale up code region.    */
X    }
X}
END_OF_code_sft.c
if test 12376 -ne `wc -c <code_sft.c`; then
    echo shar: \"code_sft.c\" unpacked with wrong size!
fi
# end of overwriting check
fi
if test -f decode.c -a "${1}" != "-c" ; then 
  echo shar: Will not over-write existing file \"decode.c\"
else
echo shar: Extracting \"decode.c\" \(776 characters\)
sed "s/^X//" >decode.c <<'END_OF_decode.c'
X/* DECODE.C - MAIN PROGRAM FOR DECODING. */
X
X#include <stdio.h>
X
X#define global
X
X#include "code.h"
X#include "model.h"
X
Xvoid main(void)
X{
X    int symbol;			/* Character to decoded as a symbol index    */
X    int ch; 			/* Character to decoded as a character code  */
X
X    start_model();				/* Set up other modules.    */
X    start_inputing_bits();
X    start_decoding();
X
X    for (;;) {					/* Loop through characters. */
X
X        symbol = decode_symbol(cum_freq);	/* Decode next symbol.      */
X        if (symbol==EOF_symbol) break;		/* Exit loop if EOF symbol. */
X        ch = index_to_char[symbol];		/* Translate to a character.*/
X        putc(ch,stdout);			/* Write that character.    */
X        update_model(symbol);			/* Update the model.        */
X    }
X
X    exit(0);
X}
END_OF_decode.c
if test 776 -ne `wc -c <decode.c`; then
    echo shar: \"decode.c\" unpacked with wrong size!
fi
# end of overwriting check
fi
if test -f decpic.c -a "${1}" != "-c" ; then 
  echo shar: Will not over-write existing file \"decpic.c\"
else
echo shar: Extracting \"decpic.c\" \(1809 characters\)
sed "s/^X//" >decpic.c <<'END_OF_decpic.c'
X/* DECPIC.C - MAIN PROGRAM FOR DECODING PICTURES. */
X
X#include <stdio.h>
X
X#define global 
X
X#include "code.h"
X
X#define Height 40			/* Height of images                 */
X#define Width 40                        /* Width of images                  */
X
Xstatic int image[Height][Width];	/* The image to be encoded          */
X
Xstatic int freq0[2][2];			/* Frequencies of '0' in contexts   */
Xstatic int freq1[2][2];			/* Frequencies of '1' in contexts   */
X
Xstatic int inc[2][2];			/* Current increment                */
X
Xvoid main(void)
X{   
X    int i, j, a, l;
X
X    start_inputing_bits();
X    start_decoding();
X
X    /* Initialize model. */
X
X    for (a = 0; a<2; a++) {
X        for (l = 0; l<2; l++) {
X            inc[a][l] = Freq_half;
X            freq0[a][l] = inc[a][l];		/* Set frequencies of 0's   */
X            freq1[a][l] = inc[a][l];		/* and 1's to be equal.     */
X        }
X    }
X
X    /* Decode and write image. */
X
X    for (i = 0; i<Height; i++) {
X        for (j = 0; j<Width; j++) {
X            a = i==0 ? 0 : image[i-1][j];	/* Find current context.    */
X            l = j==0 ? 0 : image[i][j-1];
X            image[i][j] = 			/* Decode pixel.            */
X              decode_bit(freq0[a][l],freq1[a][l]);
X            printf("%c%c",image[i][j] ? '#' : '.', 
X                          j==Width-1 ? '\n' : ' ');
X            if (image[i][j]) {			/* Update frequencies for   */
X                freq1[a][l] += inc[a][l];       /* this context.            */
X            }
X            else {
X                freq0[a][l] += inc[a][l];
X            }
X            if (freq0[a][l]+freq1[a][l]>Freq_full) { 
X                freq0[a][l] = (freq0[a][l]+1) >> 1; 
X                freq1[a][l] = (freq1[a][l]+1) >> 1;
X                if (inc[a][l]>1) inc[a][l] >>= 1;
X            }
X        }
X    }
X
X    exit(0);
X}
END_OF_decpic.c
if test 1809 -ne `wc -c <decpic.c`; then
    echo shar: \"decpic.c\" unpacked with wrong size!
fi
# end of overwriting check
fi
if test -f encode.c -a "${1}" != "-c" ; then 
  echo shar: Will not over-write existing file \"encode.c\"
else
echo shar: Extracting \"encode.c\" \(924 characters\)
sed "s/^X//" >encode.c <<'END_OF_encode.c'
X/* ENCODE.C - MAIN PROGRAM FOR ENCODING. */
X
X#include <stdio.h>
X
X#define global
X
X#include "code.h"
X#include "model.h"
X
Xvoid main(void)
X{   
X    int ch; 			/* Character to encode as a character code  */
X    int symbol;			/* Character to encode as a symbol index    */
X
X    start_model();				/* Set up other modules.    */
X    start_outputing_bits();
X    start_encoding();
X
X    for (;;) {					/* Loop through characters. */
X
X        ch = getc(stdin);			/* Read the next character. */
X        if (ch==EOF) break;			/* Exit loop on end-of-file.*/
X        symbol = char_to_index[ch];		/* Translate to an index.   */
X        encode_symbol(symbol,cum_freq);		/* Encode that symbol.      */
X        update_model(symbol);			/* Update the model.        */
X    }
X
X    encode_symbol(EOF_symbol,cum_freq);		/* Encode the EOF symbol.   */
X
X    done_encoding();				/* Send the last few bits.  */
X    done_outputing_bits();
X
X    exit(0);
X}
END_OF_encode.c
if test 924 -ne `wc -c <encode.c`; then
    echo shar: \"encode.c\" unpacked with wrong size!
fi
# end of overwriting check
fi
if test -f encpic.c -a "${1}" != "-c" ; then 
  echo shar: Will not over-write existing file \"encpic.c\"
else
echo shar: Extracting \"encpic.c\" \(2309 characters\)
sed "s/^X//" >encpic.c <<'END_OF_encpic.c'
X/* ENCPIC.C - MAIN PROGRAM FOR ENCODING PICTURES. */
X
X#include <stdio.h>
X
X#define global 
X
X#include "code.h"
X
X#define Height 40			/* Height of images                 */
X#define Width 40                        /* Width of images                  */
X
Xstatic int image[Height][Width];	/* The image to be encoded          */
X
Xstatic int freq0[2][2];			/* Frequencies of '0' in contexts   */
Xstatic int freq1[2][2];			/* Frequencies of '1' in contexts   */
X
Xstatic int inc[2][2];			/* Current increment                */
X
Xvoid main(void)
X{   
X    int i, j, a, l, ch;
X
X    start_outputing_bits();
X    start_encoding();
X
X    /* Read image. */
X
X    for (i = 0; i<Height; i++) {
X        for (j = 0; j<Width; j++) {
X            do {
X                ch = getc(stdin);		/* Read the next character, */
X            } while (ch=='\n' || ch==' ');      /* ignoring whitespace.     */
X            if (ch!='.' && ch!='#') {           /* Check for bad character. */
X                fprintf(stderr,"Bad image file\n");
X                exit(-1);
X            }
X            image[i][j] = ch=='#';		/* Convert char to pixel.   */
X        }
X    }
X
X    /* Initialize model. */
X
X    for (a = 0; a<2; a++) {
X        for (l = 0; l<2; l++) {
X            inc[a][l] = Freq_half;
X            freq0[a][l] = inc[a][l];		/* Set frequencies of 0's   */
X            freq1[a][l] = inc[a][l];		/* and 1's to be equal.     */
X        }
X    }
X
X    /* Encode image. */
X
X    for (i = 0; i<Height; i++) {
X        for (j = 0; j<Width; j++) {
X            a = i==0 ? 0 : image[i-1][j];	/* Find current context.    */
X            l = j==0 ? 0 : image[i][j-1];
X            encode_bit(image[i][j],             /* Encode pixel.            */
X                       freq0[a][l],freq1[a][l]);
X            if (image[i][j]) {			/* Update frequencies for   */
X                freq1[a][l] += inc[a][l];       /* this context.            */
X            }
X            else {
X                freq0[a][l] += inc[a][l];
X            }
X            if (freq0[a][l]+freq1[a][l]>Freq_full) { 
X                freq0[a][l] = (freq0[a][l]+1) >> 1; 
X                freq1[a][l] = (freq1[a][l]+1) >> 1;
X                if (inc[a][l]>1) inc[a][l] >>= 1;
X            }
X        }
X    }
X
X    done_encoding();				/* Send the last few bits.  */
X    done_outputing_bits();
X
X    exit(0);
X}
END_OF_encpic.c
if test 2309 -ne `wc -c <encpic.c`; then
    echo shar: \"encpic.c\" unpacked with wrong size!
fi
# end of overwriting check
fi
if test -f model.c -a "${1}" != "-c" ; then 
  echo shar: Will not over-write existing file \"model.c\"
else
echo shar: Extracting \"model.c\" \(2354 characters\)
sed "s/^X//" >model.c <<'END_OF_model.c'
X/* MODEL.C - THE ADAPTIVE SOURCE MODEL */
X
X#include <stdio.h>
X
X#define global extern
X
X#include "code.h"
X#include "model.h"
X
Xstatic freq_value freq[No_of_symbols+1]; /* Symbol frequencies                */
Xstatic freq_value inc;		         /* Value to increment frequencies by */
X
X
X/* INITIALIZE THE MODEL. */
X
Xvoid start_model(void)
X{   
X    int i;
X
X    for (i = 0; i<No_of_chars; i++) {		/* Set up tables that       */
X        char_to_index[i] = i+1;			/* translate between symbol */
X        index_to_char[i+1] = i;			/* indexes and characters.  */
X    }
X
X    inc = 1;
X    while (inc*No_of_symbols<=Freq_half) {	/* Find increment that puts */
X        inc *= 2;				/* total in required range. */
X    }
X
X    cum_freq[No_of_symbols] = 0;		/* Set up initial frequency */
X    for (i = No_of_symbols; i>0; i--) {		/* counts to be equal for   */
X        freq[i] = inc;				/* all symbols.             */
X        cum_freq[i-1] = cum_freq[i] + freq[i];	
X    }
X
X    freq[0] = 0;				/* Freq[0] must not be the  */
X						/* same as freq[1].         */
X}
X
X
X/* UPDATE THE MODEL TO ACCOUNT FOR A NEW SYMBOL. */
X
Xvoid update_model(symbol)
X    int symbol;			/* Index of new symbol                      */
X{   
X    int ch_i, ch_symbol;	/* Temporaries for exchanging symbols       */
X    int i;
X
X    for (i = symbol; freq[i]==freq[i-1]; i--) ;	/* Find symbol's new index. */
X
X    if (i<symbol) {
X        ch_i = index_to_char[i];		/* Update the translation   */
X        ch_symbol = index_to_char[symbol];	/* tables if the symbol has */
X        index_to_char[i] = ch_symbol;           /* moved.                   */
X        index_to_char[symbol] = ch_i;
X        char_to_index[ch_i] = symbol;
X        char_to_index[ch_symbol] = i;
X    }
X
X    freq[i] += inc;				/* Increment the frequency  */
X    while (i>0) {				/* count for the symbol and */
X        i -= 1;					/* update the cumulative    */
X        cum_freq[i] += inc;			/* frequencies.             */
X    }
X
X    if (cum_freq[0]>Freq_full) {		/* See if frequency counts  */
X        cum_freq[No_of_symbols] = 0;		/* are past their maximum.  */
X        for (i = No_of_symbols; i>0; i--) {	/* If so, halve all counts  */
X            freq[i] = (freq[i]+1) >> 1;		/* (keeping them non-zero). */
X            cum_freq[i-1] = cum_freq[i] + freq[i]; 
X        }
X        if (inc>1) inc >>= 1;			/* Halve increment if can.  */
X    }
X}
END_OF_model.c
if test 2354 -ne `wc -c <model.c`; then
    echo shar: \"model.c\" unpacked with wrong size!
fi
# end of overwriting check
fi
if test -f redundancy.c -a "${1}" != "-c" ; then 
  echo shar: Will not over-write existing file \"redundancy.c\"
else
echo shar: Extracting \"redundancy.c\" \(1597 characters\)
sed "s/^X//" >redundancy.c <<'END_OF_redundancy.c'
X/* REDUNDANCY.C - PROGRAM TO CALCULATE MAXIMUM CODING INEFFICIENCY. */
X
X#include <stdio.h>
X#include <math.h>
X
X
X/* This program calculates a bound on the expected number of extra bits
X   produced as a result of doing arithmetic coding using divides of a given 
X   precision, expressed as a percentage of the optimal coding size. The
X   expectation is calculated on the assumption that the model's symbol
X   probabilities are correct.
X
X   The bound assumes that the coding range and total frequencies are such
X   as to cause maximum truncation in the division, with the minimum true
X   quotient. The bound varies as a function of the probability, p, of
X   the most probable symbol. All but at most one of the remaining symbols
X   are assumed to have probabilities equal to p (this gives minimum optimal
X   coding size, and hence maximum relative inefficiency). */
X
Xmain()
X{
X    double p;
X
X    printf(
X     "\nPrecision:    2      3      4      5      6      7      8      9\n\n");
X   
X    for (p = 0.001; p<0.0095; p+=0.001) do_p(p);
X    for (p = 0.010; p<0.9905; p+=0.010) do_p(p);
X    for (p = 0.991; p<0.9995; p+=0.001) do_p(p);
X  
X    exit(0);
X}
X
Xdo_p(p)
X  double p;
X{
X    int precision, n;
X    double e, opt, excess;
X
X    printf("  p=%5.3f ",p);
X    for (precision = 2; precision<10; precision++) {
X        e = pow(2.0,-(double)precision)/(0.25+pow(2.0,-(double)precision));
X        n = (int)(1/p);
X        opt = - n*p*log(p);
X        if (n*p!=1) opt -= (1-n*p)*log(1-n*p);
X        excess = - (1-p)*log(1-e) - p*log(1-e+e/p);
X        printf("  %5.2f",100*excess/opt);
X    }
X    printf("\n");
X}
END_OF_redundancy.c
if test 1597 -ne `wc -c <redundancy.c`; then
    echo shar: \"redundancy.c\" unpacked with wrong size!
fi
# end of overwriting check
fi
if test -f tstpic -a "${1}" != "-c" ; then 
  echo shar: Will not over-write existing file \"tstpic\"
else
echo shar: Extracting \"tstpic\" \(3200 characters\)
sed "s/^X//" >tstpic <<'END_OF_tstpic'
X. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
X. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
X. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
X. . . # # . . . . . . . . . . . . . . . . . . . . . . # . . . . . . . . . . . .
X. . . # # . . . . . . . . . . . . . . . . . . . # . # . . . . . . . . . . . . .
X. . . # # . . . . . . . . . . . . . . . . . # # # # . . . . . . . . . . . . . .
X. . . . . . . . . . # # . . . . . . . . . . . . . # . . . . . . . . . . . . . .
X. . . . . . . . # # # # # . . . . . . . . . . . . . . . . . . . . . . . . . . .
X. . . . . . . . . . # . # . . . . . . . . . . . . . . . . . . . . . . . . . . .
X. . . . . . . . . . . # . . . . . . . . . . . . . . . . . . . . . . . . . . . .
X. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
X. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
X. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
X. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
X. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
X# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
X. . . . . . . . . . . . # # # # . . . . . . . . . . . . . . . . . . . . . . . .
X. . . . . . . . . . . . . . # . . . . . . . . . . . . . . . . . . . . . . . . .
X. . . . . . . . . . . . . . . . . . . . . . . . . . . . # # . . . . . . . . . .
X. . . . . . . . . . . . . . . . . . . . . . . . . . # # # # # # # . . . . . . .
X. . . . . . . . . . . . . . . . . . . . . . . . . # # # # # # # # . . . . . . .
X. . . . . . . . . . . . . . . . . . . . . . . . . . # # # # . # # . . . . . . .
X. . . . . . . . . . . . . . . . . . . . . . . . . . # # # # # # # . . . . . . .
X. . . . . . . . . . . . . . . . . . . . . . . . . . . . . # # # . . . . . . . .
X. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
X. . . . . . . . . # # . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
X. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
X. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
X. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
X. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
X. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
X. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
X. . . . . . . . . . . . . . . . . . . . . . . . . . . . # . . . . . . . . . . .
X. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
X. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . #
X. . . . . # # # . # # # # . . . . . . . . . . . . . . . . . . . . . . . . . # #
X. . . . . . . . # . . . . # # . . . . . . . . . . . . . . . . . . . . . . . . #
X. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . # #
X. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . # # #
X. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . # # #
END_OF_tstpic
if test 3200 -ne `wc -c <tstpic`; then
    echo shar: \"tstpic\" unpacked with wrong size!
fi
# end of overwriting check
fi
echo shar: End of shell archive.
exit 0