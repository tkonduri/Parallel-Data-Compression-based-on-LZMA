/******************************************************************************
File: 		main.c

Authors: 	John Carpinelli   (johnfc@ecr.mu.oz.au)
	 	Wayne Salamonsen  (wbs@mundil.cs.mu.oz.au)
		Lang Stuiver      (langs@cs.mu.oz.au)

Purpose:	Arithmetic coding data compression driver module


Copyright 1995 John Carpinelli and Wayne Salamonsen, All Rights Reserved.
Copyright 1996 Lang Stuiver.  All rights reserved.

These programs are supplied free of charge for research purposes only,
and may not sold or incorporated into any commercial product.  There is
ABSOLUTELY NO WARRANTY of any sort, nor any undertaking that they are
fit for ANY PURPOSE WHATSOEVER.  Use them at your own risk.  If you do
happen to find a bug, or have modifications to suggest, please report
the same to Alistair Moffat, alistair@cs.mu.oz.au.  The copyright
notice above and this statement of conditions must remain an integral
part of each and every copy made of these files.

******************************************************************************
Main.c is the main program.  It processes the command line, reads and
writes the initial headers of the compressed files, sets any parameters
and runs the appropriate model.
It also keeps track of memory usage, to keep it within any specified limits
and report on its usage.
******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#ifdef SYSV
#  include <sys/times.h>
#  include <limits.h>
#  include <unistd.h>
#endif

#include "bitio.h"
#include "arith.h"
#include "stats.h"

#include "main.h"



/* local function prototypes */
static void print_results(int operation, int method);
static void usage(char *str1);
static void encode_only(char *str1, int selected, char **argv);
static int determine_model(char *argv);
static int decode_magic(unsigned char *str1);
static void describe_setup(int method);

#ifdef RCSID
static char
   rcsid[] = "$Id: main.c,v 1.1 1996/08/07 01:34:11 langs Exp $";
#endif

/* 
 * The model list is of type ModelType.  Each model has associated with
 * it a name, its magic number, an encode and decode function, whether
 * it dynamically uses memory, and if so, whether it will purge it.
 * There is also a results function which can be specified to describe
 * statistics specific to that model after it has run.  It is passed a
 * variable indicating whether it was run to ENCODE, or to DECODE.  The
 * magic numbers all start with character 255 (\377) to ensure the
 * compressed file will not be mistaken for a text file.
 */

typedef struct {
   char *name;			/* Name, as selected with -t option	*/
   char *magic;			/* Magic number				*/
   void (*encode)(void);	/* Encode function			*/
   void (*decode)(void);	/* Decode function			*/
   int needs_mem;		/* Does it need the -m memory option,	*/
				/* and use memory dynamically ?		*/
   int purge_mem;		/* If so, does it purge as well?	*/
   void (*results)(int);	/* Model specific results function	*/
   char *desc;			/* Description of model			*/
  } ModelType;

ModelType model[] =
  {
/*	  name		magic		encoder	     decoder, needs_mem, purge
 *	       results,  description
 */
	{ "char",	"\377" "23c",	encode_char,	decode_char,	0, 0,
		NULL,
		"Character based model"	},
	{ "uint",	"\377" "23n",	encode_uints,	decode_uints,	0, 0,
		NULL,
		"Binary uint based model (with fake-escape coding)"	},
	{ "word",	"\377" "23w",	encode_word,	decode_word,	1, 1,
		print_results_word,
		"Word based model"	},
     	{ "bits",	"\377" "23b",	encode_bits,	decode_bits,	1, 0,
		print_results_bits,
		"Bit based model"	},
	{ NULL,	  	NULL,	NULL,		NULL, 	0, 0, NULL, NULL }
  };


/* global variables */
       int verbose = 0;		/* flag set if stats are to be printed */

/* Used by the memory counting section */
static int mbytes = DEFAULT_MEM;/* stores no. megabytes allowable for mem */
static int total_memory = 0;	/* total memory used by all models */
static int peak_memory = 0;	/* Peak usage of total_memory before purge */
static int purge_counter=0;	/* counts number of memory purges */

/* 
 * parse command line arguments. Decide whether to decode or encode
 * and optional memory size. Also replaces stdin with input file
 */
int main(int argc, char *argv[])
{	
    int i;				/* loop counter */
    unsigned char
	tempstore[MAGICNO_LENGTH];	/* stores magic no */
    char version_str[VERSION_LEN+1];	/* Stores version str read in */
    int	selected = -1;			/* whether decoding or encoding */
    int method;				/* The coding method used */
    int mem_specified = 0;		/* Is '-m' on the command line */
    int default_method;
    int filename_supplied = 0;
    char *modelstr = argv[0];	/* Points to name of string
				 * to indicate default compression
				 * type.  Normally program name
				 *	(word, bits, char)
				 */
    
    for (i = 1; i < argc; ) 
    {
	if (argv[i][0] == '-') 
	{
	    switch(argv[i][1]) 
	    {
	      case 'e':		/* do encode */
		selected = ENCODE;
		i++;
		break;
	      case 'd':		/* do decode */
		selected = DECODE;
		i++;
		break;
	      case 'm':		/* set memory size */
		encode_only("Memory size", selected, argv);
		mem_specified = 1;
		i++;
		if (i>=argc) usage(argv[0]);
		mbytes = atoi(argv[i++]);
		break;
	      case 'v':		/* set verbose flag to print stats */
		verbose = 1;
		i++;
		break;
	      case 'f':		/* set number of F bits */
		encode_only("Frequency bits", selected, argv);
		i++;
		if (i>=argc) usage(argv[0]);
#ifdef VARY_NBITS
		F_bits = atoi(argv[i++]);
#else
		if (F_bits != atoi(argv[i++]))
			{ fprintf(stderr,"Invalid F_bits (-f option): "
					 "F_bits fixed at %i\n", F_bits);
			  exit(1);
			}
#endif
		break;
	      case 'c':
		i++;
		if (i>=argc) usage(argv[0]);
		bits_context = atoi(argv[i++]);
		break;
	      case 'b':
		encode_only("Code bits", selected, argv);
		i++;
		if (i>=argc) usage(argv[0]);
#ifdef VARY_NBITS
		B_bits = atoi(argv[i++]);
#else
		if (B_bits != atoi(argv[i++]))
			{ fprintf(stderr, "Invalid B_bits (-b option): "
					  "B_bits fixed at %i\n", B_bits);
			  exit(1);
			}
#endif
		break;
	      case 't':		/* Type of compression, 'progname' */
		i++;
		if (i>argc) usage(argv[0]);
		modelstr=argv[i];
		i++;
		break;
	      default:		/* incorrect args */
		usage(argv[0]);
	    }
	}
	else 
	{ 
	  if (filename_supplied)
	   {
		fprintf(stderr,"Only one filename can be specified\n");
		exit(1);
   	   }
	  if (freopen(argv[i++], "rb", stdin) == (FILE *)NULL) 
	   {
	    fprintf(stderr, "%s: cannot read file %s\n",
		    argv[0], argv[--i]);
	    exit(1);
	   }
	   filename_supplied = 1;
	}
    }
    
    /* check if memory limit is within allowable range */ 
    if (mbytes < MIN_MBYTES || mbytes > MAX_MBYTES)
    {
	fprintf(stderr, "memory limit must be between %d and %d\n", 
		MIN_MBYTES, MAX_MBYTES);
	exit(1);
    }

    /* check if B_bits is within allowable range */
    if (B_bits < 3 || B_bits > MAX_B_BITS)
    {
	fprintf(stderr, "number of B_bits must be between %d and %d\n",
		3, MAX_B_BITS);
	exit(1);
    }

    /* check if F_bits is within allowable range */
    if (F_bits < 1 || F_bits > MAX_F_BITS)
    {
	fprintf(stderr, "number of f bits must be between %d and %d\n",
		1, MAX_F_BITS);
	exit(1);
    }

    if (bits_context < MIN_CONTEXT_BITS || bits_context > MAX_CONTEXT_BITS)
    {
	fprintf(stderr, "Bits of context must be between %d and %d\n",
		MIN_CONTEXT_BITS, MAX_CONTEXT_BITS);
	exit(1);
    }

    if (selected == -1)
	usage(argv[0]);

    if (selected == ENCODE)					/* do ENCODE */
    {
	if (B_bits < F_bits + 2)
	{ fprintf(stderr, "Code bits must be at least freq bits + 2.\n");
	  fprintf(stderr, "(Code bits = %i, freq bits = %i)\n",
			B_bits, F_bits);
	  exit(1);
	}

	method = determine_model(modelstr);	/* Encoding method to use */
	if (method == -1) method = determine_model(DEFAULT_MODEL);
	if (method == -1) method = 0;

	/* write magic number to output file */
	BITIO_FWRITE(model[method].magic, 1, MAGICNO_LENGTH);

	/* Write the version string (excluding terminating \0) */
	BITIO_FWRITE(VERSION, 1, VERSION_LEN);


	/* store F_bits and B_bits being used in output */
	OUTPUT_BYTE(F_bits);
	OUTPUT_BYTE(B_bits);

	if (model[method].needs_mem)
	{
	  /* store memory limit being used in output */
	  OUTPUT_BYTE(mbytes);
	}
	if (!model[method].needs_mem && mem_specified)
	{
	   fprintf(stderr,"This compression method doesn't"
		" use the -m option\n");
	   usage(argv[0]);
	}

	if (verbose) describe_setup(method);
	model[method].encode();  			/* call ENCODER */
    }

    else
							/* do DECODE */
    {
	BITIO_FREAD(tempstore, 1, MAGICNO_LENGTH);
	method = decode_magic(tempstore);

	BITIO_FREAD(version_str, 1, VERSION_LEN);
	version_str[VERSION_LEN] = '\0';
	if (strcmp(VERSION,version_str) != 0)
		{ 
		  fprintf(stderr,"Wrong version!\n");
		  fprintf(stderr,"This is version %s\n",VERSION);
		  fprintf(stderr,"Compression was with version %s\n",
				  version_str);
		  exit(1);
		}
        
	default_method = determine_model(modelstr);
	if (default_method == -1) default_method = method;

	/* If decompressing with a different model than user might expect
	 * tell them  Eg: running "char -e foo | bits -d" will work,
	 * but only because they are both the same program, containing all
	 * the models in them, and model "char" is selected instead of "bits"
	 * to decompress
	 */
	if (method != default_method)
		fprintf(stderr,"Using method: %s\n",model[method].desc);

	/* get number of F_bits and B_bits to be used */

#ifdef VARY_NBITS
	F_bits = INPUT_BYTE();
	B_bits = INPUT_BYTE();
#else
	{ int f1, b1;
		f1 = INPUT_BYTE();
		b1 = INPUT_BYTE();
	if (F_bits != f1 || B_bits != b1)
	 {
	  fprintf(stderr, "Differing value for frequency / code bits:\n"
	    "This program was compiled with fixed  F_bits=%i, B_bits=%i\n"
	    "but the data file was compressed with F_bits=%i, B_bits=%i.\n",
		F_bits, B_bits, f1, b1);
	  exit(1);
	 }
	}
#endif

	if (B_bits - F_bits < 2)
	     { fprintf(stderr,"Corrupt input file; B_bits - F_bits < 2\n");
	       exit(1);
	     }

	if (model[method].needs_mem)
	{
	  /* read memory limit to be used and store in mbytes */
	  mbytes = INPUT_BYTE();
	}

	if (verbose) describe_setup(method);
	model[method].decode();			/* Call DECODER */
    }
    
    /* statistics section if using verbose flag */
    if (verbose)
	{
	  print_results(selected, method);
	}
    return 0;			/* exited cleanly */
}

/*
 * Ensure encoding options are only specifed after a '-e' on the command
 * line, more to point, ensure they are only used for encoding
 */
static void encode_only(char *str1, int selected, char **argv)
{
  if (selected!=ENCODE)
  {
   fprintf(stderr,
	"%s is defined by the encoder,\n"
	"and can not be specified by the decoder.\n"
   	"('-e' is required on command line before any encode options)\n",
	 str1);
   usage(argv[0]);
  }		
}


/* 
 * If argv[0] matches one of the methods in the list, return it, else
 * return -1.  Used so that the program name name can determine the
 * default compression model.
 */
static int determine_model(char *arg0)
{
    char *progname;
    int i;
    
    progname = arg0;				/* Ignore pathname */
    if (strrchr(progname, '/') != NULL)
	progname = strrchr(progname, '/') + 1;
	
    i = 0;
    while (model[i].name != NULL &&
	   strcmp(progname, model[i].name) != 0)
			i++;

	/* If no names in list match progname, see if they are
	 * contained within progname, eg: 'bits1' would match 'bits' */

    if (model[i].name == NULL)
	{
	for (i=0; model[i].name != NULL; i++)
	    if (strstr(progname, model[i].name) != NULL) break;
	}

	/* Return -1 if no matching model */
    if (model[i].name == NULL)
	return -1;

    return i;
}

/*
 * decode_magic(str1)
 * Search through the magic numbers for the coding methods, trying to
 * match with str1.  Return coding method if found, else exit program.
 */

static int decode_magic(unsigned char *str1)
{
    int i = 0;
    
    while (model[i].name != NULL &&
	   memcmp(str1, model[i].magic, MAGICNO_LENGTH) != 0)
			i++;

    if (model[i].name == NULL)
	{
	    fprintf(stderr, "Bad Magic Number\n");
	    exit(1);
	}
    return i;

}

/*
 * usage(argv[0])
 */
static void usage(char *str1)
{
  int default_method;
  char model_list[1024];
  char freq_string[127];
  char code_string[127];
  int i;

  /* Build up a list of models.  Ie: "char, bits, word" */

  model_list[0]='\0';
  strcpy(model_list, model[0].name);
  for (i = 1; model[i].name!=NULL; i++)
	{
	  strcat(model_list, ", ");
	  strcat(model_list, model[i].name);
	}


#if defined(VARY_NBITS)
  sprintf(freq_string,"%i..%i, default = %i",1, MAX_F_BITS, F_bits);
  sprintf(code_string,"%i..%i, default = %i",3, MAX_B_BITS, B_bits);
#else
  sprintf(freq_string,"%i, fixed at compile time", F_bits);
  sprintf(code_string,"%i, fixed at compile time", B_bits);
#endif
 
  
  /* Determine which is the default method, so can tell user */
  default_method = determine_model(str1);
  if (default_method == -1) default_method = determine_model(DEFAULT_MODEL);
  if (default_method == -1) default_method = 0;

  fprintf(stderr,
	  "\nUsage: "
	  "%s [-e [-t s] [-f n] [-b n] [-m n] [-c n] | -d] [-v] [file]\n\n",
	   str1);
  fprintf(stderr,
	"-e: Encode\n"
	"    -t s    Encoding method               (%s, default = %s)\n"
	"    -b n    Code bits to use              (%s)\n"
	"    -f n    Frequency bits to use         (%s)\n"
	"    -m n    Memory size to use (Mbytes)   (%i..%i, default = %i)"
			"    bits & word\n"
	"    -c n    Bits of context to use        (%i..%i, default = %i)"
			"    bits only\n"
	"-d: Decode\n"
	"-v: Verbose Give timing, compression, and memory "
	     "usage information.\n\n",
	model_list, model[default_method].name,
	code_string,
	freq_string,
	MIN_MBYTES, MAX_MBYTES, DEFAULT_MEM,
	MIN_CONTEXT_BITS, MAX_CONTEXT_BITS, DEFAULT_BITS_CONTEXT
	);

  if (verbose) {
	fprintf(stderr,"Version: %s\n\n",VERSION);
	describe_setup(-1);
	}
  exit(1);
}


/* 
 * Describe setup for 'method', or in general if method < 0 
 */
static void describe_setup(int method)
{
  if (method>=0)
	fprintf(stderr,"Model                  : %s\n",model[method].desc);
  fprintf(stderr,"Stats                  : %s\n",stats_desc);
  fprintf(stderr,"Coder                  : %s\n",coder_desc);

#if defined(VARY_NBITS)
  if (method >= 0)
  {
    fprintf(stderr,"Code bits              : %10i\n", B_bits);
    fprintf(stderr,"Frequency bits         : %10i\n", F_bits);
  }
#else
    fprintf(stderr,"Code bits              : %10i (Fixed)\n", B_bits);
    fprintf(stderr,"Frequency bits         : %10i (Fixed)\n", F_bits);
#endif
  if (method >= 0 && model[method].needs_mem)
    fprintf(stderr,"Memory limit           : %10i Mb\n",mbytes);
}



/*
 *
 * print the results of compressing/decompressing a file
 *
 */
static void print_results(int operation, int method)
{ int bytes_compressed, bytes_uncompressed;


    if (operation == ENCODE)
	{
	  bytes_compressed = bitio_bytes_out();
	  bytes_uncompressed = bitio_bytes_in();
	}
    else
	{
	  bytes_compressed = bitio_bytes_in();
	  bytes_uncompressed = bitio_bytes_out();
	}
	
    fprintf(stderr,"Uncompressed           : %10u bytes\n", bytes_uncompressed);
    fprintf(stderr,"Compressed             : %10u bytes\n", bytes_compressed);

      if (bytes_uncompressed > 0)
	fprintf(stderr, "Compression rate       : %10.3f bpc (%0.2f%%) \n", 
		8.0 * bytes_compressed / bytes_uncompressed, 
		(float)bytes_compressed/bytes_uncompressed*100);


   /* only provide timing details if "times()" function is available */
			/* Give kb/s in terms of uncompressed data */
#ifdef 	SYSV
   {
    struct tms cpu_usage;
    float cpu_used, comp_rate;

    times(&cpu_usage);    	/* determine the cpu time used */
    cpu_used = ((float) cpu_usage.tms_utime) / sysconf(_SC_CLK_TCK);

    if (cpu_used == 0)
	comp_rate = 0;
    else
    {
	    comp_rate = ((float) bytes_uncompressed) / (1024 * cpu_used);
    }

    fprintf(stderr, "Compression time       : %10.2f seconds (%0.2f Kb/s)\n",
	    cpu_used, comp_rate);
   }
#endif

    if (model[method].needs_mem)
      {
	if (model[method].purge_mem)
	    {
		fprintf(stderr, "Memory purges          : %10d time%s\n",
			purge_counter, (purge_counter == 1 ? "" : "s") );
	    }
	/* Count peak memory usage, even if later purged */
	if (peak_memory > total_memory)
		total_memory = peak_memory;
	fprintf(stderr, "Peak memory used       : %10.1f Kbytes\n",
			total_memory*1.0/1024);
      }

	/* Call per model method results if they exist */
    if (model[method].results != NULL)
	model[method].results(operation);
}


/************************************************************************/
/*
 * Memory Routines.  Used by word.c and bits.c.  They malloc, realloc, purge
 *	and reserve memory as required, while staying within the specified
 *	memory limit (artificially returns NULL if would exceed memory
 *	limit).
 */

/* 
 * 
 * call the standard C function realloc after checking that the memory
 * limit isn't exceeded. If limit is exceeded return NULL
 * 
 */
void *do_realloc(void *ptr, size_t size)
{
    if (((total_memory+size) / MEGABYTE) >= mbytes)
	return NULL;

    total_memory += size;
    return (realloc(ptr, size));
}


/*
 *
 * call the standard C function malloc after checking against the memory
 * limit. If the limit is exceeded return NULL
 *
 */
void *do_malloc(size_t size)
{
    if (((total_memory+size) / MEGABYTE) >= mbytes)
	return NULL;
    total_memory += size;
    return (malloc(size));
}


/*
 *
 * adds specified memory to current memory count
 * returns total bytes now allocated if successful, NOMEMLEFT if memory limit
 * is reached
 */
int get_memory(size_t size)
{
    if (((total_memory+size) / MEGABYTE) >= mbytes)
	return NOMEMLEFT;
    total_memory += size;
   
    return total_memory;
}

/*
 * Clear memory counter (assume free() already done by calling routine)
 *
 */

void purge_memory(void)
{
  if (total_memory > peak_memory)
	peak_memory = total_memory;
  total_memory = 0;
  purge_counter++;
}
