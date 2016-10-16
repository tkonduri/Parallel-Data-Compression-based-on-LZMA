#define poly 0x1021
    /* Calculating XMODEM CRC-16 in 'C'
   ================================ */
/* On entry, addr=>start of data
             num = length of data
             crc = incoming CRC     */
int crc16(char *addr, int num, int crc)
{
    int i;

    for (; num>0; num--)              /* Step through bytes in memory */
    {
        crc = crc ^ (*addr++ << 8);     /* Fetch byte from memory, XOR into CRC top byte*/
        for (i=0; i<8; i++)             /* Prepare to rotate 8 bits */
        {
            if (crc & 0x10000)            /* b15 is set... */
                crc = (crc << 1) ^ poly;    /* rotate and XOR with XMODEM polynomic */
            else                          /* b15 is clear... */
                crc <<= 1;                  /* just rotate */
        }                             /* Loop for 8 bits */
        crc &= 0xFFFF;                  /* Ensure CRC remains 16-bit value */
    }                               /* Loop until num=0 */
    return(crc);                    /* Return updated CRC */
}

