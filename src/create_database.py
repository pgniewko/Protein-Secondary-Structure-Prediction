#! /usr/bin/env python
# BioE 134, Fall 2017
# Author: Pawel Gniewek (pawel.gniewek@berkeley.edu)
# License: BSD
#
# Using: i) window width = (5*2+1), ii) three letter alphabet
# Usage: ./create_database.py 5 3

import os
import sys
import numpy as np

from proteinss import ProteinSS

map_dssp_3_alphabet = {"H":1,"I":1,"G":1,"E":2,"B":2,"T":3,"S":3, "_":3, "?":3}
map_dssp_8_alphabet = {"H":1,"I":2,"G":3,"E":4,"B":5,"T":6,"S":7, "_":8, "?":8}


def extract_file(ss_file):
    print "Extracting :", ss_file

    fi = open(ss_file, 'rU')
    seq_string = ""
    ss_string  = ""
    alignments = []
    for line in fi:
        if line.startswith("RES:"):
           seq_string = line.split(":")[1].rstrip('\n')
           
        if line.startswith("DSSP:"):
           ss_string = line.split(":")[1].rstrip('\n')
 
        if line.startswith("align"):
           alignment = line.split(":")[1].rstrip('\n')
           alignments.append( alignment.split(",")[0:-1] )

    seq_l = seq_string.split(",")[0:-1]
    ss_l  = ss_string.split(",")[0:-1]

    prot = ProteinSS(seq_l, ss_l)

    for al_ in alignments:
        prot.add_alignment(al_)

    return prot


def prepare_db(cb513_path, db_ofile, db_classes, wsize, alphabet):
    fo1 = open(db_ofile, 'w')
    fo2 = open(db_classes, 'w')
    list_ = []
    
    for root, dirs, files in os.walk(cb513_path):
        for name in files:
            if name.endswith(( ".all" )):
                ss_file = root + "/" + name
                prot = extract_file ( ss_file )
                if prot.is_valid():
                    s1, s2 = prot.get_db_strings(wsize, map_dssp_3_alphabet, True)
                    fo1.write(s1)
                    fo2.write(s2)

    fo1.close()
    fo2.close()


if __name__ == "__main__":
    cb513_path = '../data/CB513'
    classes_ofile   = "../data/db/ss_a3.dat" 
    
    window_width = int( sys.argv[1] )
    db_ofile = "../data/db/aa_w" + str(window_width) + "_a3.dat"
    
    if sys.argv[2] == "3":
        alphabet = map_dssp_3_alphabet
    elif sys.argv[2] == "8":
        alphabet = map_dssp_8_alphabet
    else:
        print "Alphabet not recognized: choose between 3 or 8"
        sys.exit(1)

    pubs_data = prepare_db(cb513_path, db_ofile, classes_ofile, window_width, alphabet)


