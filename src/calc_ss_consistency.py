#! /usr/bin/env python
# Author PG
# License: BSD
#

import os
import sys
import numpy as np

map_dssp_3codes   = {"H":1,"I":1,"G":1,"E":2,"B":2,"T":3,"S":3, "_":3, "?":3}
map_stride_3codes = {"H":1,"I":1,"G":1,"E":2,"B":2,"T":3,"C":3, "?":3}
map_define_3codes = {"H":1,"G":1,"E":2,"_":3,"S":3,"?":3}

def calc_consistency(ss1, ss2, ss3):
    N1 = len(ss1)
    N2 = len(ss2)
    N3 = len(ss3)

    if N1 != N2:
        print "N1 != N2"
    
    if N1 != N3:
        print "N1 != N3"
    
    if N2 != N3:
        print "N2 != N3"


    s = 0.0
    s_total = 0.0
    for i in range(N1):
        if map_dssp_3codes[ ss1[i] ] == map_stride_3codes[ ss2[i].upper() ]:
            s += 1
    s_total += (s / N1)

    s = 0.0
    for i in range(N1):
        if map_dssp_3codes[ ss1[i] ] == map_define_3codes[ ss3[i] ]:
            s += 1
    s_total += (s / N1)
    
    s = 0.0
    for i in range(N1):
        if map_stride_3codes[ ss2[i].upper() ] == map_define_3codes[ ss3[i] ]:
            s += 1
    s_total += (s / N1)


    s_total /= 3.0
    return s_total


def extract_ss(ss_file):
    print "Extracting :", ss_file

    fi = open(ss_file, 'rU')
    
    ss1 = ""
    ss2 = ""
    ss3 = ""
    
    for line in fi:
        if line.startswith("DSSP:"):
           ss1 = line.split(":")[1].rstrip('\n')
        
        if line.startswith("STRIDE:"):
           ss2 = line.split(":")[1].rstrip('\n')
        
        if line.startswith("DEFINE:"):
           ss3 = line.split(":")[1].rstrip('\n')
        
    ss1  = ss1.split(",")[0:-1]
    ss2  = ss2.split(",")[0:-1]
    ss3  = ss3.split(",")[0:-1]

    return ss1, ss2, ss3


def scan_db(cb513_path):

    counter = 0
    average = 0.0

    for root, dirs, files in os.walk(cb513_path):
        for name in files:
            if name.endswith(( ".all" )):
                ss_file = root + "/" + name
                ss1, ss2, ss3 = extract_ss ( ss_file )
                counter += 1
                average += calc_consistency(ss1, ss2, ss3)


    return average / counter

if __name__ == "__main__":
    cb513_path = '../data/CB513'
    
    average_consistency = scan_db(cb513_path)
    print average_consistency
