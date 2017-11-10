import sys
import numpy as np
from sklearn.preprocessing import normalize

SEQ_MAP = {"A":1,"R":2,"N":3,"D":4,"C":5,"Q":6,"E":7,"G":8,"H":9,\
           "I":10,"L":11,"K":12,"M":13,"F":14,"P":15,"S":16,"T":17,\
           "W":18,"Y":19,"V":20,"B":21,"Z":22,"X":23,"-":24, ".":0}

class ProteinSS:

    def __init__(self, aal, ssl):
        self.ss_list = ssl
        self.aa_list = aal
        self.al_list = []

    def get_seq(self):
        return ''.join(self.aa_list)


    def get_ss(self):
        return ''.join(self.ss_list)


    def add_alignment(self, al_):
        self.al_list.append(al_)


    def is_valid(self):
        return len(self.ss_list) == len(self.aa_list)

    def get_db_strings(self, width, ss_map, flag=True):
        N = len(self.ss_list)
        aa_db_string = ""
        ss_db_string = ""
        for i in range(N):
            if flag:
                aa_db_string += self._aa_table_stats(i, width) 
            else:
                aa_db_string += self._aa_substring(i, width)
            
            ss_db_string += str( ss_map[ self.ss_list[i] ] ) + "\n"

        return aa_db_string, ss_db_string


    def _aa_substring(self, i, w):
        w_list = self._aa_substring4seq(self, i, w, self.aa_list)

        for i in range( len(w_list) ):
            w_list[i] = str( w_list[i] )

        final_string = " ".join(w_list) + "\n"
        return final_string

    def _aa_substring4seq(self, i, w, seq_):
        aa_seq = w*"-" + ''.join(seq_) + w * "-"
        # i+w new index of ith aa
        # i+w-w 
        # i+w+w+1 : (i+w) new index; w-window width; +1 include the last one
        w_list = aa_seq[i+w-w:i+w+w+1]
        if len(w_list) != (2*w+1):
            print "error. handle it."
            sys.exit(1)

        w_list = list(w_list)
        
        for i in range( len(w_list) ):
            w_list[i] = SEQ_MAP[w_list[i]]

        return w_list


    
    def _aa_table_stats(self, i, w):

        nrows = len( SEQ_MAP.keys() )
        table_ = np.zeros( ( nrows, 2*w+1 ) )

        w_list = self._aa_substring4seq(i, w, self.aa_list) 

        for i_, el_ in enumerate(w_list):
            table_[el_][i_] += 1

        for al_ in self.al_list:
            if len(al_) != len(self.aa_list):
                continue

            w_list = self._aa_substring4seq(i, w, al_) 
            for i_, el_ in enumerate(w_list):
                table_[ el_ ][i_] += 1
       
        table_ = normalize(table_, axis=0, norm='l1')

        s_ = ""
        for x_ in range(nrows):
            for y_ in range(2*w+1):
                s_ += "%4.3f " % ( table_[x_][y_] )

        s_ += "\n"
        return s_

