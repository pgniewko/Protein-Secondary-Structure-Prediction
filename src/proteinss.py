import sys

SEQ_MAP = {"A":1,"R":2,"N":3,"D":4,"C":5,"Q":6,"E":7,"G":8,"H":9,\
           "I":10,"L":11,"K":12,"M":13,"F":14,"P":15,"S":16,"T":17,\
           "W":18,"Y":19,"V":20,"B":21,"Z":22,"X":23,"-":24}


class ProteinSS:

    def __init__(self, aal, ssl):
        self.ss_list = ssl
        self.aa_list = aal
    
    def get_seq(self):
        return ''.join(self.aa_list)
    
    def get_ss(self):
        return ''.join(self.ss_list)

    def is_valid(self):
        return len(self.ss_list) == len(self.aa_list)

    def get_db_strings(self, width, ss_map):
        N = len(self.ss_list)
        aa_db_string = ""
        ss_db_string = ""
        for i in range(N):
            aa_db_string += self._aa_substring(i, width)
            ss_db_string += str( ss_map[ self.ss_list[i] ] ) + "\n"

        return aa_db_string, ss_db_string

    def _aa_substring(self, i, w):
        aa_seq = w*"-" + self.get_seq() + w * "-"
        # i+w new index of ith aa
        # i+w-w 
        # i+w+w+1 : (i+w) new index; w-window width; +1 include the last one
        w_list = aa_seq[i+w-w:i+w+w+1]
        if len(w_list) != (2*w+1):
            print "error. handle it."
            sys.exit(1)

        w_list = list(w_list)
        
        for i in range( len(w_list) ):
            w_list[i] = str( SEQ_MAP[w_list[i]] )

        final_string = " ".join(w_list) + "\n"
        return final_string




