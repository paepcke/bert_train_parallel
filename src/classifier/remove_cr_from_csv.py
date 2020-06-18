#!/usr/bin/env python3
'''
Created on Jun 18, 2020

@author: paepcke
'''
import csv
import os
import re
import sys


class NewlineRemover(object):
    '''
    classdocs
    '''


    def __init__(self, csv_path):
        '''
        Constructor
        '''
        # A real line starts with a large int:
        true_line_start_pat = re.compile(r'^[\d]{5,}$')
        crlf_pat = re.compile(r'[\n\r]')
        
        with open(csv_path, 'r') as fd:
            reader = csv.reader(fd)
            writer = csv.writer(sys.stdout,
                                quoting=csv.QUOTE_MINIMAL
                                )
            # Spit out the column header:
            writer.writerow(next(reader))
            
            line_accumulator = []
            for line in reader:
                # Line is an array of words. Replace nls:
                if true_line_start_pat.search(line[0]) is not None:
                    # Likely found true new CSV line
                    is_line_start = True
                else:
                    is_line_start = False
                    
                # Remove any embedded \n or \r:
                line_no_nl_arr = [re.sub(crlf_pat, ' ', phrase)
                                     for phrase in line]

                if is_line_start and len(line_accumulator) > 0:
                    # Found start of a real CSV line
                    writer.writerow(line_accumulator)
                    # Remember this first part of the new line.
                    line_accumulator = line_no_nl_arr
                    continue
                else:
                    # Continuation of a line in CSV
                    line_accumulator.extend(line_no_nl_arr)

if __name__ == '__main__':
    
    #********
    #sys.argv = ['remove_cr_from_csv.py', '/Users/paepcke/EclipseWorkspacesNew/facebook_ad_classifier/src/classifier/datasets/facebook_ads.csv']
    #********
    if len(sys.argv) != 2 or sys.argv[1] == '-h' or sys.argv[1] == '--help':
        print("Usage: remove_cr_from_csv.py <path_to_csv_file")
        sys.exit(1)
    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"Path {csv_path} does not exist")
        sys.exit(1)
    NewlineRemover(sys.argv[1])                
