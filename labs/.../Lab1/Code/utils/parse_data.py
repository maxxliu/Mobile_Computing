# @Author: Andrea F. Daniele <afdaniele>
# @Date:   Monday, January 22nd 2018
# @Email:  afdaniele@ttic.edu
# @Last modified by:   afdaniele
# @Last modified time: Monday, January 22nd 2018

import os
from os.path import isfile, join
import re
import ast
from utils import ProgressBar


def load_data( data_dir, verbose=True ):
    file_pattern = re.compile("activity-team[1-9]-(Standing|Walking|Jumping|Driving)-([0-9]+).txt");
    # get txt files matching the pattern
    txt_files = [
        f for f in os.listdir(data_dir)
        if  isfile(join(data_dir,f))
            and
            re.match( file_pattern, f )
    ]
    # create different datasets for different activities
    datasets = {}
    print 'Loading data: ',
    pbar = ProgressBar(maxVal=len(txt_files))
    for txt_file in txt_files:
        # read file content
        json_content = None
        with open(join(data_dir, txt_file), 'r') as fopen:
            json_content = fopen.read();
        # convert JSON to python dict
        activity = ast.literal_eval( json_content )
        # add activity to dataset
        activity_type = activity['type']
        activity_data = activity['seq']
        if( activity_type not in datasets ): datasets[activity_type] = []
        datasets[activity_type].extend( activity_data )
        # update progress bar
        pbar.next()

    if verbose:
        # print statistics about the datasets loaded
        print 'Datasets:'
        for activity_type, dataset in datasets.items():
            print '\t%s: %d points' % ( activity_type, len(dataset) )
