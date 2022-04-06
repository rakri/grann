#!/usr/bin/env python3
import pathlib
import matplotlib.pyplot as plt

labels = [5, 10, 15, 20, 25, 30, 40]
FILE_PREFIX = 'lsh_search_results_'
ALGO = 'lsh'
LABEL_DIST = 'ads'

results = {}

for f in labels:
    results[f] = []
    recalls = []
    mean_cmpss = []
    i = 0
    with open(FILE_PREFIX + str(f), 'r+') as fd:
        num_tables, num_hps, recall, mean_cmps = 0, 0, 0, 0
        for line in fd:
            if line.startswith('/'):
                fname = pathlib.PurePath(line).name.split('_')
                num_tables = int(fname[2])
                num_hps = int(fname[3])
            if line.startswith('   '):
                nums = line.split()
                recall = float(nums[4])
                mean_cmps = float(nums[5])
            if line.startswith('Finished'):
                #print(num_tables, num_hps, recall, mean_cmps)
                results[f].append((num_tables, num_hps, recall, mean_cmps))
                recalls.append(recall)
                mean_cmpss.append(mean_cmps)
                i += 1
                num_tables, num_hps, recall, mean_cmps = 0, 0, 0, 0
       
        plt.xlabel('recall')
        plt.ylabel('mean cmps')
        plt.title('Filter ' + str(f) + ' for ' + ALGO)
        plt.scatter(recalls, mean_cmpss)
        plt.savefig(str(f) + '_' + ALGO + '.png')
        plt.clf()
