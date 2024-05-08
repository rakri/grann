# %%
import hnswlib
import numpy as np
import numpy as np
import h5py
import os
import requests
import tempfile
import time
import struct
import itertools
import sys
from core_utils import read_float_binary_file,read_i8_binary_file,read_u8_binary_file,write_float_binary_file,read_int32_binary_file,compute_recall,get_bin_metadata


# Check if the number of command line arguments is correct.
if len(sys.argv) != 3:
    print("Usage: python hnsw_build.py <M> <efC>")
    sys.exit(1)

# Use the command line arguments in your program.
Mb = int(sys.argv[1])
efb = int(sys.argv[2])
#Mb = 64
#efb = 800

print("Going to build HNSW index with Mb = ", Mb, " and efb = ", efb)

# %%

query_file = '/home/rakri/wiki_rnd1m/wikipedia_query.bin'
gt_file = '/home/rakri/wikipedia_gt_unfilt.bin'
base_file = '/home/rakri/wikipedia/mem_r64_l100.data'
dataset_name='wiki_large'

#gt_file = '/home/rakri/wiki_rnd1m_gt100.bin'
#base_file = '/home/rakri/wiki_rnd1m_data.bin'
#dataset_name='wiki_rnd1m'


[nqgt, nkgt, gt] = read_int32_binary_file(gt_file)
[nq, ndq, queries] = read_float_binary_file(query_file)
[nb, nd] = get_bin_metadata(base_file)

print ("queries,gt, and base metadata loaded")

#names and parameters
index_name = '/home/rakri/hnsw_'+dataset_name+'M='+str(Mb)+'_efC='+str(efb) 

# for hnsw, it is only ef_search
search_params = [(10), (20), (30), (40), (50), (60), (70), (80), (90), (100), (150), (200), (250), (300)]

# %%
# Create an HNSW index
index = hnswlib.Index(space='l2', dim=nd)

index.init_index(max_elements=nb, M = Mb, ef_construction = efb, random_seed = 100)

print ("index initialized")
if os.path.isfile(index_name):
    print("Index exists, so loading")
    index = hnswlib.Index(space='l2', dim=nd)
    index.load_index(index_name, max_elements = nb)
else:
    #index.set_num_threads()
    start = time.time()
    [nb, nd, dataset] = read_float_binary_file(base_file)
    index.add_items(dataset)
    end = time.time()
    print("Index built in", end-start, " seconds.")
    index.save_index(index_name)

# %%
# Search for the nearest neighbor of a query vector

index.set_num_threads(1)
nk=10
print("ef\tRecall\t\ttime")
for param in search_params:
    (ef_search) = param
    index.set_ef(ef_search)
    start = time.time()
    neighbors, distances = index.knn_query(queries, k=nk)
    end = time.time()
    recall = compute_recall(neighbors, gt[:,:nk])
    print(ef_search,"\t",recall,"\t",1000000*(end-start)/nq)
# Print the nearest neighbor
#print(neighbors)
#print(distances)


