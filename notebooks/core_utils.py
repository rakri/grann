import struct
import numpy as np
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


def read_float_binary_file(filename):
    try:
        with open(filename, 'rb') as f:
            # Read the first 8 bytes (number of rows and columns)
            num_rows, num_columns = struct.unpack('ii', f.read(8))
            
            # Read the entire data matrix at once
            raw_data = np.fromfile(f, dtype=np.float32, count=num_rows * num_columns)
            raw_data = raw_data.reshape(num_rows, num_columns)
         
            return num_rows, num_columns, raw_data
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return None

def read_i8_binary_file(filename):
    try:
        with open(filename, 'rb') as f:
            # Read the first 8 bytes (number of rows and columns)
            num_rows, num_columns = struct.unpack('ii', f.read(8))
            
            # Read the entire data matrix at once
            raw_data = np.fromfile(f, dtype=np.int8, count=num_rows * num_columns)
            raw_data = raw_data.reshape(num_rows, num_columns)
            data = raw_data.astype(np.float32)
         
            return num_rows, num_columns, data
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return None

def read_u8_binary_file(filename):
    try:
        with open(filename, 'rb') as f:
            # Read the first 8 bytes (number of rows and columns)
            num_rows, num_columns = struct.unpack('ii', f.read(8))
            
            # Read the entire data matrix at once
            raw_data = np.fromfile(f, dtype=np.uint8, count=num_rows * num_columns)
            raw_data = raw_data.reshape(num_rows, num_columns)
            data = raw_data.astype(np.float32)
         
            return num_rows, num_columns, data
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return None


def write_float_binary_file(filename, data):
# Get the shape of the matrix
    n_rows, n_columns = data.shape

# Pack the metadata and data
    header = struct.pack('ii', n_rows, n_columns)

# Write the header and data to the binary file
    with open(filename, 'wb') as f:
        f.write(header)
        f.write(data)    

def compute_mean(data):
    row_means = np.mean(data, axis=0)    
    return row_means
    
def center_data(data, row_means):
    data -= row_means[np.newaxis,:]

def read_int32_binary_file(filename):
    try:
        with open(filename, 'rb') as f:
            # Read the first 8 bytes (number of rows and columns)
            num_rows, num_columns = struct.unpack('ii', f.read(8))
            
            # Read the entire data matrix at once
            raw_data = np.fromfile(f, dtype=np.uint32, count=num_rows * num_columns)
            raw_data = raw_data.reshape(num_rows, num_columns)
         
            return num_rows, num_columns, raw_data
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return None

def compute_recall(neighbors, true_neighbors):
    total = 0
    for gt_row, row in zip(true_neighbors, neighbors):
        total += np.intersect1d(gt_row, row).shape[0]
    return total / true_neighbors.size