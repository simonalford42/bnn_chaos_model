# data_generation.py

import numpy as np
import argparse
import pickle
import os

def generate_data(N, output_path):
    # Generate N valid cyclic quadrilaterals
    a_list = []
    b_list = []
    c_list = []
    d_list = []
    A_list = []
    s_list = []
    count = 0
    while count < N:
        a = np.random.uniform(1, 10)
        b = np.random.uniform(1, 10)
        c = np.random.uniform(1, 10)
        d = np.random.uniform(1, 10)
        s = (a + b + c + d) / 2
        K = (s - a)*(s - b)*(s - c)*(s - d)
        if K >= 0:
            A = np.sqrt(K)
            a_list.append(a)
            b_list.append(b)
            c_list.append(c)
            d_list.append(d)
            s_list.append(s)
            A_list.append(A)
            count += 1

    data = {
        'a': np.array(a_list),
        'b': np.array(b_list),
        'c': np.array(c_list),
        'd': np.array(d_list),
        's': np.array(s_list),
        'A': np.array(A_list)
    }

    # Save data to a file
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Generate data for Brahmagupta formula discovery')
    parser.add_argument('--n', type=int, default=10000, help='Number of data points to generate')
    parser.add_argument('--output_path', type=str, default='data.pkl', help='Path to save the generated data')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    generate_data(args.n, args.output_path)
