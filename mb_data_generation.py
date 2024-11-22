# mb_data_generation.py

import numpy as np
import argparse
import pickle
import os

def generate_data(N, output_path, vary_mass):
    # Constants
    k_B = 1.380649e-23  # Boltzmann constant in J/K

    # Data lists
    v_list = []
    T_list = []
    m_list = []
    ln_P_list = []

    for _ in range(N):
        # Random speed between 0 and some upper limit (e.g., 3000 m/s)
        v = np.random.uniform(0.1, 3000)

        # Random temperature between 100 K and 1000 K
        T = np.random.uniform(100, 1000)

        if vary_mass:
            # Random mass between 1e-27 kg and 1e-25 kg
            m = np.random.uniform(1e-27, 1e-25)
        else:
            # Fix mass (e.g., mass of nitrogen molecule)
            m = 4.65e-26  # Mass of N2 molecule in kg

        # Compute ln P(v)
        ln_P = (
            np.log(4 * np.pi) +
            (3/2) * np.log(m / (2 * np.pi * k_B * T)) +
            2 * np.log(v) -
            (m * v**2) / (2 * k_B * T)
        )

        # Append to lists
        v_list.append(v)
        T_list.append(T)
        m_list.append(m)
        ln_P_list.append(ln_P)

    data = {
        'v': np.array(v_list),
        'T': np.array(T_list),
        'm': np.array(m_list),
        'ln_P': np.array(ln_P_list)
    }

    # Save data to a file
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Generate data for Maxwell-Boltzmann equation discovery')
    parser.add_argument('--n', type=int, default=10000, help='Number of data points to generate')
    parser.add_argument('--output_path', type=str, default='mb_data.pkl', help='Path to save the generated data')
    parser.add_argument('--vary_mass', action='store_true', help='Whether to vary particle mass')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    generate_data(args.n, args.output_path, args.vary_mass)
