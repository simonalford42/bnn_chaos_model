# import numpy as np
# import argparse
# import pickle
# import os

# def generate_data(N, output_path):
#     # Generate N valid triangles
#     a_list = []
#     b_list = []
#     c_list = []
#     A_list = []
#     count = 0
#     while count < N:
#         # Randomly generate side lengths between 1 and 10
#         a = np.random.uniform(1, 10)
#         b = np.random.uniform(1, 10)
#         c = np.random.uniform(1, 10)

#         # Check if the sides satisfy the triangle inequality
#         if (a + b > c) and (a + c > b) and (b + c > a):
#             # Compute semi-perimeter s internally (not included in data)
#             s = (a + b + c) / 2
#             # Compute area using Heron's formula
#             K = s * (s - a) * (s - b) * (s - c)
#             if K >= 0:
#                 A = np.sqrt(K)
#                 a_list.append(a)
#                 b_list.append(b)
#                 c_list.append(c)
#                 A_list.append(A)
#                 count += 1

#     data = {
#         'a': np.array(a_list),
#         'b': np.array(b_list),
#         'c': np.array(c_list),
#         'A': np.array(A_list)
#     }

#     # Save data to a file
#     with open(output_path, 'wb') as f:
#         pickle.dump(data, f)
#     print(f"Data saved to {output_path}")

# def parse_args():
#     parser = argparse.ArgumentParser(description='Generate data for Heron\'s formula discovery')
#     parser.add_argument('--n', type=int, default=10000, help='Number of data points to generate')
#     parser.add_argument('--output_path', type=str, default='data.pkl', help='Path to save the generated data')
#     args = parser.parse_args()
#     return args

# if __name__ == '__main__':
#     args = parse_args()
#     generate_data(args.n, args.output_path)




import numpy as np
import argparse
import pickle
import os

def generate_data(N, output_path):
    # Generate N valid triangles
    a_list = []
    b_list = []
    c_list = []
    neg_sum_list = []
    A_list = []
    count = 0
    while count < N:
        # Randomly generate side lengths between 1 and 10
        a = np.random.uniform(1, 10)
        b = np.random.uniform(1, 10)
        c = np.random.uniform(1, 10)

        # Check if the sides satisfy the triangle inequality
        if (a + b > c) and (a + c > b) and (b + c > a):
            # Compute semi-perimeter s internally (not included in data)
            s = (a + b + c) / 2
            # Compute area using Heron's formula
            K = s * (s - a) * (s - b) * (s - c)
            if K > 0:
                A = np.sqrt(K)
                neg_sum = -a - b - c  # Compute the additional feature
                a_list.append(a)
                b_list.append(b)
                c_list.append(c)
                neg_sum_list.append(neg_sum)
                A_list.append(A)
                count += 1

    data = {
        'a': np.array(a_list),
        'b': np.array(b_list),
        'c': np.array(c_list),
        '-(a+b+c)': np.array(neg_sum_list),
        'A': np.array(A_list)
    }

    # Save data to a file
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Generate data for Heron\'s formula discovery with additional feature')
    parser.add_argument('--n', type=int, default=10000, help='Number of data points to generate')
    parser.add_argument('--output_path', type=str, default='data_with_neg_sum.pkl', help='Path to save the generated data')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    generate_data(args.n, args.output_path)




# import numpy as np
# import argparse
# import pickle
# import os

# def generate_data(N, output_path):
#     # Generate N valid triangles
#     a_list = []
#     b_list = []
#     c_list = []
#     lnA_list = []
#     count = 0
#     while count < N:
#         # Randomly generate side lengths between 1 and 10
#         a = np.random.uniform(1, 10)
#         b = np.random.uniform(1, 10)
#         c = np.random.uniform(1, 10)

#         # Check if the sides satisfy the triangle inequality
#         if (a + b > c) and (a + c > b) and (b + c > a):
#             # Compute semi-perimeter s internally (not included in data)
#             s = (a + b + c) / 2
#             # Compute ln(area) using the logarithm of Heron's formula
#             K = s * (s - a) * (s - b) * (s - c)
#             if K > 0:
#                 lnA = 0.5 * (np.log(s) + np.log(s - a) + np.log(s - b) + np.log(s - c))
#                 a_list.append(a)
#                 b_list.append(b)
#                 c_list.append(c)
#                 lnA_list.append(lnA)
#                 count += 1

#     data = {
#         'a': np.array(a_list),
#         'b': np.array(b_list),
#         'c': np.array(c_list),
#         'lnA': np.array(lnA_list)
#     }

#     # Save data to a file
#     with open(output_path, 'wb') as f:
#         pickle.dump(data, f)
#     print(f"Data saved to {output_path}")

# def parse_args():
#     parser = argparse.ArgumentParser(description='Generate data for log of Heron\'s formula discovery')
#     parser.add_argument('--n', type=int, default=10000, help='Number of data points to generate')
#     parser.add_argument('--output_path', type=str, default='heron_log_data.pkl', help='Path to save the generated data')
#     args = parser.parse_args()
#     return args

# if __name__ == '__main__':
#     args = parse_args()
#     generate_data(args.n, args.output_path)
