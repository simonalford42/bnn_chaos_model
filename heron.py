import subprocess
import wandb
import pysr
from pysr import PySRRegressor
import random

import numpy as np
import argparse
import os
import pickle

def load_data(config):
    # Load data from file
    with open(config['data_path'], 'rb') as f:
        data = pickle.load(f)

    # Include the new feature '-(a+b+c)'
    X = np.vstack([data['a'], data['b'], data['c'], data['-(a+b+c)']]).T
    variable_names = ['a', 'b', 'c', 'neg_sum']

    y = data['A'].reshape(-1, 1)  # Area of the triangle
    return X, y, variable_names

def get_config(args):
    id = random.randint(0, 100000)
    while os.path.exists(f'sr_results/{id}.pkl'):
        id = random.randint(0, 100000)

    if not os.path.exists('sr_results'):
        os.makedirs('sr_results')

    path = f'sr_results/{id}.pkl'
    # Replace '.pkl' with '.csv' for the equation file
    csv_path = path[:-3] + 'csv'

    num_cpus = os.cpu_count()
    pysr_config = dict(
        procs=num_cpus,
        populations=3 * num_cpus,
        batching=True,
        equation_file=csv_path,
        niterations=args.niterations,
        binary_operators=["+", "*", '/', '-'],
        unary_operators=['sqrt'],
        maxsize=args.max_size,
        timeout_in_seconds=int(60 * 60 * args.time_in_hours),
        constraints={'^': (-1, 1)},
        nested_constraints={"sqrt": {"sqrt": 0}},
        ncyclesperiteration=1000,
        random_state=args.seed, 
        complexity_of_constants=1000
    )

    config = vars(args)
    config.update(pysr_config)
    config['pysr_config'] = pysr_config
    config.update({
        'id': id,
        'equation_file': csv_path,
    })

    return config

def run_pysr(config):
    # Set random seeds for reproducibility
    seed = config.get('seed', 0)
    np.random.seed(seed)
    random.seed(seed)

    X, y, variable_names = load_data(config)

    print(f"Shape of X: {X.shape}")
    num_features = X.shape[1]
    print(f"Number of features in X: {num_features}")

    model_kwargs = config['pysr_config']

    # Default to None; will be set if sr_residual is True
    complexity_of_variables = None

    if config.get('sr_residual', False):
        # Load the previous PySR results
        with open(config['previous_sr_path'], 'rb') as f:
            previous_sr_model = pickle.load(f)

        # Evaluate previous model equations on X
        flag = config.get('flag', 'topk')  # Can be 'all', 'top', 'topk'
        variable_name_map = {}  # To store variable names and their equations
        additional_features = []

        # Get the equations DataFrame
        equations_df = previous_sr_model.equations_
        equations = equations_df

        # Start with complexities of 1 for original features
        complexities = [1] * X.shape[1]

        if flag == "top":
            # Get the equation with the highest complexity
            max_complexity = equations['complexity'].max()
            max_complexity_eq = equations[equations['complexity'] == max_complexity].iloc[0]
            lambda_func = max_complexity_eq['lambda_format']
            evaluated_result = lambda_func(X).reshape(-1, 1)
            additional_features.append(evaluated_result)
            variable_name = f'prev_eq_complexity_{max_complexity}'
            variable_name_map[variable_name] = (max_complexity, max_complexity_eq['equation'])
            variable_names += [variable_name]
            # Assign complexity 1 to the new feature
            complexities += [1]
            X = np.hstack([X, evaluated_result])

        elif flag == "topk":
            # Select equations with specific complexities
            selected_complexities = set(map(int, config.get('selected_complexities', '8').split(',')))
            selected_equations = equations[equations['complexity'].isin(selected_complexities)]
            for idx, row in selected_equations.iterrows():
                lambda_func = row['lambda_format']
                evaluated_result = lambda_func(X).reshape(-1, 1)
                additional_features.append(evaluated_result)
                variable_name = f'prev_eq_complexity_{row["complexity"]}'
                variable_name_map[variable_name] = (row['complexity'], row['equation'])
                variable_names += [variable_name]
                # Assign complexity 1 to each added feature
                complexities.append(1)
            if additional_features:
                X = np.hstack([X] + additional_features)

        elif flag == "all":
            for idx, row in equations.iterrows():
                lambda_func = row['lambda_format']
                evaluated_result = lambda_func(X).reshape(-1, 1)
                additional_features.append(evaluated_result)
                variable_name = f'prev_eq_complexity_{row["complexity"]}'
                variable_name_map[variable_name] = (row['complexity'], row['equation'])
                variable_names += [variable_name]
                # Assign complexity 1 to each added feature
                complexities.append(1)
            if additional_features:
                X = np.hstack([X] + additional_features)

        else:
            raise ValueError(f"Unknown flag: {flag}")

        # Set complexity_of_variables to the updated complexities list
        complexity_of_variables = complexities

        # Print the variable names and their equations
        print("Variable names, their associated complexities, and equations:")
        for variable_name, (original_complexity, equation_str) in variable_name_map.items():
            print(f"{variable_name}: original complexity {original_complexity}, assigned complexity 1, equation: {equation_str}")

    model = pysr.PySRRegressor(**model_kwargs)

    if not config['no_log']:
        wandb.init(
            entity='bnn-chaos-model',
            project='planets-sr',
            config=config,
        )

    # Fit the model, passing complexity_of_variables if set
    if complexity_of_variables is not None:
        print("Running PySR with custom complexity_of_variables (all complexities set to 1 for reused equations).")
        model.fit(X, y, variable_names=variable_names, complexity_of_variables=complexity_of_variables)
    else:
        print("Running PySR without custom complexity_of_variables.")
        model.fit(X, y, variable_names=variable_names)

    print('Done running PySR')

    # Save the model
    with open(f'sr_results/{config["id"]}.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Print the discovered equations
    print(model)

    # Optionally log to wandb
    if not config['no_log']:
        losses = [min(eqs['loss']) for eqs in model.equation_file_contents_]
        wandb.log({'avg_loss': sum(losses) / len(losses),
                   'losses': losses,
                   })

    # Try to delete Julia backup files
    try:
        subprocess.run('rm julia*.out', shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while trying to delete the backup files: {e}")

    print(f"Saved to path: {config['equation_file']}")

def parse_args():
    parser = argparse.ArgumentParser(description='Run PySR for Heron\'s formula discovery with additional feature')
    parser.add_argument('--data_path', type=str, default='data_with_neg_sum.pkl', help='Path to the data file')
    parser.add_argument('--no_log', action='store_true', default=False, help='Disable WandB logging')
    parser.add_argument('--time_in_hours', type=float, default=0.5, help='Time limit for PySR run in hours')
    parser.add_argument('--niterations', type=int, default=100000, help='Number of iterations for PySR')
    parser.add_argument('--max_size', type=int, default=50, help='Maximum size of equations in PySR')
    parser.add_argument('--sr_residual', action='store_true', help='Use residual PySR run with previous model')
    parser.add_argument('--previous_sr_path', type=str, default='sr_results/first_run.pkl', help='Path to previous PySR model')
    parser.add_argument('--flag', type=str, default='topk', choices=['all', 'top', 'topk'], help='Flag to select equations from previous run')
    parser.add_argument('--selected_complexities', type=str, default='8', help='Comma-separated complexities to select for topk')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    config = get_config(args)
    run_pysr(config)



# import subprocess
# import wandb
# import pysr
# from pysr import PySRRegressor
# import random

# import numpy as np
# import argparse
# import os
# import pickle

# def load_data(config):
#     # Load data from file
#     with open(config['data_path'], 'rb') as f:
#         data = pickle.load(f)

#     # For Heron's formula, we only have sides a, b, c
#     X = np.vstack([data['a'], data['b'], data['c']]).T
#     variable_names = ['a', 'b', 'c']

#     y = data['A'].reshape(-1, 1)  # Area of the triangle
#     return X, y, variable_names

# def get_config(args):
#     id = random.randint(0, 100000)
#     while os.path.exists(f'sr_results/{id}.pkl'):
#         id = random.randint(0, 100000)

#     if not os.path.exists('sr_results'):
#         os.makedirs('sr_results')

#     path = f'sr_results/{id}.pkl'
#     # Replace '.pkl' with '.csv' for the equation file
#     csv_path = path[:-3] + 'csv'

#     num_cpus = os.cpu_count()
#     pysr_config = dict(
#         procs=num_cpus,
#         populations=3*num_cpus,
#         batching=True,
#         equation_file=csv_path,
#         niterations=args.niterations,
#         binary_operators=["+", "*", '/', '-', '^'],
#         unary_operators=['sqrt'],
#         maxsize=args.max_size,
#         timeout_in_seconds=int(60*60*args.time_in_hours),
#         constraints={'^': (-1, 1)},
#         nested_constraints={"sqrt": {"sqrt": 0}},
#         ncyclesperiteration=1000,
#         random_state=args.seed,  # Include the seed for reproducibility
#     )

#     config = vars(args)
#     config.update(pysr_config)
#     config['pysr_config'] = pysr_config
#     config.update({
#         'id': id,
#         'equation_file': csv_path,
#     })

#     return config

# def run_pysr(config):
#     # Set random seeds for reproducibility
#     seed = config.get('seed', 0)
#     np.random.seed(seed)
#     random.seed(seed)

#     X, y, variable_names = load_data(config)

#     print(f"Shape of X: {X.shape}")
#     num_features = X.shape[1]
#     print(f"Number of features in X: {num_features}")

#     model_kwargs = config['pysr_config']

#     # Default to None; will be set if sr_residual is True
#     complexity_of_variables = None

#     if config.get('sr_residual', False):
#         # Load the previous PySR results
#         with open(config['previous_sr_path'], 'rb') as f:
#             previous_sr_model = pickle.load(f)

#         # Evaluate previous model equations on X
#         flag = config.get('flag', 'topk')  # Can be 'all', 'top', 'topk'
#         variable_name_map = {}  # To store variable names and their equations
#         additional_features = []

#         # Get the equations DataFrame
#         equations_df = previous_sr_model.equations_
#         equations = equations_df

#         # Start with complexities of 1 for original features
#         complexities = [1] * X.shape[1]

#         if flag == "top":
#             # Get the equation with the highest complexity
#             max_complexity = equations['complexity'].max()
#             max_complexity_eq = equations[equations['complexity'] == max_complexity].iloc[0]
#             lambda_func = max_complexity_eq['lambda_format']
#             evaluated_result = lambda_func(X).reshape(-1, 1)
#             additional_features.append(evaluated_result)
#             variable_name = f'prev_eq_complexity_{max_complexity}'
#             variable_name_map[variable_name] = (max_complexity, max_complexity_eq['equation'])
#             variable_names += [variable_name]
#             # Set complexity of variables: Assign 1 to the new feature
#             complexities += [1]
#             X = np.hstack([X, evaluated_result])

#         elif flag == "topk":
#             # Select equations with specific complexities
#             selected_complexities = set(map(int, config.get('selected_complexities', '8').split(',')))
#             selected_equations = equations[equations['complexity'].isin(selected_complexities)]
#             for idx, row in selected_equations.iterrows():
#                 lambda_func = row['lambda_format']
#                 evaluated_result = lambda_func(X).reshape(-1, 1)
#                 additional_features.append(evaluated_result)
#                 variable_name = f'prev_eq_complexity_{row["complexity"]}'
#                 variable_name_map[variable_name] = (row['complexity'], row['equation'])
#                 variable_names += [variable_name]
#                 # Assign complexity 1 to each added feature
#                 complexities.append(1)
#             # Set complexity of variables
#             complexity_of_variables = complexities
#             if additional_features:
#                 X = np.hstack([X] + additional_features)

#         elif flag == "all":
#             for idx, row in equations.iterrows():
#                 lambda_func = row['lambda_format']
#                 evaluated_result = lambda_func(X).reshape(-1, 1)
#                 additional_features.append(evaluated_result)
#                 variable_name = f'prev_eq_complexity_{row["complexity"]}'
#                 variable_name_map[variable_name] = (row['complexity'], row['equation'])
#                 variable_names += [variable_name]
#                 # Assign complexity 1 to each added feature
#                 complexities.append(1)
#             # Set complexity of variables
#             complexity_of_variables = complexities
#             if additional_features:
#                 X = np.hstack([X] + additional_features)

#         else:
#             raise ValueError(f"Unknown flag: {flag}")

#         # Set complexity_of_variables to the updated complexities list
#         complexity_of_variables = complexities

#         # Print the variable names and their equations
#         print("Variable names, their associated complexities, and equations:")
#         for variable_name, (original_complexity, equation_str) in variable_name_map.items():
#             print(f"{variable_name}: original complexity {original_complexity}, assigned complexity 1, equation: {equation_str}")

#     model = pysr.PySRRegressor(**model_kwargs)

#     if not config['no_log']:
#         wandb.init(
#             entity='bnn-chaos-model',
#             project='planets-sr',
#             config=config,
#         )

#     # Fit the model, passing complexity_of_variables if set
#     if complexity_of_variables is not None:
#         print("Running PySR with custom complexity_of_variables (all complexities set to 1 for reused equations).")
#         model.fit(X, y, variable_names=variable_names, complexity_of_variables=complexity_of_variables)
#     else:
#         print("Running PySR without custom complexity_of_variables.")
#         model.fit(X, y, variable_names=variable_names)

#     print('Done running PySR')

#     # Save the model
#     with open(f'sr_results/{config["id"]}.pkl', 'wb') as f:
#         pickle.dump(model, f)

#     # Print the discovered equations
#     print(model)

#     losses = [min(eqs['loss']) for eqs in model.equation_file_contents_]

#     if not config['no_log']:
#         wandb.log({'avg_loss': sum(losses)/len(losses),
#                    'losses': losses,
#                    })

#     # Try to delete Julia backup files
#     try:
#         subprocess.run(f'rm julia*.out', shell=True, check=True)
#     except subprocess.CalledProcessError as e:
#         print(f"An error occurred while trying to delete the backup files: {e}")

#     print(f"Saved to path: {config['equation_file']}")

# def parse_args():
#     parser = argparse.ArgumentParser(description='Run PySR for Heron\'s formula discovery')
#     parser.add_argument('--data_path', type=str, default='heron_data.pkl', help='Path to the data file')
#     parser.add_argument('--no_log', action='store_true', default=False, help='Disable WandB logging')
#     parser.add_argument('--time_in_hours', type=float, default=0.5, help='Time limit for PySR run in hours')
#     parser.add_argument('--niterations', type=int, default=100000, help='Number of iterations for PySR')
#     parser.add_argument('--max_size', type=int, default=50, help='Maximum size of equations in PySR')
#     parser.add_argument('--sr_residual', action='store_true', help='Use residual PySR run with previous model')
#     parser.add_argument('--previous_sr_path', type=str, default='sr_results/first_run.pkl', help='Path to previous PySR model')
#     parser.add_argument('--flag', type=str, default='topk', choices=['all', 'top', 'topk'], help='Flag to select equations from previous run')
#     parser.add_argument('--selected_complexities', type=str, default='8', help='Comma-separated complexities to select for topk')
#     parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
#     args = parser.parse_args()
#     return args

# if __name__ == '__main__':
#     args = parse_args()
#     config = get_config(args)
#     run_pysr(config)


# import subprocess
# import wandb
# import pysr
# from pysr import PySRRegressor
# import random

# import numpy as np
# import argparse
# import os
# import pickle

# def load_data(config):
#     # Load data from file
#     with open(config['data_path'], 'rb') as f:
#         data = pickle.load(f)

#     # For Heron's formula, we only have sides a, b, c
#     X = np.vstack([data['a'], data['b'], data['c']]).T
#     variable_names = ['a', 'b', 'c']

#     y = data['lnA'].reshape(-1, 1)  # Logarithm of the area of the triangle
#     return X, y, variable_names

# def get_config(args):
#     id = random.randint(0, 100000)
#     while os.path.exists(f'sr_results/{id}.pkl'):
#         id = random.randint(0, 100000)

#     if not os.path.exists('sr_results'):
#         os.makedirs('sr_results')

#     path = f'sr_results/{id}.pkl'
#     # Replace '.pkl' with '.csv' for the equation file
#     csv_path = path[:-3] + 'csv'

#     num_cpus = os.cpu_count()
#     pysr_config = dict(
#         procs=num_cpus,
#         populations=3*num_cpus,
#         batching=True,
#         equation_file=csv_path,
#         niterations=args.niterations,
#         binary_operators=["+", "*", '/', '-', '^'],
#         unary_operators=['log'],
#         maxsize=args.max_size,
#         timeout_in_seconds=int(60*60*args.time_in_hours),
#         constraints={'^': (-1, 1)},
#         nested_constraints={"log": {"log": 0}},
#         ncyclesperiteration=1000,
#         random_state=args.seed,  # Include the seed for reproducibility
#     )

#     config = vars(args)
#     config.update(pysr_config)
#     config['pysr_config'] = pysr_config
#     config.update({
#         'id': id,
#         'equation_file': csv_path,
#     })

#     return config

# def run_pysr(config):
#     # Set random seeds for reproducibility
#     seed = config.get('seed', 0)
#     np.random.seed(seed)
#     random.seed(seed)

#     X, y, variable_names = load_data(config)

#     print(f"Shape of X: {X.shape}")
#     num_features = X.shape[1]
#     print(f"Number of features in X: {num_features}")

#     model_kwargs = config['pysr_config']

#     # Default to None; will be set if sr_residual is True
#     complexity_of_variables = None

#     if config.get('sr_residual', False):
#         # Load the previous PySR results
#         with open(config['previous_sr_path'], 'rb') as f:
#             previous_sr_model = pickle.load(f)

#         # Evaluate previous model equations on X
#         flag = config.get('flag', 'topk')  # Can be 'all', 'top', 'topk'
#         variable_name_map = {}  # To store variable names and their equations
#         additional_features = []

#         # Get the equations DataFrame
#         equations_df = previous_sr_model.equations_

#         # Use equations from the first output (assuming single output)
#         equations = equations_df[0]

#         # Start with complexities of 1 for original features
#         complexities = [1] * X.shape[1]

#         if flag == "top":
#             # Get the equation with the highest complexity
#             max_complexity = equations['complexity'].max()
#             max_complexity_eq = equations[equations['complexity'] == max_complexity].iloc[0]
#             lambda_func = max_complexity_eq['lambda_format']
#             evaluated_result = lambda_func(X).reshape(-1, 1)
#             additional_features.append(evaluated_result)
#             variable_name = f'prev_eq_complexity_{max_complexity}'
#             variable_name_map[variable_name] = (max_complexity, max_complexity_eq['equation'])
#             variable_names += [variable_name]
#             # Set complexity of variables: Assign 1 to the new feature
#             complexities += [1]
#             X = np.hstack([X, evaluated_result])

#         elif flag == "topk":
#             # Select equations with specific complexities
#             selected_complexities = set(map(int, config.get('selected_complexities', '8').split(',')))
#             selected_equations = equations[equations['complexity'].isin(selected_complexities)]
#             for idx, row in selected_equations.iterrows():
#                 lambda_func = row['lambda_format']
#                 evaluated_result = lambda_func(X).reshape(-1, 1)
#                 additional_features.append(evaluated_result)
#                 variable_name = f'prev_eq_complexity_{row["complexity"]}'
#                 variable_name_map[variable_name] = (row['complexity'], row['equation'])
#                 variable_names += [variable_name]
#                 # Assign complexity 1 to each added feature
#                 complexities.append(1)
#             # Set complexity of variables
#             complexity_of_variables = complexities
#             if additional_features:
#                 X = np.hstack([X] + additional_features)

#         elif flag == "all":
#             for idx, row in equations.iterrows():
#                 lambda_func = row['lambda_format']
#                 evaluated_result = lambda_func(X).reshape(-1, 1)
#                 additional_features.append(evaluated_result)
#                 variable_name = f'prev_eq_complexity_{row["complexity"]}'
#                 variable_name_map[variable_name] = (row['complexity'], row['equation'])
#                 variable_names += [variable_name]
#                 # Assign complexity 1 to each added feature
#                 complexities.append(1)
#             # Set complexity of variables
#             complexity_of_variables = complexities
#             if additional_features:
#                 X = np.hstack([X] + additional_features)

#         else:
#             raise ValueError(f"Unknown flag: {flag}")

#         # Set complexity_of_variables to the updated complexities list
#         complexity_of_variables = complexities

#         # Print the variable names and their equations
#         print("Variable names, their associated complexities, and equations:")
#         for variable_name, (original_complexity, equation_str) in variable_name_map.items():
#             print(f"{variable_name}: original complexity {original_complexity}, assigned complexity 1, equation: {equation_str}")

#     model = pysr.PySRRegressor(**model_kwargs)

#     if not config['no_log']:
#         wandb.init(
#             entity='bnn-chaos-model',
#             project='planets-sr',
#             config=config,
#         )

#     # Fit the model, passing complexity_of_variables if set
#     if complexity_of_variables is not None:
#         print("Running PySR with custom complexity_of_variables (all complexities set to 1 for reused equations).")
#         model.fit(X, y, variable_names=variable_names, complexity_of_variables=complexity_of_variables)
#     else:
#         print("Running PySR without custom complexity_of_variables.")
#         model.fit(X, y, variable_names=variable_names)

#     print('Done running PySR')

#     # Save the model
#     with open(f'sr_results/{config["id"]}.pkl', 'wb') as f:
#         pickle.dump(model, f)

#     # Print the discovered equations
#     print(model)

#     losses = [min(eqs['loss']) for eqs in model.equation_file_contents_]

#     if not config['no_log']:
#         wandb.log({'avg_loss': sum(losses)/len(losses),
#                    'losses': losses,
#                    })

#     # Try to delete Julia backup files
#     try:
#         subprocess.run(f'rm julia*.out', shell=True, check=True)
#     except subprocess.CalledProcessError as e:
#         print(f"An error occurred while trying to delete the backup files: {e}")

#     print(f"Saved to path: {config['equation_file']}")

# def parse_args():
#     parser = argparse.ArgumentParser(description='Run PySR for log of Heron\'s formula discovery')
#     parser.add_argument('--data_path', type=str, default='heron_log_data.pkl', help='Path to the data file')
#     parser.add_argument('--no_log', action='store_true', default=False, help='Disable WandB logging')
#     parser.add_argument('--time_in_hours', type=float, default=0.5, help='Time limit for PySR run in hours')
#     parser.add_argument('--niterations', type=int, default=100000, help='Number of iterations for PySR')
#     parser.add_argument('--max_size', type=int, default=50, help='Maximum size of equations in PySR')
#     parser.add_argument('--sr_residual', action='store_true', help='Use residual PySR run with previous model')
#     parser.add_argument('--previous_sr_path', type=str, default='sr_results/first_run.pkl', help='Path to previous PySR model')
#     parser.add_argument('--flag', type=str, default='topk', choices=['all', 'top', 'topk'], help='Flag to select equations from previous run')
#     parser.add_argument('--selected_complexities', type=str, default='8', help='Comma-separated complexities to select for topk')
#     parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
#     args = parser.parse_args()
#     return args

# if __name__ == '__main__':
#     args = parse_args()
#     config = get_config(args)
#     run_pysr(config)


# import subprocess
# import wandb
# import pysr
# from pysr import PySRRegressor
# import random

# import numpy as np
# import argparse
# import os
# import pickle

# def load_data(config):
#     # Load data from file
#     with open(config['data_path'], 'rb') as f:
#         data = pickle.load(f)

#     if config.get('include_s', False):
#         X = np.vstack([data['a'], data['b'], data['c'], data['d'], data['s']]).T
#         variable_names = ['a', 'b', 'c', 'd', 's']
#     else:
#         X = np.vstack([data['a'], data['b'], data['c'], data['d']]).T
#         variable_names = ['a', 'b', 'c', 'd']

#     y = data['A'].reshape(-1, 1)
#     return X, y, variable_names

# def get_config(args):
#     id = random.randint(0, 100000)
#     while os.path.exists(f'sr_results/{id}.pkl'):
#         id = random.randint(0, 100000)

#     if not os.path.exists('sr_results'):
#         os.makedirs('sr_results')

#     path = f'sr_results/{id}.pkl'
#     # Replace '.pkl' with '.csv' for the equation file
#     csv_path = path[:-3] + 'csv'

#     num_cpus = os.cpu_count()
#     pysr_config = dict(
#         procs=num_cpus,
#         populations=3*num_cpus,
#         batching=True,
#         equation_file=csv_path,
#         niterations=args.niterations,
#         binary_operators=["+", "*", '/', '-', '^'],
#         unary_operators=['sqrt'],
#         maxsize=args.max_size,
#         timeout_in_seconds=int(60*60*args.time_in_hours),
#         constraints={'^': (-1, 1)},
#         nested_constraints={"sqrt": {"sqrt": 0}},
#         ncyclesperiteration=1000,
#     )

#     config = vars(args)
#     config.update(pysr_config)
#     config['pysr_config'] = pysr_config
#     config.update({
#         'id': id,
#         'equation_file': csv_path,
#     })

#     return config

# def run_pysr(config):
#     X, y, variable_names = load_data(config)

#     print(f"Shape of X: {X.shape}")
#     num_features = X.shape[1]
#     print(f"Number of features in X: {num_features}")

#     model_kwargs = config['pysr_config']

#     # Default complexities are 1 for all features
#     complexity_of_variables = [1] * X.shape[1]

#     if config.get('sr_residual', False):
#         # Load the previous PySR results
#         with open(config['previous_sr_path'], 'rb') as f:
#             previous_sr_model = pickle.load(f)

#         # Decide which equations to use based on the 'flag' parameter
#         flag = config.get('flag', 'topk')

#         variable_name_map = {}
#         additional_features = []

#         equations_df = previous_sr_model.equations_
#         equations = equations_df[0]

#         if flag == "top":
#             max_complexity = equations['complexity'].max()
#             max_complexity_eq = equations[equations['complexity'] == max_complexity].iloc[0]
#             lambda_func = max_complexity_eq['lambda_format']
#             evaluated_result = lambda_func(X)
#             evaluated_result = evaluated_result.reshape(-1, 1)
#             additional_features.append(evaluated_result)
#             variable_name = f'prev_eq_complexity_{max_complexity}'
#             variable_name_map[variable_name] = (max_complexity, max_complexity_eq['equation'])
#             variable_names += [variable_name]
#             complexity_of_variables += [max_complexity]
#             X = np.hstack([X, evaluated_result])

#         elif flag == "topk":
#             selected_complexities = set(map(int, config.get('selected_complexities', '8').split(',')))
#             selected_equations = equations[equations['complexity'].isin(selected_complexities)]
#             for idx, row in selected_equations.iterrows():
#                 lambda_func = row['lambda_format']
#                 evaluated_result = lambda_func(X)
#                 evaluated_result = evaluated_result.reshape(-1, 1)
#                 additional_features.append(evaluated_result)
#                 variable_name = f'prev_eq_complexity_{row["complexity"]}'
#                 variable_name_map[variable_name] = (row['complexity'], row['equation'])
#                 variable_names += [variable_name]
#                 complexity_of_variables.append(row['complexity'])
#             if additional_features:
#                 X = np.hstack([X] + additional_features)

#         elif flag == "all":
#             for idx, row in equations.iterrows():
#                 lambda_func = row['lambda_format']
#                 evaluated_result = lambda_func(X)
#                 evaluated_result = evaluated_result.reshape(-1, 1)
#                 additional_features.append(evaluated_result)
#                 variable_name = f'prev_eq_complexity_{row["complexity"]}'
#                 variable_name_map[variable_name] = (row['complexity'], row['equation'])
#                 variable_names += [variable_name]
#                 complexity_of_variables.append(row['complexity'])
#             if additional_features:
#                 X = np.hstack([X] + additional_features)

#         else:
#             raise ValueError(f"Unknown flag: {flag}")

#         # Print the variable names and their equations
#         print("Variable names, their associated complexities, and equations:")
#         for variable_name, (complexity, equation_str) in variable_name_map.items():
#             print(f"{variable_name}: complexity {complexity}, equation: {equation_str}")

#         # Generate the custom complexity mapping function
#         num_features = X.shape[1]
#         N0 = num_features - len(additional_features)
#         max_features = num_features
#         added_feature_complexities = complexity_of_variables[N0:]

#         # Generate Julia code for complexity_mapping
#         complexity_mapping_code = f"""
# function my_complexity(expr)
#     tree = get_tree(expr)
#     base_complexity = count(node -> node.degree != 0 || node.constant, tree)
#     max_features = {max_features}
#     feature_count = zeros(UInt16, max_features)
#     foreach(tree) do node
#         if node.degree == 0 && !node.constant
#             feature_count[node.feature + 1] += 1
#         end
#     end
#     complexity = base_complexity + (
#         sum(feature_count[1:{N0}])  # Original features, complexity 1 per occurrence
# """

#         for idx, complexity in enumerate(added_feature_complexities):
#             feature_idx = N0 + idx
#             julia_feature_idx = feature_idx + 1  # Julia uses 1-based indexing
#             complexity_mapping_code += f"""
#         + (feature_count[{julia_feature_idx}] > 0 ? {complexity} + (feature_count[{julia_feature_idx}] - 1) : 0)
# """

#         complexity_mapping_code += """
#      )
#      return Int(complexity)
# end
# """

#         # Assign the custom complexity mapping
#         model_kwargs['complexity_of_variables'] = None
#         model_kwargs['complexity_mapping'] = complexity_mapping_code

#     else:
#         model_kwargs['complexity_of_variables'] = complexity_of_variables

#     model = pysr.PySRRegressor(**model_kwargs)

#     if not config['no_log']:
#         wandb.init(
#             entity='bnn-chaos-model',
#             project='planets-sr',
#             config=config,
#         )

#     # Fit the model
#     print("Running PySR...")
#     model.fit(X, y, variable_names=variable_names)

#     print('Done running PySR')

#     # Save the model
#     with open(f'sr_results/{config["id"]}.pkl', 'wb') as f:
#         pickle.dump(model, f)

#     # Print the discovered equations
#     print(model)

#     losses = [min(eqs['loss']) for eqs in model.equation_file_contents_]

#     if not config['no_log']:
#         wandb.log({'avg_loss': sum(losses)/len(losses),
#                    'losses': losses,
#                    })

#     # Try to delete Julia backup files
#     try:
#         subprocess.run(f'rm julia*.out', shell=True, check=True)
#     except subprocess.CalledProcessError as e:
#         print(f"An error occurred while trying to delete the backup files: {e}")

#     print(f"Saved to path: {config['equation_file']}")

# def parse_args():
#     parser = argparse.ArgumentParser(description='Run PySR for Brahmagupta formula discovery')
#     parser.add_argument('--data_path', type=str, default='data.pkl', help='Path to the data file')
#     parser.add_argument('--no_log', action='store_true', default=False, help='Disable wandb logging')
#     parser.add_argument('--time_in_hours', type=float, default=0.5, help='Time limit for PySR run in hours')
#     parser.add_argument('--niterations', type=int, default=10000, help='Number of iterations for PySR')
#     parser.add_argument('--max_size', type=int, default=50, help='Maximum size of equations in PySR')
#     parser.add_argument('--sr_residual', action='store_true', help='Use residual PySR run with previous model')
#     parser.add_argument('--previous_sr_path', type=str, default='sr_results/first_run.pkl', help='Path to previous PySR model')
#     parser.add_argument('--include_s', action='store_true', help='Include s as an input variable')
#     parser.add_argument('--flag', type=str, default='topk', choices=['all', 'top', 'topk'], help='Flag to select equations from previous run')
#     parser.add_argument('--selected_complexities', type=str, default='8', help='Comma-separated complexities to select for topk')
#     args = parser.parse_args()
#     return args

# if __name__ == '__main__':
#     args = parse_args()
#     config = get_config(args)
#     run_pysr(config)