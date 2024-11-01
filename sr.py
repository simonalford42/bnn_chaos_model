import subprocess
import wandb
import pysr
from pysr import PySRRegressor
import random

from matplotlib import pyplot as plt
import seaborn as sns
import os
sns.set_style('darkgrid')
import spock_reg_model
import numpy as np
import argparse
from einops import rearrange
import utils
import pickle
from utils import assert_equal
import einops
import torch
import pandas as pd
import math
from sklearn.metrics import mean_squared_error    

LL_LOSS = """
function elementwise_loss(prediction, target)

    function safe_log_erf(x)
        if x < -1
            0.485660082730562*x + 0.643278438654541*exp(x)
            + 0.00200084619923262*x^3 - 0.643250926022749
            - 0.955350621183745*x^2
        else
            log(1 + erf(x))
        end
    end

    mu = prediction

    if mu < 1 || mu > 14
        # The farther away from a reasonable range, the more we punish it
        return 100 * (mu - 7)^2
    end

    sigma = one(prediction)

    # equation 8 in Bayesian neural network paper
    log_like = if target >= 9
        safe_log_erf((mu - 9) / sqrt(2 * sigma^2))
    else
        (
            zero(prediction)
            - (target - mu)^2 / (2 * sigma^2)
            - log(sigma)
            - safe_log_erf((mu - 4) / sqrt(2 * sigma^2))
        )
    end

    return -log_like
end
"""


LABELS = ['time', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno', 'a1', 'e1', 'i1', 'cos_Omega1', 'sin_Omega1', 'cos_pomega1', 'sin_pomega1', 'cos_theta1', 'sin_theta1', 'a2', 'e2', 'i2', 'cos_Omega2', 'sin_Omega2', 'cos_pomega2', 'sin_pomega2', 'cos_theta2', 'sin_theta2', 'a3', 'e3', 'i3', 'cos_Omega3', 'sin_Omega3', 'cos_pomega3', 'sin_pomega3', 'cos_theta3', 'sin_theta3', 'm1', 'm2', 'm3', 'nan_mmr_near', 'nan_mmr_far', 'nan_megno']

LABEL_TO_IX = {label: i for i, label in enumerate(LABELS)}

def get_sr_included_ixs():
    ''' for running pysr to imitate f1, we only want to use the inputs that f1 uses '''
    # hard coded based off the default CL args passed
    skipped = ['nan_mmr_near', 'nan_mmr_far', 'nan_megno', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno']
    assert len(skipped) == 10

    included_ixs = [i for i in range(len(LABELS)) if LABELS[i] not in skipped]
    return included_ixs


INCLUDED_IXS = get_sr_included_ixs()
INPUT_VARIABLE_NAMES = [LABELS[ix] for ix in INCLUDED_IXS]


def load_inputs_and_targets(config):
    model = spock_reg_model.load(version=config['version'])
    model.make_dataloaders()
    model.eval()

    data_iterator = iter(model.train_dataloader())
    x, y = next(data_iterator)
    while x.shape[0] < config['n']:
        next_x, next_y = next(data_iterator)
        x = torch.cat([x, next_x], dim=0)
        y = torch.cat([y, next_y], dim=0)

    # we use noisy val bc it is used during training the NN too
    out_dict = model.forward(x, return_intermediates=True, noisy_val=False)

    if config['target'] == 'f1':
        # inputs to SR are the inputs to f1 neural network
        # we use this instead of x because the model zeros the unused inputs
        X = out_dict['inputs']  # [B, T, F]
        # targets for SR are the outputs of the f1 neural network
        y = out_dict['f1_output']  # [B, T, F]
        # f1 acts on timesteps independently, so we can just use different
        #  time steps as different possible samples
        X = rearrange(X, 'B T F -> (B T) F')
        y = rearrange(y, 'B T F -> (B T) F')

        # extract just the input variables we're actually using
        assert len(LABELS) == X.shape[1]
        X = X[..., INCLUDED_IXS]

        in_dim = len(INCLUDED_IXS)
        out_dim = model.hparams['latent']

        variable_names = INPUT_VARIABLE_NAMES
    
    #y=y
    elif config['target'] == 'f2':
        # inputs to SR are the inputs to f2 neural network
        X = out_dict['summary_stats']  # [B, 40]
        # outputs are the (mean, std) predictions of the nn
        y = y

        in_dim = model.summary_dim
        out_dim = 2

        n = X.shape[1] // 2
        variable_names = [f'm{i}' for i in range(n)] + [f's{i}' for i in range(n)]

        if config['sr_residual']:
            # 1. Load the previous PySR results, using highest complexity
            with open(config['previous_sr_path'], 'rb') as f:
                previous_sr_model = pickle.load(f)

            # 2. Calculate mean and std from previous results (previous run must have used target=='f2')
            if isinstance(out_dict['summary_stats'], torch.Tensor):
                summary_stats_np = out_dict['summary_stats'].detach().numpy()
            else:
                summary_stats_np = out_dict['summary_stats']

            flag = "topk"
            variable_name_map = {}

            if flag == "middle":
                # Get the middle complexity mean equation
                mean_equations = previous_sr_model.equations_[0].sort_values('complexity')
                middle_idx_mean = len(mean_equations) // 2

                mean_equation = mean_equations.iloc[middle_idx_mean]

                additional_features = []
                lambda_func = mean_equation['lambda_format']
                evaluated_result = lambda_func(summary_stats_np)
                evaluated_result = evaluated_result.reshape(-1, 1)
                additional_features.append(evaluated_result)
                # Map the variable name to the corresponding complexity and equation
                variable_name_map[f'prev0'] = (mean_equation['complexity'], mean_equation['equation'])

            elif flag == "top":
                # Get the highest complexity mean equation
                mean_equations = previous_sr_model.equations_[0]
                max_complexity_idx_mean = mean_equations['complexity'].idxmax()

                mean_equation = mean_equations.loc[max_complexity_idx_mean]

                additional_features = []
                lambda_func = mean_equation['lambda_format']
                evaluated_result = lambda_func(summary_stats_np)
                evaluated_result = evaluated_result.reshape(-1, 1)
                additional_features.append(evaluated_result)
                # Map the variable name to the corresponding complexity and equation
                variable_name_map[f'prev0'] = (mean_equation['complexity'], mean_equation['equation'])

            elif flag == "topk":
                selected_complexities = {8}
                selected_equations = []
                for equation in previous_sr_model.equations_[0].iterrows():
                    complexity = equation[1]['complexity']
                    if complexity in selected_complexities:
                        lambda_func = equation[1]['lambda_format']
                        selected_equations.append((lambda_func, complexity, equation[1]['equation']))

                # Evaluate the selected mean equations and add them as features
                additional_features = []
                for i, (lambda_func, complexity, equation_str) in enumerate(selected_equations):
                    evaluated_result = lambda_func(summary_stats_np)
                    evaluated_result = evaluated_result.reshape(-1, 1)
                    additional_features.append(evaluated_result)
                    # Map the variable name to the corresponding complexity and equation
                    variable_name_map[f'1prev{i}'] = (complexity, equation_str)

            elif flag == "all":
                # Entire pareto front mean evaluation (y=y)
                additional_features = []
                for index, equation in previous_sr_model.equations_[0].iterrows():
                    lambda_func = equation['lambda_format']
                    evaluated_result = lambda_func(summary_stats_np)
                    evaluated_result = evaluated_result.reshape(-1, 1)
                    additional_features.append(evaluated_result)
                    # Map the variable name to the corresponding complexity and equation
                    variable_name_map[f'prev{index}'] = (equation['complexity'], equation['equation'])

            # Convert list of arrays to a single numpy array
            additional_features = np.hstack(additional_features)
            # Concatenate the original summary stats with the evaluated results
            X = np.concatenate([additional_features, summary_stats_np], axis=1)
            in_dim = model.summary_dim + additional_features.shape[1]
            variable_names = [f'prev{i}' for i in range(additional_features.shape[1])] + variable_names

            # Print the variable names, associated complexities, and their equations
            print("Variable names, their associated complexities, and equations:")
            for variable_name, (complexity, equation_str) in variable_name_map.items():
                print(f"{variable_name}: complexity {complexity}, equation: {equation_str}")
    
    # #y=y
    # elif config['target'] == 'f2':
    #     # inputs to SR are the inputs to f2 neural network
    #     X = out_dict['summary_stats']  # [B, 40]
    #     # outputs are the (mean, std) predictions of the nn
    #     y = y

    #     in_dim = model.summary_dim
    #     out_dim = 2

    #     n = X.shape[1] // 2
    #     variable_names = [f'm{i}' for i in range(n)] + [f's{i}' for i in range(n)]

    #     if config['sr_residual']:
    #         # 1. Load the previous PySR results, using highest complexity
    #         with open(config['previous_sr_path'], 'rb') as f:
    #             previous_sr_model = pickle.load(f)

    #         # 2. Calculate mean and std from previous results (previous run must have used target=='f2')
    #         if isinstance(out_dict['summary_stats'], torch.Tensor):
    #             summary_stats_np = out_dict['summary_stats'].detach().numpy()
    #         else:
    #             summary_stats_np = out_dict['summary_stats']

    #         flag = "topk"
    #         variable_name_map = {}

    #         # Evaluate equations from the first PySR (direct) run
    #         additional_features = []
    #         if flag == "middle":
    #             # Get the middle complexity mean equation
    #             mean_equations = previous_sr_model.equations_[0].sort_values('complexity')
    #             middle_idx_mean = len(mean_equations) // 2

    #             mean_equation = mean_equations.iloc[middle_idx_mean]

    #             lambda_func = mean_equation['lambda_format']
    #             evaluated_result = lambda_func(summary_stats_np).reshape(-1, 1)
    #             additional_features.append(evaluated_result)
    #             variable_name_map[f'prev0'] = (mean_equation['complexity'], mean_equation['equation'])

    #         elif flag == "top":
    #             # Get the highest complexity mean equation
    #             mean_equations = previous_sr_model.equations_[0]
    #             max_complexity_idx_mean = mean_equations['complexity'].idxmax()

    #             mean_equation = mean_equations.loc[max_complexity_idx_mean]

    #             lambda_func = mean_equation['lambda_format']
    #             evaluated_result = lambda_func(summary_stats_np).reshape(-1, 1)
    #             additional_features.append(evaluated_result)
    #             variable_name_map[f'prev0'] = (mean_equation['complexity'], mean_equation['equation'])

    #         elif flag == "topk":
    #             selected_complexities = {8}  # Example complexities to select from direct run
    #             selected_equations = []
    #             for equation in previous_sr_model.equations_[0].iterrows():
    #                 complexity = equation[1]['complexity']
    #                 if complexity in selected_complexities:
    #                     lambda_func = equation[1]['lambda_format']
    #                     selected_equations.append((lambda_func, complexity, equation[1]['equation']))

    #             for i, (lambda_func, complexity, equation_str) in enumerate(selected_equations):
    #                 evaluated_result = lambda_func(summary_stats_np).reshape(-1, 1)
    #                 additional_features.append(evaluated_result)
    #                 variable_name_map[f'prev{i}'] = (complexity, equation_str)

    #         elif flag == "all":
    #             for index, equation in previous_sr_model.equations_[0].iterrows():
    #                 lambda_func = equation['lambda_format']
    #                 evaluated_result = lambda_func(summary_stats_np).reshape(-1, 1)
    #                 additional_features.append(evaluated_result)
    #                 variable_name_map[f'prev{index}'] = (equation['complexity'], equation['equation'])

    #         # If `previous_sr_path_2` is specified, repeat the process with it
    #         additional_features_2 = []
    #         if 'previous_sr_path_2' in config:
    #             with open(config['previous_sr_path_2'], 'rb') as f:
    #                 previous_sr_model_2 = pickle.load(f)

    #             # Evaluate equations from the first residual PySR run
    #             if flag == "middle":
    #                 mean_equations = previous_sr_model_2.equations_[0].sort_values('complexity')
    #                 middle_idx_mean = len(mean_equations) // 2

    #                 mean_equation = mean_equations.iloc[middle_idx_mean]

    #                 lambda_func = mean_equation['lambda_format']
    #                 evaluated_result = lambda_func(summary_stats_np).reshape(-1, 1)
    #                 additional_features_2.append(evaluated_result)
    #                 variable_name_map[f'1prev0'] = (mean_equation['complexity'], mean_equation['equation'])

    #             elif flag == "top":
    #                 mean_equations = previous_sr_model_2.equations_[0]
    #                 max_complexity_idx_mean = mean_equations['complexity'].idxmax()

    #                 mean_equation = mean_equations.loc[max_complexity_idx_mean]

    #                 lambda_func = mean_equation['lambda_format']
    #                 evaluated_result = lambda_func(summary_stats_np).reshape(-1, 1)
    #                 additional_features_2.append(evaluated_result)
    #                 variable_name_map[f'1prev0'] = (mean_equation['complexity'], mean_equation['equation'])

    #             elif flag == "topk":
    #                 topk2 = {14}  # Example set for topk2 from previous_sr_path_2
    #                 selected_equations = []
    #                 for equation in previous_sr_model_2.equations_[0].iterrows():
    #                     complexity = equation[1]['complexity']
    #                     if complexity in topk2:
    #                         lambda_func = equation[1]['lambda_format']
    #                         selected_equations.append((lambda_func, complexity, equation[1]['equation']))

    #                 for i, (lambda_func, complexity, equation_str) in enumerate(selected_equations):
    #                     evaluated_result = lambda_func(summary_stats_np).reshape(-1, 1)
    #                     additional_features_2.append(evaluated_result)
    #                     variable_name_map[f'1prev{i}'] = (complexity, equation_str)

    #             elif flag == "all":
    #                 for index, equation in previous_sr_model_2.equations_[0].iterrows():
    #                     lambda_func = equation['lambda_format']
    #                     evaluated_result = lambda_func(summary_stats_np).reshape(-1, 1)
    #                     additional_features_2.append(evaluated_result)
    #                     variable_name_map[f'1prev{index}'] = (equation['complexity'], equation['equation'])

    #         # Concatenate additional_features and additional_features_2 with summary_stats_np
    #         additional_features = np.hstack(additional_features)
    #         if additional_features_2:
    #             additional_features_2 = np.hstack(additional_features_2)
    #             X = np.concatenate([additional_features, additional_features_2, summary_stats_np], axis=1)
    #             in_dim = model.summary_dim + additional_features.shape[1] + additional_features_2.shape[1]
    #             variable_names = (
    #                 [f'prev{i}' for i in range(additional_features.shape[1])] +
    #                 [f'1prev{i}' for i in range(additional_features_2.shape[1])] +
    #                 variable_names
    #             )
    #         else:
    #             X = np.concatenate([additional_features, summary_stats_np], axis=1)
    #             in_dim = model.summary_dim + additional_features.shape[1]
    #             variable_names = [f'prev{i}' for i in range(additional_features.shape[1])] + variable_names

    #         # Print the variable names, associated complexities, and their equations
    #         print("Variable names, their associated complexities, and equations:")
    #         for variable_name, (complexity, equation_str) in variable_name_map.items():
    #             print(f"{variable_name}: complexity {complexity}, equation: {equation_str}")



    # # Continue for y = y - previous_prediction
    # elif config['target'] == 'f2':
    #     # inputs to SR are the inputs to f2 neural network
    #     X = out_dict['summary_stats']  # [B, 40]
    #     # outputs are the (mean, std) predictions of the nn
    #     y = y 

    #     in_dim = model.summary_dim
    #     out_dim = 2

    #     n = X.shape[1] // 2
    #     variable_names = [f'm{i}' for i in range(n)] + [f's{i}' for i in range(n)]

    #     if config['sr_residual']:
    #         # 1. load the previous pysr results, using highest complexity
    #         with open(config['previous_sr_path'], 'rb') as f:
    #             previous_sr_model = pickle.load(f)

    #         # 2. calculate mean and std from previous results (previous run must have used target=='f2')
    #         if isinstance(out_dict['summary_stats'], torch.Tensor):
    #             summary_stats_np = out_dict['summary_stats'].detach().numpy()
    #         else:
    #             summary_stats_np = out_dict['summary_stats']

    #         # get the highest complexity equation
    #         mean_equations = previous_sr_model.equations_[0]
    #         std_equations = previous_sr_model.equations_[1]
    #         max_complexity_idx_mean = mean_equations['complexity'].idxmax()
    #         max_complexity_idx_std = std_equations['complexity'].idxmax()

    #         mean_equation = mean_equations.loc[max_complexity_idx_mean]
    #         std_equation = std_equations.loc[max_complexity_idx_std]

    #         # This part is for y = y - previous_prediction
    #         # mean and std
    #         results = []
    #         for equation in [mean_equation, std_equation]:
    #             lambda_func = equation['lambda_format']
    #             evaluated_result = lambda_func(summary_stats_np)
    #             # Ensure the result is reshaped to match the batch size
    #             evaluated_result = evaluated_result.reshape(-1, 1)
    #             results.append(evaluated_result)

    #         # 3. concatenate mean and std to produce previous "prediction" of shape [B, 2]
    #         previous_prediction = np.hstack(results)
    #         assert previous_prediction.shape == (X.shape[0], 2)  # [B, 2]

    #         # 4. subtract previous prediction from y to get residual target
    #         if isinstance(y, torch.Tensor):
    #             y = y.detach().numpy()
    #         y = y - previous_prediction

    elif config['target'] == 'f2_ifthen':
        # inputs to SR are the inputs to f2 neural network
        X = out_dict['summary_stats']  # [B, 40]
        # target for SR is the predicates from the ifthen network
        y = out_dict['ifthen_preds']  # [B, 10]

        in_dim = model.summary_dim
        out_dim = 10

        n = X.shape[1] // 2
        variable_names = [f'm{i}' for i in range(n)] + [f's{i}' for i in range(n)]

    elif config['target'] == 'f2_direct':
        # inputs to SR are the inputs to f2 neural network
        X = out_dict['summary_stats']  # [B, 40]
        # target for SR is the ground truth mean, which we already have
        y = y # [B, 2]
        # there are two ground truth predictions. create a data point for each
        X = einops.repeat(X, 'B F -> (B two) F', two=2)
        y = einops.rearrange(y, 'B two -> (B two) 1')
        in_dim = model.summary_dim 
        out_dim = 1

        n = X.shape[1] // 2
        variable_names = [f'm{i}' for i in range(n)] + [f's{i}' for i in range(n)]

        if config['sr_residual']:
            # Load the PySR equations from the previous round
            with open(config['previous_sr_path'], 'rb') as f:
                previous_sr_model = pickle.load(f)

            # Evaluate the previous PySR equations on the inputs
            if isinstance(out_dict['summary_stats'], torch.Tensor):
                summary_stats_np = out_dict['summary_stats'].detach().numpy()
            else:
                summary_stats_np = out_dict['summary_stats']
            
            additional_features = []
            for equation_set in previous_sr_model.equations_:
                for index, equation in equation_set.iterrows():
                    lambda_func = equation['lambda_format']
                    evaluated_result = lambda_func(summary_stats_np)
                    # Ensure the result is reshaped to match the batch size
                    evaluated_result = evaluated_result.reshape(-1, 1)
                    additional_features.append(evaluated_result)

            # Convert list of arrays to a single numpy array
            additional_features = np.hstack(additional_features)
            # Concatenate the original summary stats with the evaluated results
            # summary_stats_np: [1, 2000, 40]
            # additional_features: (2000, 49)
            # (args.n, in_dim) = (250, 2*20 + additional_features dimension)
            X = np.concatenate([summary_stats_np, additional_features], axis=1)
            in_dim = model.summary_dim + additional_features.shape[1]
            variable_names += [f'm{i}' for i in range(additional_features.shape[1])]
    else:
        raise ValueError(f"Unknown target: {config['target']}")

    if config['residual']:
        assert config['target'] == 'f2_direct', 'residual requires a direct target'
        # target is the residual error of the model's prediction from the ground truth
        # because target was f2 direct, y shape is [B * 2, 1]
        # but predicted mean is [B, 1]
        # so repeat the predicted means
        y_old = out_dict['mean']
        assert y_old.shape[1] == 1
        y_old = einops.repeat(y_old, 'B one -> (B repeat) one', repeat=2)
        assert y_old.shape == y.shape
        y = y - y_old

    # go down from having a batch of size B to just N
    ixs = np.random.choice(X.shape[0], size=config['n'], replace=False)
    X, y = X[ixs], y[ixs]

    # Ensure X and y are NumPy arrays
    if isinstance(X, torch.Tensor):
        X = X.detach().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().numpy()

    assert X.shape == (config['n'], in_dim)
    assert y.shape == (config['n'], out_dim)

    return X, y, variable_names


def get_config(args):
    id = random.randint(0, 100000)
    while os.path.exists(f'sr_results/{id}.pkl'):
        id = random.randint(0, 100000)

    path = f'sr_results/{id}.pkl'
    # replace '.pkl' with '.csv'
    path = path[:-3] + 'csv'

    # https://stackoverflow.com/a/57474787/4383594
    num_cpus = int(os.environ.get('SLURM_CPUS_ON_NODE')) * int(os.environ.get('SLURM_JOB_NUM_NODES'))
    pysr_config = dict(
        procs=num_cpus,
        populations=3*num_cpus,
        batching=True,
        # cluster_manager='slurm',
        equation_file=path,
        niterations=args.niterations,
        # multithreading=False,
        binary_operators=["+", "*", '/', '-', '^'],
        unary_operators=['sin'],  # removed "log"
        maxsize=args.max_size,
        timeout_in_seconds=int(60*60*args.time_in_hours),
        # prevent ^ from using complex exponents, nesting power laws is expressive but uninterpretable
        # base can have any complexity, exponent can have max 1 complexity
        constraints={'^': (-1, 1)},
        nested_constraints={"sin": {"sin": 0}},
        ncyclesperiteration=1000, # increase utilization since usually using 32-ish cores?
    )

    if args.loss_fn == 'll':
        assert args.target == 'f2_direct', 'log likelihood loss only useful for f2_direct'
        pysr_config['elementwise_loss'] = LL_LOSS

    config = vars(args)
    config.update(pysr_config)
    config['pysr_config'] = pysr_config
    config.update({
        'id': id,
        'results_cmd': f'vim $(ls {path[:-4]}.csv*)',
        'slurm_id': os.environ.get('SLURM_JOB_ID', None),
        'slurm_name': os.environ.get('SLURM_JOB_NAME', None),
    })

    return config


# def get_config(args):
#     id = random.randint(0, 100000)
#     while os.path.exists(f'sr_results/{id}.pkl'):
#         id = random.randint(0, 100000)

#     path = f'sr_results/{id}.pkl'
#     # replace '.pkl' with '.csv'
#     path = path[:-3] + 'csv'

#     # https://stackoverflow.com/a/57474787/4383594
#     num_cpus = int(os.environ.get('SLURM_CPUS_ON_NODE')) * int(os.environ.get('SLURM_JOB_NUM_NODES'))
#     pysr_config = dict(
#         procs=num_cpus,
#         populations=3*num_cpus,
#         batching=True,
#         # cluster_manager='slurm',
#         equation_file=path,
#         niterations=args.niterations,
#         # multithreading=False,
#         binary_operators=["+", "*", '/', '-', '^'],
#         unary_operators=['sin'],  # removed "log"
#         maxsize=args.max_size,
#         timeout_in_seconds=int(60*60*args.time_in_hours),
#         # prevent ^ from using complex exponents, nesting power laws is expressive but uninterpretable
#         # base can have any complexity, exponent can have max 1 complexity
#         constraints={'^': (-1, 1)},
#         nested_constraints={"sin": {"sin": 0}},
#         ncyclesperiteration=1000, # increase utilization since usually using 32-ish cores?
#     )

#     model = spock_reg_model.load(version=config['version'])
#     model.make_dataloaders()
#     model.eval()

#     data_iterator = iter(model.train_dataloader())
#     x, y = next(data_iterator)
#     while x.shape[0] < config['n']:
#         next_x, next_y = next(data_iterator)
#         x = torch.cat([x, next_x], dim=0)
#         y = torch.cat([y, next_y], dim=0)
#     out_dict = model.forward(x, return_intermediates=True, noisy_val=False)
#     X = out_dict['summary_stats']  # [B, 40]
#     y = y 

#     if args.loss_fn == 'll':
#         assert args.target == 'f2_direct', 'log likelihood loss only useful for f2_direct'
#         pysr_config['elementwise_loss'] = LL_LOSS
#     else:
#         # Evaluate the original SR model
#         with open(config['previous_sr_path'], 'rb') as f:
#             previous_sr_model = pickle.load(f)

#         summary_stats_np = X if isinstance(X, np.ndarray) else X.detach().numpy()
        
#         mean_equations = previous_sr_model.equations_[0]
#         std_equations = previous_sr_model.equations_[1]
#         max_complexity_idx_mean = mean_equations['complexity'].idxmax()
#         max_complexity_idx_std = std_equations['complexity'].idxmax()

#         mean_equation = mean_equations.loc[max_complexity_idx_mean]
#         std_equation = std_equations.loc[max_complexity_idx_std]

#         results = []
#         for equation in [mean_equation, std_equation]:
#             lambda_func = equation['lambda_format']
#             evaluated_result = lambda_func(summary_stats_np)
#             # Ensure the result is reshaped to match the batch size
#             evaluated_result = evaluated_result.reshape(-1, 1)
#             results.append(evaluated_result)

#         previous_prediction = np.hstack(results)
#         assert previous_prediction.shape == (X.shape[0], 2)  # [B, 2]

#         # Evaluate the residual SR model
#         if 'residual_sr_path' in config:
#             with open(config['residual_sr_path'], 'rb') as f:
#                 residual_sr_model = pickle.load(f)
#         else:
#             residual_sr_model = None
#         if residual_sr_model:
#             residual_results = []
#             residual_mean_equations = residual_sr_model.equations_[0]
#             residual_std_equations = residual_sr_model.equations_[1]
#             max_complexity_idx_residual_mean = residual_mean_equations['complexity'].idxmax()
#             max_complexity_idx_residual_std = residual_std_equations['complexity'].idxmax()

#             residual_mean_equation = residual_mean_equations.loc[max_complexity_idx_residual_mean]
#             residual_std_equation = residual_std_equations.loc[max_complexity_idx_residual_std]

#             for equation in [residual_mean_equation, residual_std_equation]:
#                 lambda_func = equation['lambda_format']
#                 evaluated_result = lambda_func(summary_stats_np)
#                 evaluated_result = evaluated_result.reshape(-1, 1)
#                 residual_results.append(evaluated_result)

#             residual_prediction = np.hstack(residual_results)
#             assert residual_prediction.shape == (X.shape[0], 2)  # [B, 2]

#             # Combine original and residual predictions
#             combined_prediction = previous_prediction + residual_prediction
#             assert combined_prediction.shape == (X.shape[0], 2)  # [B, 2]

#             mse_error = mean_squared_error(y, combined_prediction)
#             print(f"Combined MSE Error: {mse_error}")
#         else:
#             # Calculate MSE with the original predictions
#             mse_error = mean_squared_error(y, previous_prediction)
#             print(f"Original MSE Error: {mse_error}")

#         pysr_config['mse'] = mse_error

#     config = vars(args)
#     config.update(pysr_config)
#     config['pysr_config'] = pysr_config
#     config.update({
#         'id': id,
#         'results_cmd': f'vim $(ls {path[:-4]}.csv*)',
#         'slurm_id': os.environ.get('SLURM_JOB_ID', None),
#         'slurm_name': os.environ.get('SLURM_JOB_NAME', None),
#     })

#     return config


def run_pysr(config):
    command = utils.get_script_execution_command()
    print(command)

    X, y, variable_names = load_inputs_and_targets(config)

    # Print out the shape of X to see the number of features (columns)
    print(f"Shape of X: {X.shape} (rows, columns)")
    num_features = X.shape[1]
    print(f"Number of features in X: {num_features}")

    model_kwargs = config['pysr_config']
    complexity_of_variables = None  # Default to None

    flag = "topk"

    # Check if it is an sr_residual run
    if config.get('sr_residual', False):
        # Load the previous PySR results
        with open(config['previous_sr_path'], 'rb') as f:
            previous_sr_model = pickle.load(f)

        # Only interested in the mean equations for all flags
        mean_equations = previous_sr_model.equations_[0]

        if flag == "top":
            # Get the highest complexity mean equation
            max_complexity_idx_mean = mean_equations['complexity'].idxmax()
            max_complexity = mean_equations.loc[max_complexity_idx_mean]['complexity']

            # Set complexity_of_variables to the max complexity of the selected equation followed by 1's
            complexity_of_variables = [max_complexity] + [1] * (num_features - 1)

            # Print the selected equation and complexity
            print(f"Top complexity selected: {max_complexity}")
            print(f"Complexity of variables for 'top': {complexity_of_variables}")

        elif flag == "topk":
            selected_complexities = {8}
            complexities_list = []

            for _, equation in mean_equations.iterrows():
                complexity = equation['complexity']
                if complexity in selected_complexities:
                    complexities_list.append(complexity)

            # Set complexity_of_variables to the complexities of the selected top k equations followed by 1's
            complexity_of_variables = complexities_list + [1] * (num_features - len(complexities_list))

            # Print the selected equations and complexities
            print(f"TopK complexities selected: {complexities_list}")
            print(f"Complexity of variables for 'topk': {complexity_of_variables}")

        elif flag == "all":
            # Use all equations for the residual
            complexities = mean_equations['complexity'].values
            complexity_of_variables = list(complexities[:X.shape[1]])  # Adjust the length accordingly

            # If the number of residual features is not equal to the number of features, append or adjust
            if len(complexity_of_variables) != num_features:
                print(f"Adjusting complexity list to match the number of features in X.")
                if len(complexity_of_variables) < num_features:
                    # If there are fewer complexities, fill in with default values (e.g., 1)
                    complexity_of_variables += [1] * (num_features - len(complexity_of_variables))
                elif len(complexity_of_variables) > num_features:
                    # If there are more complexities, trim the list
                    complexity_of_variables = complexity_of_variables[:num_features]

            # Print the adjusted complexities for debugging
            print(f"Complexity of variables for 'all': {complexity_of_variables}")
            print(f"Number of complexities: {len(complexity_of_variables)}")

    # Initialize PySRRegressor without complexity_of_variables in the __init__
    model = pysr.PySRRegressor(**model_kwargs)

    if not config['no_log']:
        wandb.init(
            entity='bnn-chaos-model',
            project='planets-sr',
            config=config,
        )

    # Fit the model, passing complexity_of_variables only if it exists
    if complexity_of_variables:
        print(f"Running PySR with custom complexities: {complexity_of_variables}")
        model.fit(X, y, variable_names=variable_names, complexity_of_variables=complexity_of_variables)
    else:
        print("Running PySR without custom complexities.")
        model.fit(X, y, variable_names=variable_names)

    print('Done running pysr')

    losses = [min(eqs['loss']) for eqs in model.equation_file_contents_]

    if not config['no_log']:
        wandb.log({'avg_loss': sum(losses)/len(losses),
                   'losses': losses,
                   })

    try:
        # delete julia files: julia-1911988-17110333239-0016.out
        subprocess.run(f'rm julia*.out', shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while trying to delete the backup files: {e}")

    print(f"Saved to path: {config['equation_file']}")


# def run_pysr(config):
#     command = utils.get_script_execution_command()
#     print(command)

#     X, y, variable_names = load_inputs_and_targets(config)

#     # Print out the shape of X to see the number of features (columns)
#     print(f"Shape of X: {X.shape} (rows, columns)")
#     num_features = X.shape[1]
#     print(f"Number of features in X: {num_features}")

#     model_kwargs = config['pysr_config']
#     complexity_of_variables = None  # Default to None

#     flag = "topk"

#     # Check if it is an sr_residual run
#     if config.get('sr_residual', False):
#         # Load the previous PySR results from the direct run
#         with open(config['previous_sr_path'], 'rb') as f:
#             previous_sr_model = pickle.load(f)

#         # Only interested in the mean equations for all flags
#         mean_equations = previous_sr_model.equations_[0]

#         # Initialize complexity list for the direct run (previous_sr_path)
#         complexities_list = []

#         if flag == "top":
#             # Get the highest complexity mean equation
#             max_complexity_idx_mean = mean_equations['complexity'].idxmax()
#             max_complexity = mean_equations.loc[max_complexity_idx_mean]['complexity']

#             complexities_list = [max_complexity]

#             # Print the selected equation and complexity
#             print(f"Top complexity selected (previous_sr_path): {max_complexity}")

#         elif flag == "topk":
#             selected_complexities = {8}  # Example complexities to select from direct run
#             for _, equation in mean_equations.iterrows():
#                 complexity = equation['complexity']
#                 if complexity in selected_complexities:
#                     complexities_list.append(complexity)

#             print(f"TopK complexities selected (previous_sr_path): {complexities_list}")

#         elif flag == "all":
#             complexities_list = mean_equations['complexity'].values.tolist()

#             # Adjust if the number of complexities is different from num_features
#             if len(complexities_list) != num_features:
#                 print(f"Adjusting complexity list to match the number of features in X.")
#                 if len(complexities_list) < num_features:
#                     complexities_list += [1] * (num_features - len(complexities_list))
#                 elif len(complexities_list) > num_features:
#                     complexities_list = complexities_list[:num_features]

#             print(f"All complexities selected (previous_sr_path): {complexities_list}")

#         # Initialize additional complexity list for the first residual run if previous_sr_path_2 is provided
#         complexities_list_2 = []
#         if 'previous_sr_path_2' in config:
#             with open(config['previous_sr_path_2'], 'rb') as f:
#                 previous_sr_model_2 = pickle.load(f)

#             mean_equations_2 = previous_sr_model_2.equations_[0]

#             if flag == "top":
#                 max_complexity_idx_mean_2 = mean_equations_2['complexity'].idxmax()
#                 max_complexity_2 = mean_equations_2.loc[max_complexity_idx_mean_2]['complexity']

#                 complexities_list_2 = [max_complexity_2]

#                 print(f"Top complexity selected (previous_sr_path_2): {max_complexity_2}")

#             elif flag == "topk":
#                 topk2 = {14}  # Example complexities to select from the first residual run
#                 for _, equation in mean_equations_2.iterrows():
#                     complexity = equation['complexity']
#                     if complexity in topk2:
#                         complexities_list_2.append(complexity)

#                 print(f"TopK complexities selected (previous_sr_path_2): {complexities_list_2}")

#             elif flag == "all":
#                 complexities_list_2 = mean_equations_2['complexity'].values.tolist()

#                 # Adjust if the number of complexities is different from num_features
#                 if len(complexities_list_2) != num_features:
#                     print(f"Adjusting complexity list to match the number of features in X.")
#                     if len(complexities_list_2) < num_features:
#                         complexities_list_2 += [1] * (num_features - len(complexities_list_2))
#                     elif len(complexities_list_2) > num_features:
#                         complexities_list_2 = complexities_list_2[:num_features]

#                 print(f"All complexities selected (previous_sr_path_2): {complexities_list_2}")

#         # Combine complexities from both runs
#         complexity_of_variables = complexities_list + complexities_list_2 + [1] * (num_features - len(complexities_list) - len(complexities_list_2))
#         print(f"Combined Complexity of variables: {complexity_of_variables}")

#     # Initialize PySRRegressor without complexity_of_variables in the __init__
#     model = pysr.PySRRegressor(**model_kwargs)

#     if not config['no_log']:
#         wandb.init(
#             entity='bnn-chaos-model',
#             project='planets-sr',
#             config=config,
#         )

#     # Fit the model, passing complexity_of_variables only if it exists
#     if complexity_of_variables:
#         print(f"Running PySR with custom complexities: {complexity_of_variables}")
#         model.fit(X, y, variable_names=variable_names, complexity_of_variables=complexity_of_variables)
#     else:
#         print("Running PySR without custom complexities.")
#         model.fit(X, y, variable_names=variable_names)

#     print('Done running pysr')

#     losses = [min(eqs['loss']) for eqs in model.equation_file_contents_]

#     if not config['no_log']:
#         wandb.log({'avg_loss': sum(losses)/len(losses),
#                    'losses': losses,
#                    })

#     try:
#         # delete julia files: julia-1911988-17110333239-0016.out
#         subprocess.run(f'rm julia*.out', shell=True, check=True)
#     except subprocess.CalledProcessError as e:
#         print(f"An error occurred while trying to delete the backup files: {e}")

#     print(f"Saved to path: {config['equation_file']}")



def spock_features(X):
    features = ['a1', 'a2', 'a3']

    def x(f):
        return X[:, LABEL_TO_IX[f]]

    def e_cross_inner(x):
        return (x('a2') - x('a1')) / x('a1')

    def e_cross_outer(x):
        return (x('a3') - x('a2')) / x('a2')

    y = [e_cross_inner(x), e_cross_outer(x)]
    y = einops.rearrange(y, 'n B -> B n')
    return y


def parse_args():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')
    # when importing from jupyter nb, it passes an arg to --f which we should just ignore
    parser.add_argument('--no_log', action='store_true', default=False, help='disable wandb logging')
    parser.add_argument('--version', type=int, help='')

    parser.add_argument('--time_in_hours', type=float, default=1)
    parser.add_argument('--niterations', type=float, default=500000) # by default, use time in hours as limit
    parser.add_argument('--max_size', type=int, default=60)
    parser.add_argument('--target', type=str, default='f2_direct', choices=['f1', 'f2', 'f2_ifthen', 'f2_direct', 'f2_2'])
    parser.add_argument('--residual', action='store_true', help='do residual training of your target')
    parser.add_argument('--n', type=int, default=10000, help='number of data points for the SR problem')
    parser.add_argument('--batch_size', type=int, default=500, help='number of data points for the SR problem')
    parser.add_argument('--sr_residual', action='store_true', help='do residual training of your target with previous sr run as base')
    parser.add_argument('--loss_fn', type=str, choices=['mse', 'll'], help='choose "ll" to use loglikelidhood loss')
    parser.add_argument('--previous_sr_path', type=str, default='sr_results/16506.pkl')
    parser.add_argument('--previous_sr_path_2', type=str, default='sr_results/86792.pkl')

    args = parser.parse_args()
    return args


def load_results(id):
    path = 'sr_results/id.pkl'
    results: PySRRegressor = pickle.load(open(path, 'rb'))
    return results


def plot_pareto(path):
    results = pickle.load(open(path, 'rb'))
    results = results.equations_[0]
    x = results['complexity']
    y = results['loss']
    # plot the pareto frontier
    plt.scatter(x, y)
    plt.xlabel('complexity')
    plt.ylabel('loss')
    plt.title('pareto frontier for' + path)
    # save the plot
    plt.savefig('pareto.png')


if __name__ == '__main__':
    args = parse_args()
    config = get_config(args)
    run_pysr(config)




