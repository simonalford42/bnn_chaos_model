"""This file trains a model to minima, then saves it for run_swag.py"""
import pysr
import seaborn as sns
sns.set_style('darkgrid')
import spock_reg_model
spock_reg_model.HACK_MODEL = True
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import numpy as np
import pickle
from scipy.stats import truncnorm
import sys
from parse_swag_args import parse
import utils
from modules import mlp
import os

rand = lambda lo, hi: np.random.rand()*(hi-lo) + lo
irand = lambda lo, hi: int(np.random.rand()*(hi-lo) + lo)
log_rand = lambda lo, hi: 10**rand(np.log10(lo), np.log10(hi))
ilog_rand = lambda lo, hi: int(10**rand(np.log10(lo), np.log10(hi)))

parse_args = parse()
checkpoint_filename = utils.ckpt_path(parse_args.version, parse_args.seed)

# Fixed hyperparams:
TOTAL_STEPS = parse_args.total_steps
if TOTAL_STEPS == 0:
    import sys
    sys.exit(0)

TRAIN_LEN = 78660
batch_size = 2000 #ilog_rand(32, 3200)
steps_per_epoch = int(1+TRAIN_LEN/batch_size)
epochs = int(1+TOTAL_STEPS/steps_per_epoch)
if parse_args.eval:
    epochs = 1

command = utils.get_script_execution_command()
print(command)
print(f'Training for {epochs} epochs')

args = {
    'seed': parse_args.seed,
    'batch_size': batch_size,
    'f1_depth': 1,
    'swa_lr': parse_args.lr/2,
    'f2_depth': parse_args.f2_depth,
    'samp': 5,
    'swa_start': epochs//2,
    'weight_decay': 1e-14,
    'to_samp': 1,
    'epochs': epochs,
    'scheduler': True,
    'scheduler_choice': 'swa',
    'steps': TOTAL_STEPS,
    'beta_in': 1e-5,
    'beta_out': 0.001,
    'act': 'softplus',
    'noisy_val': False,
    'gradient_clip': 0.1,
    # Much of these settings turn off other parameters tried:
    'fix_megno': False, #avg,std of megno
    'fix_megno2': True, #Throw out megno completely
    'include_angles': True,
    'include_mmr': False,
    'include_nan': False,
    'include_eplusminus': False,
    'power_transform': False,
    # moving some parse args to here to clean up
    'plot': False,
    'plot_random': False,
    'train_all': False,
    'lower_std': False,
    'slurm_id': os.environ.get('SLURM_JOB_ID', None),
    'slurm_name': os.environ.get('SLURM_JOB_NAME', None),
}

# by default, parsed args get sent as hparams
for k, v in vars(parse_args).items():
    args[k] = v

name = 'full_swag_pre_' + checkpoint_filename
# logger = TensorBoardLogger("tb_logs", name=name)
logger = WandbLogger(project='bnn-chaos-model', entity='bnn-chaos-model', name=name, mode='disabled' if args['no_log'] else 'online')
checkpointer = ModelCheckpoint(filepath=checkpoint_filename + '/{version}')

model = spock_reg_model.VarModel(args)
model.make_dataloaders()

max_l2_norm = args['gradient_clip']*sum(p.numel() for p in model.parameters() if p.requires_grad)
trainer = Trainer(
    gpus=1, num_nodes=1, max_epochs=args['epochs'],
    logger=logger if not args['no_log'] else False,
    checkpoint_callback=checkpointer, benchmark=True,
    terminate_on_nan=True, gradient_clip_val=max_l2_norm,
)

# torch.autograd.set_detect_anomaly(True)
# do this early too so they show up while training, or if the run crashes before finishing
logger.log_hyperparams(params=args)

if args['calc_scores']:
    # print the model equations
    print(model.regress_nn)

    # just do this, don't train
    import main_figures
    main_figures.calc_scores_nonswag(model, logger, plot_random=args['plot_random'], train_all=args['train_all'])
    logger.save()
    logger.finalize('success')
    import sys
    sys.exit(0)

try:
    trainer.fit(model)
except ValueError as e:
    print('Error while training')
    print(e)
    model.load_state_dict(torch.load(checkpointer.best_model_path)['state_dict'])

logger.log_hyperparams(params=model.hparams)
logger.log_metrics({'val_loss': checkpointer.best_model_score.item()})
# in case we load the model, we want to override the hparams from the model with the args actually passed in
logger.log_hyperparams(params=args)

logger.experiment.config['val_loss'] = checkpointer.best_model_score.item()

if not ('no_plot' in args and args['no_plot']):
    import main_figures
    main_figures.calc_scores_nonswag(model, logger=logger, plot_random=args['plot_random'], train_all=args['train_all'])

logger.save()
logger.finalize('success')

logger.save()

model.load_state_dict(torch.load(checkpointer.best_model_path)['state_dict'])
model.make_dataloaders()

# loading models with pt lightning sometimes doesnt work, so lets also save the feature_nn and regress_nn directly
try:
    torch.save(model.feature_nn, f'models/{args["version"]}_feature_nn.pt')
    torch.save(model.regress_nn, f'models/{args["version"]}_regress_nn.pt')
except Exception as e:
    print('Failed to save feature_nn and regress_nn')
    print(e)

if args['f2_variant'] == 'pysr_residual':
    torch.save(model.regress_nn.module1, f'models/{args["version"]}_pysr_nn.pt')
    torch.save(model.regress_nn.module2, f'models/{args["version"]}_base_nn.pt')


print('Finished running')
