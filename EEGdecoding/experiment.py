import numpy as np
import torch
import os
import wandb

import argparse
from datetime import datetime
import random
import sys
from braindecode.models import ShallowFBCSPNet

from EEGdecoding.models.fpt import FPT
from EEGdecoding.trainer import Trainer


def experiment(
        exp_name,
        exp_args,
        **kwargs
):

    """
    Preliminary checks
    """

    # Must be able to accumulate gradient if batch size is large
    assert 'batch_size' in kwargs
    assert kwargs['batch_size'] <= exp_args['gpu_batch_size'] or \
           kwargs['batch_size'] % exp_args['gpu_batch_size'] == 0

    # Specific threads when running on HPC (https://hpc.vub.be/docs/software/usecases/#pytorch)
    if exp_args['cluster']:
        torch.set_num_threads(len(os.sched_getaffinity(0)))
        torch.set_num_interop_threads(1)


    """
    Set seeds for reproducibility
    """
    seed = kwargs['seed']
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    """
    Create dataset, model, and trainer
    """

    task = kwargs['task']
    batch_size = kwargs['batch_size']
    patch_size = kwargs['patch_size']
    device = exp_args['device']
    model_type = kwargs['model_type']

    return_last_only = True

    if task == 'mnist':
        from EEGdecoding.datasets.mnist import MNISTDataset
        dataset = MNISTDataset(batch_size=batch_size, patch_size=patch_size, device=device)
        dataset.get_batch(batch_size)
        input_dim, output_dim = patch_size ** 2, 10
        use_embeddings = False
    elif task == 'BCI_Competition_IV_2a':
        from EEGdecoding.datasets.EEGDataset import EEGDataset
        dataset = EEGDataset(batch_size=batch_size, seed=seed, patch_size=patch_size, device=device)
        input_dim, output_dim = patch_size, 4
        use_embeddings = False
        
    else:
        raise NotImplementedError('dataset not implemented')

    # Metrics
    ce_loss = torch.nn.CrossEntropyLoss()

    def loss_fn(out, y, x=None):
        out = out[:, 0]
        return ce_loss(out, y)

    def accuracy_fn(preds, true, x=None):
        preds = preds[:, 0].argmax(-1)
        return (preds == true).mean()

    # Model
    if model_type == "EEG":
        model = ShallowFBCSPNet(
                    dataset.n_channels,
                    output_dim,
                    input_window_samples=dataset.input_window_samples,
                    final_conv_length='auto',
                )
    elif model_type == "FPT":
        model = FPT(
            input_dim=input_dim,
            output_dim=output_dim,
            model_name=kwargs.get('model_name', 'gpt2'),
            pretrained=kwargs.get('pretrained', True),
            return_last_only=return_last_only,
            use_embeddings_for_in=use_embeddings,
            in_layer_sizes=kwargs.get('in_layer_sizes', None),
            out_layer_sizes=kwargs.get('out_layer_sizes', None),
            freeze_trans=kwargs.get('freeze_trans', True),
            freeze_in=kwargs.get('freeze_in', False),
            freeze_pos=kwargs.get('freeze_pos', False),
            freeze_ln=kwargs.get('freeze_ln', False),
            freeze_attn=kwargs.get('freeze_attn', True),
            freeze_ff=kwargs.get('freeze_ff', True),
            freeze_out=kwargs.get('freeze_out', False),
            dropout=kwargs['dropout'],
            orth_gain=kwargs['orth_gain'],
        )
    else:
        raise NotImplementedError('model type not implemented')
    model.to(device)

    # Trainer
    gpu_batch_size = exp_args['gpu_batch_size']
    trainer = Trainer(
        model,
        model_type,
        dataset,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        steps_per_epoch=exp_args['steps_per_iter'],
        test_steps_per_epoch=exp_args['test_steps_per_iter'],
        learning_rate=kwargs['learning_rate'],
        batch_size=gpu_batch_size if batch_size > gpu_batch_size else batch_size,
        eval_batch_size=batch_size,
        grad_accumulate=batch_size // gpu_batch_size if batch_size > gpu_batch_size else 1,
    )

    """
    Set up logging
    """

    log_to_wandb = exp_args['log_to_wandb']
    save_models = exp_args['save_models']
    wandb_project = exp_args['wandb_project']

    short_name = str(random.randint(int(1e5), int(1e6) - 1))
    run_name = f'{exp_name}-{task}-{short_name}'

    if log_to_wandb:
        config = dict(
            short_name=short_name,
            run_name=run_name,
            **exp_args,
            **kwargs,
        )
        wandb.init(
            name=f'{exp_name}-{short_name}',
            group=f'{exp_name}-{task}',
            project=wandb_project,
            config=config,
        )
        wandb.watch(model)

    for t in range(exp_args['num_iters']):
        trainer.train_epoch()

        print('=' * 57)
        print(f'| Iteration {" " * 15} | {t+1:25} |')
        for k, v in trainer.diagnostics.items():
            print(f'| {k:25} | {v:25} |')

        if log_to_wandb:
            wandb.log(trainer.diagnostics)

        if save_models and ((t+1) % exp_args['save_models_every'] == 0 or
                            (t+1) == exp_args['num_iters']):
            with open(f'models/{run_name}.pt', 'wb') as f:
                state_dict = dict(model=model.state_dict(), optim=trainer.optim.state_dict())
                torch.save(state_dict, f)
            print(f'Saved model at {t+1} iters: {run_name}')


def run_experiment(
        exp_name,
        experiment_params,
):
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_iters', '-it', type=int, default=4,
                        help='Number of iterations for trainer')
    parser.add_argument('--steps_per_iter', type=int, default=100,
                        help='Number of gradient steps per iteration')
    parser.add_argument('--test_steps_per_iter', type=int, default=25,
                        help='Number of test gradient steps per iteration')

    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False,
                        help='Whether or not to log to Weights and Biases')
    parser.add_argument('--note', '-n', type=str, default='',
                        help='An optional note to be logged to W&B')
    parser.add_argument('--wandb_project', type=str, default='fpt-for-eeg',
                        help='Project name for W&B')
    parser.add_argument('--include_date', type=bool, default=True,
                        help='Whether to include date in run name')

    parser.add_argument('--save_models', '-s', type=bool, default=False,
                        help='Whether or not to save the model files locally')
    parser.add_argument('--save_models_every', '-int', type=int, default=25,
                        help='How often to save models locally')

    parser.add_argument('--cluster', '-c', type=bool, default=False,
                        help='Whether or not the experiment is ran on the HPC cluster')
    parser.add_argument('--device', '-d', type=str, default='cuda',
                        help='Which device for Pytorch to use')
    parser.add_argument('--gpu_batch_size', '-gbs', type=int, default=16,
                        help='Max batch size to put on GPU (used for gradient accumulation)')

    exp_args = parser.parse_args(sys.argv[1:])

    if exp_args.include_date:
        timestamp = datetime.now().strftime('%m-%d')
        exp_name = f'{timestamp}-{exp_name}'

    experiment_params['exp_name'] = exp_name
    experiment_params['exp_args'] = vars(exp_args)

    experiment(xp_name=exp_name, **experiment_params)