import numpy as np
import torch
import os
import wandb

import argparse
from datetime import datetime
import random
import sys
import string
from braindecode.models import ShallowFBCSPNet

from src.models.fpt import FPT
from src.trainer import Trainer
from src.datasets.CNNDataset import CNNDataset
from src.datasets.FPTDataset import FPTDataset

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial


def experiment(exp_name, exp_args, **kwargs):

    # Extract decision bools
    cluster = exp_args["cluster"]
    optimise = kwargs["optimise"]
    log_to_wandb = exp_args["log_to_wandb"]
    save_models = exp_args["save_models"]

    # Cluster specific things
    if cluster:

        # Specific threads when running on HPC (https://hpc.vub.be/docs/software/usecases/#pytorch)
        torch.set_num_threads(len(os.sched_getaffinity(0)))
        torch.set_num_interop_threads(1)

        # Data dirs
        data_dir = os.path.join(os.environ["VSC_DATA"], "data")
        model_dir = os.path.join(os.environ["VSC_DATA"], "models")
    else:
        data_dir = os.path.abspath("./data")
        model_dir = os.path.abspath("./models")

    """
    Set seeds for reproducibility
    """
    seed = kwargs["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    """
    Training function
    """

    task = kwargs["task"]
    hyperparams = kwargs["hyperparams"]
    window_size = kwargs["window_size"]
    model_type = kwargs["model_type"]

    return_last_only = True

    # Metrics
    ce_loss = torch.nn.CrossEntropyLoss()

    def loss_fn(out, y, x=None):
        out = out[:, 0]
        return ce_loss(out, y)

    def accuracy_fn(preds, true, x=None):
        preds = preds[:, 0].argmax(-1)
        return (preds == true).mean()

    # Function
    def train_fn(hyperparams, logging_fn=None, log_to_wandb=False):
        rid = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))

        # Must be able to accumulate gradient if batch size is large
        assert "batch_size" in hyperparams
        batch_size = hyperparams["batch_size"]
        assert (
            batch_size <= exp_args["gpu_batch_size"]
            or batch_size % exp_args["gpu_batch_size"] == 0
        )

        # Device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"

        # Dataset

        # Model and dataset
        if model_type == "CNN":

            # Dataset
            dataset = CNNDataset(
                task=task,
                batch_size=batch_size,
                seed=seed,
                window_size=window_size,
                device=device,
                data_dir=data_dir,
            )
            input_dim, output_dim = window_size, dataset.classes

            # Model
            model = ShallowFBCSPNet(
                dataset.n_channels,
                output_dim,
                input_window_samples=input_dim,
                final_conv_length="auto",
            )

        elif model_type == "FPT":

            # Dataset
            dataset = FPTDataset(
                task=task,
                batch_size=batch_size,
                seed=seed,
                window_size=window_size,
                device=device,
                data_dir=data_dir,
            )
            input_dim, output_dim = dataset.n_channels, dataset.classes

            # Model
            model = FPT(
                input_dim=input_dim,
                output_dim=output_dim,
                model_name=kwargs.get("model_name", "gpt2"),
                pretrained=kwargs.get("pretrained", True),
                return_last_only=return_last_only,
                use_embeddings_for_in=False,
                in_layer_sizes=kwargs.get("in_layer_sizes", None),
                out_layer_sizes=kwargs.get("out_layer_sizes", None),
                freeze_trans=kwargs.get("freeze_trans", True),
                freeze_in=kwargs.get("freeze_in", False),
                freeze_pos=kwargs.get("freeze_pos", False),
                freeze_ln=kwargs.get("freeze_ln", False),
                freeze_attn=kwargs.get("freeze_attn", True),
                freeze_ff=kwargs.get("freeze_ff", True),
                freeze_out=kwargs.get("freeze_out", False),
                dropout=hyperparams["dropout"],
                orth_gain=hyperparams["orth_gain"],
            )
        else:
            raise NotImplementedError("model type not implemented")

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.to(device)

        if log_to_wandb:
            kwargs["hyperparams"] = hyperparams
            config = dict(
                **exp_args,
                **kwargs,
            )
            if optimise:
                group_name = f"{exp_name}-{task}-{model_type}-optimisation"
                model_name = f"{rid}"
                wandb.init(
                    name=model_name,
                    group=group_name,
                    project=wandb_project,
                    config=config,
                )
            else:
                group_name = f"{exp_name}-{task}"
                model_name = f"{model_type}-{rid}"
                wandb.init(
                    name=model_name,
                    group=group_name,
                    project=wandb_project,
                    config=config,
                )
            wandb.watch(model)

        # Trainer
        gpu_batch_size = exp_args["gpu_batch_size"]
        trainer = Trainer(
            model,
            model_type,
            dataset,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            steps_per_epoch=exp_args["steps_per_iter"],
            test_steps_per_epoch=exp_args["test_steps_per_iter"],
            learning_rate=hyperparams["learning_rate"],
            batch_size=gpu_batch_size if batch_size > gpu_batch_size else batch_size,
            eval_batch_size=batch_size,
            grad_accumulate=batch_size // gpu_batch_size
            if batch_size > gpu_batch_size
            else 1,
        )

        # Training
        for iter in range(exp_args["num_iters"]):

            # Train single epoch
            test_loss, accuracy = trainer.train_epoch(iter)

            # Log to wandb
            if log_to_wandb:
                wandb.log(trainer.diagnostics)

            # Log to terminal
            if logging_fn is not None:
                logging_fn(iter, model, trainer, rid)

            # Log to tune
            if optimise:
                tune.report(loss=test_loss, accuracy=accuracy)

    """
    Training
    """
    wandb_project = exp_args["wandb_project"]

    if optimise:

        # Tune scheduler
        scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=10,
            grace_period=1,
            reduction_factor=2,
        )

        # Tune reporter
        reporter = CLIReporter(
            metric_columns=["loss", "accuracy", "training_iteration"]
        )

        # Optimisation
        result = tune.run(
            partial(train_fn, log_to_wandb=log_to_wandb),
            resources_per_trial={"cpu": 1, "gpu": 0},
            config=hyperparams,
            num_samples=10,
            scheduler=scheduler,
            progress_reporter=reporter,
            local_dir=os.path.join(model_dir, "optimisation"),
        )

        # Print and save results
        best_trial = result.get_best_trial("loss", "min", "last")
        if result.config is not None:
            print("Best trial config: {}".format(best_trial.config))
            print(
                "Best trial final validation loss: {}".format(
                    best_trial.last_result["loss"]
                )
            )
            print(
                "Best trial final validation accuracy: {}".format(
                    best_trial.last_result["accuracy"]
                )
            )

    else:

        def logging_fn(iter, model, trainer, rid):
            print("=" * 57)
            print(f'| Iteration {" " * 15} | {iter+1:25} |')
            for k, v in trainer.diagnostics.items():
                print(f"| {k:25} | {v:25} |")
            print("=" * 57)

            if save_models and (
                (iter + 1) % exp_args["save_models_every"] == 0
                or (iter + 1) == exp_args["num_iters"]
            ):
                with open(
                    os.path.join(model_dir, f"{exp_name}-{task}-{model_type}-{rid}.pt"),
                    "wb",
                ) as f:
                    state_dict = dict(
                        model=model.state_dict(), optim=trainer.optim.state_dict()
                    )
                    torch.save(state_dict, f)
                print(
                    f"Saved model at {iter+1} iters: {exp_name}-{task}-{model_type}-{rid}"
                )

        # Experiment
        train_fn(hyperparams, logging_fn=logging_fn, log_to_wandb=log_to_wandb)


def run_experiment(
    exp_name,
    experiment_params,
):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_iters",
        "-it",
        type=int,
        default=20,
        help="Number of iterations for trainer",
    )
    parser.add_argument(
        "--steps_per_iter",
        type=int,
        default=100,
        help="Number of gradient steps per iteration",
    )
    parser.add_argument(
        "--test_steps_per_iter",
        type=int,
        default=25,
        help="Number of test gradient steps per iteration",
    )
    parser.add_argument(
        "--log_to_wandb",
        "-w",
        action="store_true",
        default=False,
        help="Whether or not to log to Weights and Biases",
    )
    parser.add_argument(
        "--note",
        "-n",
        type=str,
        default="",
        help="An optional note to be logged to W&B",
    )
    parser.add_argument(
        "--wandb_project", type=str, default="fpt-for-eeg", help="Project name for W&B"
    )
    parser.add_argument(
        "--include_date",
        action="store_false",
        default=True,
        help="Whether to include date in run name",
    )
    parser.add_argument(
        "--save_models",
        "-s",
        action="store_true",
        default=False,
        help="Whether or not to save the model files locally",
    )
    parser.add_argument(
        "--save_models_every",
        "-int",
        type=int,
        default=25,
        help="How often to save models locally",
    )
    parser.add_argument(
        "--cluster",
        "-c",
        action="store_true",
        default=False,
        help="Whether or not the experiment is ran on the HPC cluster",
    )
    parser.add_argument(
        "--gpu_batch_size",
        "-gbs",
        type=int,
        default=16,
        help="Max batch size to put on GPU (used for gradient accumulation)",
    )

    exp_args = parser.parse_args(sys.argv[1:])

    if exp_args.include_date:
        timestamp = datetime.now().strftime("%m-%d")
        exp_name = f"{timestamp}-{exp_name}"

    experiment_params["exp_name"] = exp_name
    experiment_params["exp_args"] = vars(exp_args)

    experiment(xp_name=exp_name, **experiment_params)
