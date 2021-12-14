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
from src.datasets.PhysioNetDataset import PhysioNetDataset
from src.datasets.CompetitionDataset import CompetitionDataset

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
from sklearn.model_selection import KFold


def experiment(exp_name, exp_args, **kwargs):

    # Extract decision bools
    cluster = exp_args["cluster"]
    optimise = kwargs["optimise"]
    log_to_wandb = exp_args["log_to_wandb"]

    # Cluster specific things
    if cluster:

        # Specific threads when running on HPC (https://hpc.vub.be/docs/software/usecases/#pytorch)
        torch.set_num_threads(len(os.sched_getaffinity(0)))
        torch.set_num_interop_threads(1)

    # Random id for experiment
    experiment_id = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))

    """
    Set seed for reproducibility
    """
    seed = kwargs["seed"]
    torch.manual_seed(seed)

    """
    Training function
    """

    task = kwargs["task"]
    hyperparams = kwargs["hyperparams"]
    window_size = kwargs["window_size"]
    model_type = kwargs["model_type"]
    folds = kwargs["folds"]

    # Device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

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

        # Must be able to accumulate gradient if batch size is large
        assert "batch_size" in hyperparams
        batch_size = hyperparams["batch_size"]
        assert (
            batch_size <= exp_args["gpu_batch_size"]
            or batch_size % exp_args["gpu_batch_size"] == 0
        )

        # Dataset
        if task == "BCI_Competition_IV_2a":
            dataset = CompetitionDataset(
                task=task,
                batch_size=batch_size,
                seed=seed,
                window_size=window_size,
                device=device,
                model_type=model_type,
            )
        elif task == "mnist":
            from src.datasets.mnist import MNISTDataset

            dataset = MNISTDataset(
                task=task,
                batch_size=batch_size,
                seed=seed,
                window_size=window_size,
                device=device,
                model_type=model_type,
            )
        else:
            raise NotImplementedError("dataset not implemented")

        # Dimensions
        output_dim = dataset.classes
        if model_type == "CNN":
            input_dim = window_size
        elif model_type == "FPT":
            input_dim = dataset.n_channels
        else:
            raise NotImplementedError("model type not implemented")

        # Trainer
        gpu_batch_size = exp_args["gpu_batch_size"]
        trainer = Trainer(
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

        # Wandb parameters
        if log_to_wandb:
            kwargs[
                "hyperparams"
            ] = hyperparams  # this is needed to actually initialise the optimisation choices
            config = dict(
                **exp_args,
                **kwargs,
            )
            group_name = f"{exp_name}-{task}-{model_type}-{experiment_id}"
            if optimise:
                group_name += "-optimisation"
                hyperpars_configuration_id = "".join(
                    random.choices(string.ascii_uppercase + string.digits, k=6)
                )
            else:
                group_name += "-crossval"

        # Define cross-validation folds
        kfold = KFold(n_splits=folds, shuffle=True)

        # For each fold
        avg_test_loss = 0
        avg_accuracy = 0
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset.windows)):

            # Model
            if model_type == "CNN":
                model = ShallowFBCSPNet(
                    dataset.n_channels,
                    output_dim,
                    input_window_samples=input_dim,
                    final_conv_length="auto",
                )
            elif model_type == "FPT":
                model = FPT(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    model_name=kwargs.get("model_name", "gpt2"),
                    pretrained=kwargs.get("pretrained", True),
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

            # Send model to device
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            model.to(device)

            # Send model to trainer
            trainer.set_model(model)

            # Set data loaders specifically for this fold
            dataset.set_loaders(train_ids, test_ids)

            # Init a wandb run for this fold
            if log_to_wandb:
                run = wandb.init(
                    name="fold" + str(fold + 1),
                    group=group_name,
                    project=exp_args["wandb_project"],
                    config=config,
                    job_type=hyperpars_configuration_id if optimise else "eval",
                    reinit=True,
                )
                wandb.watch(model)

            # Train
            for iter in range(exp_args["num_iters"]):

                # Train single epoch
                iteration_test_loss, iteration_accuracy = trainer.train_epoch(iter)
                avg_test_loss += iteration_test_loss / folds
                avg_accuracy += iteration_accuracy / folds

                # Log to wandb
                if log_to_wandb:
                    wandb.log(trainer.diagnostics)

                # Log to terminal
                if logging_fn is not None:
                    logging_fn(iter, trainer)

            # End the wandb run
            if log_to_wandb:
                run.finish()

        # Remove the model from memory
        del model

        # Log averages to tune
        if optimise:
            tune.report(loss=avg_test_loss, accuracy=avg_accuracy)

    """
    Training
    """

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

        # Terminal logging
        best_trial = result.get_best_trial("loss", "min", "last")
        if result is not None and best_trial is not None:
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

        # Terminal logging
        def logging_fn(iter, trainer):
            print("=" * 57)
            print(f'| Iteration {" " * 15} | {iter+1:25} |')
            for k, v in trainer.diagnostics.items():
                print(f"| {k:25} | {v:25} |")
            print("=" * 57)

        # Train
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
        default=4,
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
        action="store_true",
        default=False,
        help="Whether to include date in run name",
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
