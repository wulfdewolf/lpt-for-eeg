import torch
import argparse
import time
import random
import string
import wandb
import mne
import os
import numpy
import ray
import hyperopt
import multiprocessing
import tqdm
import functools
import copy

import dataset
import models

mne.set_log_level(False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tunes GPT2 on EEG data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log to WandB.",
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Whether or not cluster-specific settings apply.",
    )
    parser.add_argument(
        "--optimise",
        default=None,
        type=int,
        help="Whether or not to optimise and how much random search tries should be executed.",
    )
    parser.add_argument(
        "--name",
        default="thesis",
        help="Name of experiment.",
    )
    # Pretraining
    parser.add_argument(
        "--pretrained-transformer",
        action="store_true",
        help="Whether or not to use a pretrained version of GPT2.",
    )
    # Freezing
    parser.add_argument(
        "--freeze-pos",
        action="store_true",
        help="Whether or not to freeze GPT2's positional embedding.",
    )
    parser.add_argument(
        "--freeze-ln",
        action="store_true",
        help="Whether or not to freeze GPT2's layer-norm layers.",
    )
    parser.add_argument(
        "--freeze-attn",
        action="store_true",
        help="Whether or not to freeze GPT2's attention layers.",
    )
    parser.add_argument(
        "--freeze-ff",
        action="store_true",
        help="Whether or not to freeze GPT2's feed-forward networks.",
    )
    # Hyperparameters
    parser.add_argument(
        "--freeze-until",
        default=-1,
        type=int,
        help="Hyperparameter: decoder layers to freeze the specified modules for, starting from 0 up until the specified number, maximum 11.",
    )
    parser.add_argument(
        "--learning-rate",
        default=0.0005,
        type=float,
        help="Hyperparameter: learning rate.",
    )
    parser.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="Hyperparameter: dropout probability.",
    )
    parser.add_argument(
        "--orth-gain",
        default=None,
        type=float,
        help="Hyperparameter: orthogonal gain of input layer.",
    )
    parser.add_argument(
        "--batch-size",
        default=1,
        type=int,
        help="Hyperparameter: batch size.",
    )
    parser.add_argument(
        "--epochs",
        default=1,
        type=int,
        help="Hyperparameter: number of times to go through training data.",
    )
    args = parser.parse_args()
    cwd = os.getcwd()

    # Cluster specific settings (Hydra)
    if args.cluster:

        # Specific threads when running on Hydra HPC (https://hpc.vub.be/docs/software/usecases/#pytorch)
        torch.set_num_threads(len(os.sched_getaffinity(0)))
        torch.set_num_interop_threads(1)

    # Resources
    n_gpus = torch.cuda.device_count()
    tqdm.tqdm.write("Available GPU(S): " + str(n_gpus))
    n_cpus = multiprocessing.cpu_count()
    tqdm.tqdm.write("Available CPU(S): " + str(n_cpus))
    if torch.cuda.is_available():
        tqdm.tqdm.write("GPU(s) detected: running on GPU.")
        device = torch.device("cuda")
    else:
        tqdm.tqdm.write("No GPU(s) detected: running on CPU.")
        device = torch.device("cpu")
    gradient_accumulation = 16

    # Dataset
    # TODO: add argument to get other processed data
    subjects, n_subjects, n_channels, n_classes = dataset.dataset_per_subject(
        "/data/brussel/102/vsc10248/data/processed"
        if args.cluster
        else "data/processed",
        device,
    )

    # Run id
    run_id = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))

    # Run function
    def run_fn(hyperparams, checkpoint_dir=None):

        # Disable printing to terminal when optimising as Ray Actors can not get a lock on stdout
        if args.optimise is not None:
            tqdm.tqdm.__init__ = functools.partialmethod(
                tqdm.tqdm.__init__, disable=True
            )

        # Run type
        run_type = (
            "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
            if args.optimise is not None
            else "CV"
        )

        # Subject-wise cross validation
        test_loss_avg, test_acc_avg = 0.0, 0.0
        for validation_subject_idx in range(n_subjects):

            """
            DATA
            """

            # Must be able to accumulate gradients
            assert (
                hyperparams["batch_size"] <= gradient_accumulation
                or hyperparams["batch_size"] % gradient_accumulation == 0
            )
            eval_batch_size = hyperparams["batch_size"]
            train_batch_size = (
                gradient_accumulation
                if eval_batch_size > gradient_accumulation
                else eval_batch_size
            )
            gradient_accumulation_steps = (
                eval_batch_size // gradient_accumulation
                if eval_batch_size > gradient_accumulation
                else 1
            )

            # Validation subject
            validation_subject = subjects[validation_subject_idx]
            validation_sampler = torch.utils.data.RandomSampler(validation_subject)
            validation_loader = torch.utils.data.DataLoader(
                validation_subject,
                batch_size=eval_batch_size,
                num_workers=4,  # TODO: get good number
                sampler=validation_sampler,
                pin_memory=args.cluster,  # On the cluster there is sufficient CUDA pinned memory
            )

            # Test subject
            test_subject_idx = (
                validation_subject_idx + 1
            ) % n_subjects  # Always next one, first is test for last
            test_subject = subjects[test_subject_idx]
            test_sampler = torch.utils.data.RandomSampler(test_subject)
            test_loader = torch.utils.data.DataLoader(
                test_subject,
                batch_size=eval_batch_size,
                num_workers=4,  # TODO: get good number
                sampler=test_sampler,
                pin_memory=args.cluster,  # On the cluster there is sufficient CUDA pinned memory
            )

            # Train subjects
            train_subjects = torch.utils.data.dataset.ConcatDataset(
                [
                    subject
                    for subject in subjects
                    if subject != validation_subject and subject != test_subject
                ]
            )
            train_sampler = torch.utils.data.RandomSampler(train_subjects)
            train_loader = torch.utils.data.DataLoader(
                train_subjects,
                batch_size=train_batch_size,
                num_workers=4,  # TODO: get good number
                sampler=train_sampler,
                pin_memory=args.cluster,  # On the cluster there is sufficient CUDA pinned memory
            )

            """
            MODEL
            """

            model = models.FreezableGPT2(
                n_channels,
                n_classes,
                hyperparams["dropout"],
                orth_gain=hyperparams["orth_gain"],
                pretrained=args.pretrained_transformer,
                freeze_trans_layers_until=hyperparams["freeze_until"],
                freeze_pos=args.freeze_pos,
                freeze_ln=args.freeze_ln,
                freeze_attn=args.freeze_attn,
                freeze_ff=args.freeze_ff,
            )
            model.to(device)

            """
            LOGGING
            """

            if args.wandb:
                group_name = f"{args.name}-{run_id}"
                config = dict(
                    **vars(args),
                    hyperparams=hyperparams,
                    run_type=run_type,
                )
                run = wandb.init(
                    name="test-subject-" + str(test_subject_idx + 1),
                    group=group_name,
                    project="fpt-for-eeg",
                    config=config,
                    job_type=run_type,
                    reinit=True,
                )
                # TODO: see what this actually does
                # wandb.watch(model)

            """
            TRAINING
            """

            # Loss
            ce_loss = torch.nn.CrossEntropyLoss()

            def loss_fn(out, y):
                out = out[:, 0]
                return ce_loss(out, y)

            # Accuracy
            def acc_fn(preds, true):
                preds = preds[:, 0].argmax(-1)
                return (preds == true).mean()

            # Optimiser
            optimiser = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])

            # Epochs
            validation_acc_best = 0.0
            best_model = model
            for _ in tqdm.tqdm(
                range(hyperparams["epochs"]), desc="Epochs", unit="epochs"
            ):

                # Train
                model.train()
                train_loss, train_acc = 0.0, 0.0
                for batch_x, batch_y in tqdm.tqdm(
                    train_loader, desc="Training", unit="batches"
                ):
                    # batch_x = batch_x.to(device=device, dtype=torch.float32)
                    # batch_y = batch_y.to(device=device, dtype=torch.long)

                    # Gradient Accumulation
                    for _ in range(gradient_accumulation_steps):

                        # Pass through model
                        output = model(batch_x)

                        # Loss
                        loss = loss_fn(output, batch_y) / gradient_accumulation_steps
                        loss.backward()
                        train_loss += loss.detach().cpu().item() / len(train_loader)

                        # Accuracy
                        train_acc += acc_fn(
                            output.detach().cpu().numpy(),
                            batch_y.detach().cpu().numpy(),
                        ) / (len(train_loader) + gradient_accumulation_steps)

                    # Learn
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimiser.step()
                    optimiser.zero_grad(
                        set_to_none=True
                    )  # Setting to None is faster than to 0

                tqdm.tqdm.write("Training accuracy: " + str(train_acc))
                tqdm.tqdm.write("Training loss    : " + str(train_acc))

                # Validate
                model.eval()
                validation_loss, validation_acc = 0.0, 0.0
                with torch.no_grad():
                    for batch_x, batch_y in tqdm.tqdm(
                        validation_loader, desc="Validation", unit="batches"
                    ):
                        # batch_x = batch_x.to(device=device)
                        # batch_y = batch_y.to(device=device)

                        # Pass through model
                        output = model(batch_x)

                        # Loss
                        validation_loss += loss_fn(
                            output, batch_y
                        ).detach().cpu().item() / len(validation_loader)

                        # Accuracy
                        validation_acc += acc_fn(
                            output.detach().cpu().numpy(),
                            batch_y.detach().cpu().numpy(),
                        ) / len(validation_loader)

                tqdm.tqdm.write("Validation accuracy: " + str(train_acc))
                tqdm.tqdm.write("Validation loss    : " + str(train_acc))

                # Retain best
                if validation_acc > validation_acc_best:
                    best_model = copy.deepcopy(model)
                    validation_acc_best = validation_acc

                # Log epoch
                if args.wandb:
                    wandb.log(
                        {
                            "Train Loss": train_loss,
                            "Train Accuracy": train_acc,
                            "Validation Loss": validation_loss,
                            "Validation Accuracy": validation_acc,
                        }
                    )

            """
            EVALUATION
            """
            best_model.eval()
            test_loss, test_acc = 0.0, 0.0
            with torch.no_grad():
                for batch_x, batch_y in tqdm.tqdm(
                    test_loader, desc="Evaluation", unit="batches"
                ):
                    # batch_x = batch_x.to(device=device, dtype=torch.float32)
                    # batch_y = batch_y.to(device=device, dtype=torch.float64)

                    # Pass through model
                    output = best_model(batch_x)

                    # Loss
                    test_loss += loss_fn(output, batch_y).detach().cpu().item() / len(
                        test_loader
                    )

                    # Accuracy
                    test_acc += acc_fn(
                        output.detach().cpu().numpy(),
                        batch_y.detach().cpu().numpy(),
                    ) / len(test_loader)

            # Test subject avg
            test_loss_avg += test_loss / n_subjects
            test_acc_avg += test_acc / n_subjects

            # Log evaluation
            if args.wandb:
                wandb.log(
                    {
                        "Test Accuracy": test_acc,
                        "Test Loss": test_loss,
                    }
                )

            # Cleanup
            del model
            del best_model
            torch.cuda.synchronize()
            time.sleep(5)
            if args.wandb:
                run.finish()

        # Log test subject avgs to raytune
        if args.optimise is not None:
            ray.tune.report(
                loss=test_loss_avg,
                accuracy=test_acc_avg,
            )

        # Log test subject avgs to terminal
        tqdm.tqdm.write("Avg test accuracy: " + str(train_acc))
        tqdm.tqdm.write("Avg test loss    : " + str(train_acc))

    if args.optimise is None:
        """
        Single run using given hyperparameters
        """

        # Hyperparams
        hyperparams = {
            "freeze_until": args.freeze_until,
            "lr": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "dropout": args.dropout,
            "orth_gain": args.orth_gain,
        }

        # Simple run
        run_fn(hyperparams)

    else:
        """
        Optimisation, raytune is used to spawn #args.optimise trials with various hyperparameter values
        """
        tqdm.tqdm.write(
            "Running optimisation through raytune, ignoring passed hyperparameters."
        )

        # Hyperparameters
        hyperparams = {
            "freeze_until": hyperopt.hp.randint("freeze_until", 11),
            "lr": hyperopt.hyperopt.hp.loguniform(
                "lr", numpy.log(5e-5), numpy.log(1e-1)
            ),
            "batch_size": hyperopt.hp.choice("batch_size", [16, 32, 64, 128]),
            "epochs": hyperopt.hp.choice("epochs", [8, 16, 32, 64]),
            "dropout": hyperopt.hp.loguniform(
                "dropout", numpy.log(0.001), numpy.log(1.0)
            ),
            "orth_gain": hyperopt.hp.loguniform(
                "orth_gain", numpy.log(0.001), numpy.log(2.0)
            ),
        }

        # Tune algorithm (Tree-structured Parzen Estimator)
        hyperopt_search = ray.tune.suggest.HyperOptSearch(
            hyperparams, metric="accuracy", mode="max"
        )

        # Tune reporter
        reporter = ray.tune.CLIReporter(
            metric_columns=["loss", "accuracy", "training_iteration"],
            max_report_frequency=20,
        )

        # Optimisation
        result = ray.tune.run(
            run_fn,
            resources_per_trial={
                "cpu": multiprocessing.cpu_count() / torch.cuda.device_count()
                if n_gpus > 0
                else 1,
                "gpu": 1 if n_gpus > 0 else 0,
            },
            num_samples=args.optimise,
            search_alg=hyperopt_search,
            progress_reporter=reporter,
            checkpoint_freq=0,
            local_dir="/data/brussel/102/vsc10248/optimisation"
            if args.cluster
            else "optimisation",
        )

        # Terminal logging
        best_trial = result.get_best_trial("acc", "max", "last")
        if result is not None and best_trial is not None:
            print("Best trial config: {}".format(best_trial.config))
            print(
                "Best trial final test loss: {}".format(best_trial.last_result["loss"])
            )
            print(
                "Best trial final test accuracy: {}".format(
                    best_trial.last_result["accuracy"]
                )
            )
