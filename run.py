from dn3_ext import FPTBENDR
import torch
import tqdm
import argparse
import time
import utils
import random
import string
import wandb
import mne
import os

mne.set_log_level(False)

from functools import partial

from dn3.configuratron import ExperimentConfig
from dn3.trainable.processes import StandardClassification

from dn3_ext import LinearHeadBENDR, FPTBENDR

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tunes BENDR and FPT models.")
    parser.add_argument("model", choices=utils.MODEL_CHOICES)
    parser.add_argument(
        "--ds-config",
        default="configs/downstream.yml",
        help="The DN3 config file to use.",
    )
    parser.add_argument(
        "--metrics-config",
        default="configs/metrics.yml",
        help="Where the listings for config " "metrics are stored.",
    )
    parser.add_argument(
        "--subject-specific",
        action="store_true",
        help="Fine-tune on target subject alone.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log training to WandB.",
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
    parser.add_argument(
        "--multi-gpu", action="store_true", help="Distribute over multiple GPUs"
    )
    parser.add_argument(
        "--num-workers", default=4, type=int, help="Number of dataloader workers."
    )
    parser.add_argument(
        "--pretrained-encoder",
        action="store_true",
        help="Whether or not to use a pretrained encoder.",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Whether or not to freeze the encoder during fine-tuning.",
    )
    parser.add_argument(
        "--pretrained-transformer",
        action="store_true",
        help="Whether or not to use a pretrained transformer (only for when FPTBENDR chosen).",
    )
    parser.add_argument(
        "--freeze-transformer",
        action="store_true",
        help="Whether or not to freeze the transformer during fine-tuning (only for when FPTBENDR chosen).",
    )
    parser.add_argument(
        "--freeze-pos",
        action="store_true",
        help="Whether or not to freeze the positional layers of the transformer during fine-tuning (only for when FPTBENDR chosen).",
    )
    parser.add_argument(
        "--freeze-ln",
        action="store_true",
        help="Whether or not to freeze the layer-norm layers of the transformer during fine-tuning (only for when FPTBENDR chosen).",
    )
    parser.add_argument(
        "--freeze-attn",
        action="store_true",
        help="Whether or not to freeze the attention layers of the transformer during fine-tuning (only for when FPTBENDR chosen).",
    )
    parser.add_argument(
        "--freeze-ff",
        action="store_true",
        help="Whether or not to freeze the feed-forward networks of the transformer during fine-tuning (only for when FPTBENDR chosen).",
    )
    args = parser.parse_args()
    experiment = ExperimentConfig(args.ds_config)

    # WandB
    if args.wandb:
        experiment_id = "".join(
            random.choices(string.ascii_uppercase + string.digits, k=6)
        )
        group_name = f"{args.name}-{args.model}-{experiment_id}"
        config = dict(**vars(args), **vars(experiment))

    # Cluster specific things
    if args.cluster:

        # Specific threads when running on HPC (https://hpc.vub.be/docs/software/usecases/#pytorch)
        torch.set_num_threads(len(os.sched_getaffinity(0)))
        torch.set_num_interop_threads(1)

    # Hyperparams
    ds_name, ds = list(experiment.datasets.items())[0]
    if args.optimise is not None:
        hyperparams = {
            "lr": tune.loguniform(5e-5, 1e-1),
            "weight_decay": tune.loguniform(0.1, 1),
            "batch_size": tune.choice([2, 4, 8, 16, 32, 64]),
            "epochs": tune.choice([2, 4, 8, 16, 32]),
        }
    else:
        hyperparams = {
            "lr": ds.lr,
            "weight_decay": ds.weight_decay,
            "batch_size": ds.train_params.batch_size,
            "epochs": ds.train_params.epochs,
        }

    # Training function
    cwd = os.getcwd()

    def run_fn(hyperparams):

        # Cross validation
        added_metrics, retain_best, _ = utils.get_ds_added_metrics(
            ds_name, cwd + "/" + args.metrics_config
        )
        for fold, (training, validation, test) in enumerate(
            tqdm.tqdm(utils.get_lmoso_iterator(ds_name, ds))
        ):
            tqdm.tqdm.write(torch.cuda.memory_summary())

            if args.model == utils.MODEL_CHOICES[0]:
                model = LinearHeadBENDR.from_dataset(training)
            else:
                model = FPTBENDR.from_dataset(
                    training,
                    multi_gpu=args.multi_gpu,
                    pretrained=args.pretrained_transformer,
                    freeze_trans=args.freeze_transformer,
                    freeze_pos=args.freeze_pos,
                    freeze_ln=args.freeze_ln,
                    freeze_attn=args.freeze_attn,
                    freeze_ff=args.freeze_ff,
                )

            if args.pretrained_encoder:
                model.load_pretrained_modules(
                    experiment.encoder_weights,
                    experiment.context_weights,
                    freeze_encoder=args.freeze_encoder,
                )
            process = StandardClassification(model, metrics=added_metrics)
            process.set_optimizer(
                torch.optim.Adam(
                    process.parameters(),
                    hyperparams["lr"],
                    weight_decay=hyperparams["weight_decay"],
                )
            )

            # WandB
            if args.wandb:
                run = wandb.init(
                    name="subject " + str(fold),
                    group=group_name,
                    project="fpt-for-eeg",
                    config=config,
                    job_type="optimisation" if args.optimise is not None else "CV",
                    reinit=True,
                )
                wandb.watch(model)

                tr_accuracy = []
                tr_loss = []

                def log_callback(train_metrics, tr_accuracy=[], tr_loss=[]):
                    tr_accuracy.append(train_metrics["Accuracy"])
                    tr_loss.append(train_metrics["loss"])

                def epoch_callback(validation_metrics, tr_accuracy=[], tr_loss=[]):
                    wandb.log(
                        {
                            "Average training accuracy": sum(tr_accuracy)
                            / len(tr_accuracy),
                            "Average training loss": sum(tr_loss) / len(tr_loss),
                            "Start training accuracy": tr_accuracy[0],
                            "Start training accuracy": tr_accuracy[0],
                            "End training loss": tr_loss[-1],
                            "End training loss": tr_loss[-1],
                            "Average validation accuracy": validation_metrics[
                                "Accuracy"
                            ],
                            "Average validation loss": validation_metrics["loss"],
                        }
                    )
                    tr_accuracy = []
                    tr_loss = []

            # Fit everything
            process.fit(
                training_dataset=training,
                validation_dataset=validation,
                warmup_frac=0.1,
                retain_best=retain_best,
                pin_memory=False,
                log_callback=partial(
                    log_callback, tr_accuracy=tr_accuracy, tr_loss=tr_loss
                )
                if args.wandb
                else lambda x: None,
                epoch_callback=partial(
                    epoch_callback, tr_accuracy=tr_accuracy, tr_loss=tr_loss
                )
                if args.wandb
                else lambda x: None,
                train_log_interval=1,
                batch_size=hyperparams["batch_size"],
                epochs=hyperparams["epochs"],
            )

            # Test scores
            metrics = process.evaluate(test)
            if args.wandb:
                wandb.log(
                    {
                        "Test accuracy": metrics["Accuracy"],
                        "Test loss": metrics["loss"],
                    }
                )

            # Log averages to tune
            if args.optimise is not None:
                tune.report(loss=metrics["loss"], accuracy=metrics["Accuracy"])

            # explicitly garbage collect here, don't want to fit two models in GPU at once
            del process
            del model
            torch.cuda.synchronize()
            time.sleep(10)
            if args.wandb:
                run.finish()

    # Optimisation or simple run
    if args.optimise is not None:

        # Tune scheduler
        scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=10,
            grace_period=1,
            reduction_factor=2,
        )

        # Tune reporter
        reporter = CLIReporter(metric_columns=["loss", "accuracy"])

        # Optimisation
        result = tune.run(
            run_fn,
            resources_per_trial={"cpu": 1, "gpu": 1},
            config=hyperparams,
            num_samples=args.optimise,
            scheduler=scheduler,
            progress_reporter=reporter,
            local_dir="optimisation",
        )

        # Terminal logging
        best_trial = result.get_best_trial("loss", "min", "last")
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

    else:

        # Run
        run_fn(hyperparams)
