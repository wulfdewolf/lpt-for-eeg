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
import numpy as np
import multiprocessing

mne.set_log_level(False)

from functools import partial

from dn3.configuratron import ExperimentConfig
from dn3.trainable.processes import StandardClassification

from dn3_ext import LinearHeadBENDR, FPTBENDR

from ray import tune
from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch


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
        "--freeze-transformer-layers",
        nargs="*",
        default=[],
        help="The transformer layers to freeze, in the case of GPT2: 0-11 (only for when FPTBENDR chosen).",
    )
    parser.add_argument(
        "--freeze-transformer-layers-until",
        default=None,
        type=int,
        help="The transformer layers to freeze, starting from 0 to the specified number, in the case of GPT2 maximum 11 (only for when FPTBENDR chosen).",
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
    cwd = os.getcwd()

    # Cluster specific things
    if args.cluster:

        # Specific threads when running on HPC (https://hpc.vub.be/docs/software/usecases/#pytorch)
        torch.set_num_threads(len(os.sched_getaffinity(0)))
        torch.set_num_interop_threads(1)

    # Dataset
    ds_name, ds = list(experiment.datasets.items())[0]

    # num_workers test
    if args.num_workers_test:
        num_workers = utils.num_workers_test(ds.dataset, ds.train_params.batch_size)
    else:
        num_workers = 4

    # Hyperparams
    if args.optimise is not None:
        hyperparams = {
            "lr": hp.loguniform("lr", np.log(5e-5), np.log(1e-1)),
            "weight_decay": hp.loguniform("weight_decay", np.log(0.001), np.log(1.0)),
            "batch_size": hp.choice("batch_size", [16, 32, 64, 128]),
            "epochs": hp.choice("epochs", [8, 16, 32, 64]),
            "enc_do": hp.loguniform("enc_do", np.log(0.001), np.log(1.0)),
            "feat_do": hp.loguniform("feat_do", np.log(0.001), np.log(1.0)),
            "freeze_until": hp.randint("freeze_until", 11),
        }
    else:
        hyperparams = ds.train_params

    # Run id
    run_id = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))

    # Run function
    def run_fn(hyperparams, checkpoint_dir=None):

        # Run type
        run_type = (
            "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
            if args.optimise is not None
            else "CV"
        )

        # Test scores
        test_loss = []
        test_acc = []

        # Optional additional metrics
        added_metrics, retain_best, _ = utils.get_ds_added_metrics(
            ds_name, cwd + "/" + args.metrics_config
        )

        # Cross validation
        for fold, (training, validation, test) in enumerate(
            tqdm.tqdm(utils.get_lmoso_iterator(ds_name, ds))
        ):
            tqdm.tqdm.write(torch.cuda.memory_summary())

            if args.model == utils.MODEL_CHOICES[0]:
                model = LinearHeadBENDR.from_dataset(
                    training,
                    enc_do=hyperparams["enc_do"],
                    feat_do=hyperparams["feat_do"],
                )
            else:
                model = FPTBENDR.from_dataset(
                    training,
                    multi_gpu=args.multi_gpu,
                    pretrained=args.pretrained_transformer,
                    freeze_trans_layers=args.freeze_transformer_layers,
                    freeze_trans_layers_until=hyperparams["freeze_until"],
                    freeze_pos=args.freeze_pos,
                    freeze_ln=args.freeze_ln,
                    freeze_attn=args.freeze_attn,
                    freeze_ff=args.freeze_ff,
                    enc_do=hyperparams["enc_do"],
                    feat_do=hyperparams["feat_do"],
                )

            if args.pretrained_encoder:
                model.load_pretrained_modules(
                    cwd + "/" + experiment.encoder_weights,
                    cwd + "/" + experiment.context_weights,
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
                group_name = f"{args.name}-{args.model}-{run_id}"
                config = dict(
                    **vars(args),
                    **vars(experiment),
                    hyperparams=hyperparams,
                    run_type=run_type,
                )
                run = wandb.init(
                    name="subject " + str(fold + 1),
                    group=group_name,
                    project="fpt-for-eeg",
                    config=config,
                    job_type=run_type,
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
                            "Average Train Accuracy": sum(tr_accuracy)
                            / len(tr_accuracy),
                            "Average Train loss": sum(tr_loss) / len(tr_loss),
                            "Start Train Accuracy": tr_accuracy[0],
                            "Final Train Loss": tr_loss[-1],
                            "Average Validation Accuracy": validation_metrics[
                                "Accuracy"
                            ],
                            "Average Validation Loss": validation_metrics["loss"],
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
                pin_memory=True,
                num_workers=args.num_workers,
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
            test_loss.append(metrics["loss"])
            test_acc.append(metrics["Accuracy"])

            # Log test scores to WandB
            if args.wandb:
                wandb.log(
                    {
                        "Test Accuracy": metrics["Accuracy"],
                        "Test Loss": metrics["loss"],
                    }
                )

            # Explicitly garbage collect here, don't want to fit two models in GPU at once
            del process
            del model
            torch.cuda.synchronize()
            time.sleep(10)
            if args.wandb:
                run.finish()

        # Log test scores to tune
        if args.optimise is not None:
            tune.report(
                loss=sum(test_loss) / len(test_loss),
                accuracy=sum(test_acc) / len(test_acc),
            )

    # Optimisation or simple run
    if args.optimise is None:

        # Simple run
        run_fn(hyperparams)

    else:

        # Tune algorithm (Tree-structured Parzen Estimator)
        hyperopt_search = HyperOptSearch(hyperparams, metric="accuracy", mode="max")

        # Optimisation
        result = tune.run(
            run_fn,
            resources_per_trial={
                "cpu": multiprocessing.cpu_count() / torch.cuda.device_count(),
                "gpu": 1,
            },
            num_samples=args.optimise,
            search_alg=hyperopt_search,
            checkpoint_freq=0,
            local_dir="/data/brussel/102/vsc10248/optimisation"
            if args.cluster
            else "optimisation",
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
