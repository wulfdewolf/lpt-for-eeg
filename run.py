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
        "--name",
        default="thesis",
        help="Name of experiment.",
    )
    parser.add_argument(
        "--multi-gpu", action="store_true", help="Distribute BENDR over multiple GPUs"
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
        "--freeze-att",
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

    print("EXPERIMENT: " + args.name)
    print(args.pretrained_encoder)

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

    # Cross validation
    for ds_name, ds in tqdm.tqdm(
        experiment.datasets.items(),
        total=len(experiment.datasets.items()),
        desc="Datasets",
    ):
        added_metrics, retain_best, _ = utils.get_ds_added_metrics(
            ds_name, args.metrics_config
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
                torch.optim.Adam(process.parameters(), ds.lr, weight_decay=0.01)
            )

            # WandB
            if args.wandb:
                run = wandb.init(
                    name="subject " + str(fold),
                    group=group_name,
                    project="fpt-for-eeg",
                    config=config,
                    job_type="fold",
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
                **ds.train_params._d,
            )

            # explicitly garbage collect here, don't want to fit two models in GPU at once
            del process
            del model
            torch.cuda.synchronize()
            time.sleep(10)
            if args.wandb:
                run.finish()
