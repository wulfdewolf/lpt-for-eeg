from EEGdecoding.experiment import run_experiment
from ray import tune

if __name__ == "__main__":

    experiment_name = "noob_trials"

    experiment_params = dict(
        seed=20200220,
        # Data
        task="BCI_Competition_IV_2a",
        window_size=1000,
        # Model
        model_type="CNN",
        model_name="gpt2",
        # Pretraining
        pretrained=True,  # if vit this is forced to true, if lstm this is forced to false
        # Freezing
        freeze_trans=True,  # if False, we don't check arguments other than in and out
        freeze_in=False,
        freeze_pos=False,
        freeze_ln=False,
        freeze_attn=True,
        freeze_ff=True,
        freeze_out=False,
        # Parameters
        optimise=True,
        hyperparams=dict(
            learning_rate=tune.loguniform(1e-4, 1e-1),
            batch_size=tune.choice([2, 4, 8, 16]),
            dropout=tune.loguniform(0.1, 1),
            orth_gain=tune.loguniform(0.1, 3),
        ),
        # hyperparams=dict(
        # learning_rate=0.1,
        # batch_size=16,
        # dropout=0.1,
        # orth_gain=1.42,
        # ),
    )

    run_experiment(experiment_name, experiment_params)
