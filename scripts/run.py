from src.experiment import run_experiment
from ray import tune

if __name__ == "__main__":

    experiment_name = "first_sem_trials"

    experiment_params = dict(
        seed=20200220,
        # Data
        task="competition",
        window_size=6,
        folds=5,
        # Model
        model_type="FPT",
        model_name="gpt2",
        # Pretraining
        pretrained=True,
        # Freezing
        freeze_trans=True,
        freeze_in=False,
        freeze_pos=False,
        freeze_ln=False,
        freeze_attn=True,
        freeze_ff=True,
        freeze_out=False,
        # Hyper parameters
        # optimise=True,  # Whether or not a hyperparameters should be optimised
        # hyperparams=dict(
        #    learning_rate=tune.loguniform(1e-4, 1e-1),
        #    batch_size=tune.choice([2, 4, 8, 16, 32]),
        #    dropout=tune.loguniform(0.1, 1),
        #    orth_gain=tune.loguniform(0.1, 3),
        # ),
        optimise=False,  # Whether or not a hyperparameters should be optimised
        hyperparams=dict(
            learning_rate=0.001,
            batch_size=30,
            dropout=0.1,
            orth_gain=1.41,
        ),
    )

    run_experiment(experiment_name, experiment_params)
