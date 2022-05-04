import wandb

api = wandb.Api(timeout=20)
runs = api.runs(path="wulfdewolf/lpt-for-eeg", filters={"config.name": "signal-optimisation"})
print(len(runs))

for run in runs:
    hyperparams = run.config["hyperparams"] 
    hyperparams.pop("freeze_until")
    run.config["hyperparams"] = hyperparams
    run.config["freeze_lower"] = 1
    run.config["freeze_upper"] = 12
    run.config["freeze_between"] = [1,12]
    run.update()