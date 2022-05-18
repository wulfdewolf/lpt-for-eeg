import wandb

api = wandb.Api(timeout=20)
runs = api.runs(path="wulfdewolf/lpt-for-eeg", filters={"config.name": "signal-overfitting"})
print(len(runs))

for run in runs:
    hyperparams = run.config["hyperparams"] 
    hyperparams["decay"] = 0.9
    run.config["hyperparams"] = hyperparams
    run.config.pop("decay")
    run.update()