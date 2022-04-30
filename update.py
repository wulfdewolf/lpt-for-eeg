import wandb

api = wandb.Api()
runs = api.runs(path="wulfdewolf/lpt-for-eeg", filters={"group_name":"features-optimisation-downsampled-MI924"})

print(len(runs))
for run in runs:
    hyperparams = run.config["hyperparams"]
    hyperparams["decay"] = None
    run.config["hyperparams"] = hyperparams
    run.update()