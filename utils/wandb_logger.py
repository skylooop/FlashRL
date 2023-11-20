import wandb

def setup_wandb(config):
    run = wandb.init(config=dict(config),
                     project=config.project_name,
                     group=config.algo.name)
    return run
    
    
    