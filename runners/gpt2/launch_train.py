from pathlib import Path

from clt.config.clt_training_runner_config import CLTTrainingRunnerConfig
from clt.clt_training_runner import CLTTrainingRunner

from infra.distributed import (
    init_distributed,
    destroy_distributed,
    print_once,
)
from infra.paths import make_run_paths
from infra.fs import ensure_dirs
from infra.wandb_utils import get_synced_wandb_id

def main():

    ctx = init_distributed()
    print_once(
        f"Launcher={ctx.launcher} | "
        f"Rank {ctx.rank}/{ctx.world_size} | "
        f"Local rank {ctx.local_rank}"
    )

    MODEL = "gpt2"

    wandb_id = get_synced_wandb_id(ctx.rank)

    paths = make_run_paths(
        model=MODEL,
        project_name="clt",
        run_id=wandb_id,  # isolate runs cleanly
    )

    ensure_dirs(
        [
            paths.activations_root,
            paths.checkpoints_root,
        ]
    )

    total_training_steps = 300_000 // ctx.world_size
    train_batch_size_tokens = ctx.world_size * 1024
    total_training_tokens = train_batch_size_tokens * total_training_steps

    lr_decay_steps = (total_training_steps // 20) - 1
    final_lr_scale = 0.0
    lr_warm_up_steps = 1000

    l0_waiting_steps = 0
    l0_warm_up_steps = int(0.7 * total_training_steps) - l0_waiting_steps - 1
    decay_stable_steps = total_training_steps - l0_warm_up_steps - lr_decay_steps

    functional_loss = "kl"
    fc_coefficient = 0
    fc_warm_up_steps = 1000
    fc_waiting_steps = total_training_steps - fc_warm_up_steps - 1

    cfg = CLTTrainingRunnerConfig(
        device="cuda",
        dtype="float32",
        seed=42,
        n_checkpoints=0,
        checkpoint_path=str(paths.checkpoints_root),
        logger_verbose=True,
        model_class_name="HookedTransformer",
        model_name=MODEL,
        context_size=16,
        from_pretrained_path=None,
        d_in=768,
        expansion_factor=32,
        jumprelu_init_threshold=0.03,
        jumprelu_bandwidth=1.0,
        cached_activations_path=str(paths.activations_root),
        n_train_batch_per_buffer=100, # depends on the size of your buffer
        total_training_tokens=total_training_tokens,
        train_batch_size_tokens=train_batch_size_tokens,
        adam_beta1=0.9,
        adam_beta2=0.999,
        lr=2e-4,
        lr_warm_up_steps=lr_warm_up_steps,
        lr_decay_steps=lr_decay_steps,
        final_lr_scale=final_lr_scale,
        l0_coefficient=1.0,
        dead_penalty_coef=3e-4,
        dead_feature_window=250,
        l0_warm_up_steps=l0_warm_up_steps,
        l0_waiting_steps=l0_waiting_steps,
        decay_stable_steps=decay_stable_steps,
        log_to_wandb=True,
        wandb_project=f"{MODEL}-clt",
        wandb_id=wandb_id,
        wandb_log_frequency=10,
        eval_every_n_wandb_logs=100,
        run_name=None,
        wandb_entity=None,
        fsdp=False,
        ddp=False,
        feature_sharding=True,
        functional_loss=functional_loss,
        fc_coefficient=fc_coefficient,
        fc_warm_up_steps=fc_warm_up_steps,
        fc_waiting_steps=fc_waiting_steps,
    )

    trainer = CLTTrainingRunner(cfg, rank=ctx.rank, world_size=ctx.world_size)
    trainer.run()

    destroy_distributed()


if __name__ == "__main__":
    main()