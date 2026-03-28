import sagemaker
from sagemaker.huggingface import HuggingFace

# ── Replace these with your actual S3 bucket name ──────────────────────────
S3_BUCKET = "<your-bucket>"
S3_CHECKPOINT_URI = f"s3://{S3_BUCKET}/checkpoints"
S3_TRAIN_URI      = f"s3://{S3_BUCKET}/Train"
S3_VAL_URI        = f"s3://{S3_BUCKET}/Val"
# ───────────────────────────────────────────────────────────────────────────

role = sagemaker.get_execution_role()
sagemaker_session = sagemaker.Session()

estimator = HuggingFace(
    entry_point="train.py",
    source_dir="sm_train",
    instance_type="ml.g5.12xlarge",   # 4x A10G 24GB
    instance_count=1,
    role=role,
    sagemaker_session=sagemaker_session,
    transformers_version="4.36.0",
    pytorch_version="2.1.0",
    py_version="py310",
    distribution={"torch_distributed": {"enabled": True}},
    hyperparameters={
        "epochs": 10,
        "batch_size": 32,              # per-GPU; effective batch = 32 * 4 GPUs = 128
        "learning_rate": 5e-5,
        "warmup_steps": 200,
        "eval_every_n_epochs": 2,
        "model_name": "microsoft/trocr-large-stage1",
        "gradient_accumulation_steps": 1,
        "save_every_n_steps": 200,
        "max_grad_norm": 1.0,
    },
    max_run=86400,    # 24 hours
    checkpoint_s3_uri=S3_CHECKPOINT_URI,
)

estimator.fit({
    "training":   S3_TRAIN_URI,
    "validation": S3_VAL_URI,
})

print(f"Training job name: {estimator.latest_training_job.name}")
