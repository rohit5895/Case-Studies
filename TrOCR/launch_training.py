import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace


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
    },
    max_run=86400,    # 24 hours
    checkpoint_s3_uri="s3://<your-bucket>/checkpoints",
)

estimator.fit({
    "training":   "s3://<your-bucket>/Train",
    "validation": "s3://<your-bucket>/Val",
})
