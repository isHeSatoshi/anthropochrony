from pathlib import Path

from huggingface_hub import HfApi

from anthropochrony.artifacts import file_hash, manifest, write_json
from anthropochrony.remote import authorization_guard


def publish_adapter(trainer, output, repo_id, subject):
    from peft import PeftModel

    if not isinstance(trainer.model, PeftModel):
        raise ValueError("Only a PEFT adapter may be uploaded")
    output = Path(output)
    for adapter in trainer.model.peft_config.values():
        adapter.base_model_name_or_path = subject["repo"]
        adapter.revision = subject["revision"]
    trainer.model.save_pretrained(output, safe_serialization=True)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        manifest(output / name)
    api = HfApi()
    with authorization_guard():
        api.create_repo(repo_id, private=True, exist_ok=True)
        if api.model_info(repo_id).private is not True:
            raise PermissionError(
                "Refusing to upload research artifacts to a non-private repository"
            )
        trainer.push_to_hub(repo_id, private=True)
        api.upload_folder(
            repo_id=repo_id,
            folder_path=output,
            allow_patterns=[
                "training-summary.json",
                "training-log.jsonl",
                "trainer_state.json",
                "mask_check.json",
                "lora_check.json",
                "hardware.json",
                "lexsi_provenance.json",
                "adapter_config.json",
                "adapter_model.safetensors",
                "*.manifest.json",
            ],
        )
        info = api.model_info(repo_id)
        if info.private is not True or not info.sha:
            raise PermissionError("The uploaded adapter must remain private with a pinned commit")
        identity = {
            "model_id": repo_id,
            "model_sha": info.sha,
            "private": True,
            "base": subject,
            "training_summary_sha256": file_hash(output / "training-summary.json"),
            "adapter_weights_sha256": file_hash(output / "adapter_model.safetensors"),
        }
        write_json(output / "identity.json", identity)
        return identity
