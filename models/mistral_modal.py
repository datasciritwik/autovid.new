import modal

# Modal setup
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "mistral_inference",
        "huggingface_hub",
        "fastapi[standard]",
        "uvicorn"
    )
)

app = modal.App(name="mistral-deploy")
volume = modal.Volume.from_name("mistral-model-volume")  # Ensure this volume exists


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("hf-token")],
    gpu="A10G",
    volumes={"/root/mistral_models": volume},
    timeout=300,
)
def download_model():
    from huggingface_hub import login, snapshot_download
    from pathlib import Path
    import os

    login(os.environ["HF_TOKEN"])
    local_dir = Path("/root/mistral_models/7B-Instruct-v0.3")
    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        allow_patterns=[
            "params.json",
            "consolidated.safetensors",
            "tokenizer.model.v3"
        ],
        local_dir=local_dir,
    )
    print("✅ Model downloaded to shared volume.")


@app.cls(
    image=image,
    secrets=[modal.Secret.from_name("hf-token")],
    gpu="A10G",
    volumes={"/root/mistral_models": volume},
    timeout=600,
)
class MistralWorker:
    def __enter__(self):
        from mistral_inference.transformer import Transformer
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        from pathlib import Path

        self.model_path = Path("/root/mistral_models/7B-Instruct-v0.3")
        self.tokenizer = MistralTokenizer.from_file(
            f"{self.model_path}/tokenizer.model.v3"
        )
        self.model = Transformer.from_folder(self.model_path)

    @modal.fastapi_endpoint(method="POST")
    def infer(self, request):
        from mistral_common.protocol.instruct.messages import UserMessage
        from mistral_common.protocol.instruct.request import ChatCompletionRequest
        from mistral_inference.generate import generate

        user_input = request.json["prompt"]

        completion_request = ChatCompletionRequest(
            messages=[UserMessage(content=user_input)]
        )
        tokens = self.tokenizer.encode_chat_completion(completion_request).tokens

        out_tokens, _ = generate(
            [tokens],
            self.model,
            max_tokens=512,
            temperature=0.9,
            top_p=0.95,
            repetition_penalty=1.1,
            eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id,
        )
        result = self.tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

        return {"response": result}
    
@app.local_entrypoint()
def main():
    worker = MistralWorker()
    response = worker.infer({"json": {"prompt": "What is ML?"}})
    print(response)
