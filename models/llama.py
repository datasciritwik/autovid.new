import modal
from fastapi import Header

MODEL_ID = "NousResearch/Meta-Llama-3-8B"
MODEL_REVISION = "315b20096dc791d381d514deb5f8bd9c8d6d3061"

image = modal.Image.debian_slim().pip_install(
    "transformers==4.49.0", "torch==2.6.0", "accelerate==1.4.0", "fastapi[standard]"
)
app = modal.App("example-base-Meta-Llama-3-8B", image=image)

GPU_CONFIG = "A100-80GB:2"

CACHE_DIR = "/cache"
cache_vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)


@app.cls(
    gpu=GPU_CONFIG,
    volumes={CACHE_DIR: cache_vol},
    scaledown_window=60 * 10,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("header-api")],
)
@modal.concurrent(max_inputs=15)
class Model:
    @modal.enter()
    def setup(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import os
        from huggingface_hub import snapshot_download

        # Download the model to the cache directory
        model_path = snapshot_download(repo_id=MODEL_ID, cache_dir=CACHE_DIR)

        print(f"Model downloaded to: {model_path}")

        # Specify cache directory if needed
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

        self.pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    @modal.method()
    def generate(self, input: str):
        return self.pipeline(input)

    @modal.fastapi_endpoint(docs=True, method="POST")
    def infer(self, prompt: str, x_api_key: str = Header(...)):
        """
        FastAPI endpoint for inference. Accepts a prompt and requires X-API-KEY header to match secret.
        """
        # Retrieve the expected API key from environment (set via Modal secret)
        import os
        # secrets=[modal.Secret.from_name("header-api")]
        expected_api_key = os.getenv("HEADER_API")
        if x_api_key != expected_api_key:
            return {"error": "Unauthorized: Invalid API Key"}
        outputs = self.pipeline(prompt, max_new_tokens=256)
        # Return the generated text (handle both list and dict outputs)
        if isinstance(outputs, list) and len(outputs) > 0:
            return {"generated_text": outputs[0].get("generated_text", outputs[0].get("text", ""))}
        return {"generated_text": ""}


# ## Run the model
@app.local_entrypoint()
def main(prompt: str = None):
    if prompt is None:
        prompt = "Please write a Python function to compute the Fibonacci numbers."
    print(Model().generate.remote(prompt))