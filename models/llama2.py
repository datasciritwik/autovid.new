import modal
from fastapi import Header

# MODEL_ID = "NousResearch/Meta-Llama-3-8B"
# MODEL_REVISION = "315b20096dc791d381d514deb5f8bd9c8d6d3061"
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct-1M"
MODEL_REVISION = "main"

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

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_DIR,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

        self.tokenizer = tokenizer
        self.model = model
        self.pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )

    def _generate_text(self, prompt: str, max_new_tokens=256):
        # Apply improved decoding configuration
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return outputs[0].get("generated_text", outputs[0].get("text", ""))

    @modal.method()
    def generate(self, input: str):
        return self._generate_text(input)

    @modal.fastapi_endpoint(docs=True, method="POST")
    def infer(self, prompt: str, x_api_key: str = Header(...)):
        """
        FastAPI endpoint for inference. Accepts a prompt and requires X-API-KEY header to match secret.
        """
        import os
        expected_api_key = os.getenv("HEADER_API")
        if x_api_key != expected_api_key:
            return {"error": "Unauthorized: Invalid API Key"}

        generated_text = self._generate_text(prompt)
        return {"generated_text": generated_text}
