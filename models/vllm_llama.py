import modal

MODEL_NAME = "NousResearch/Meta-Llama-3.1-70B-Instruct"
MODEL_REVISION = "d50656ee28e2c2906d317cbbb6fcb55eb4055a84"
VLLM_PORT = 8000

image = (
    modal.Image.debian_slim()
    .pip_install("vllm", "torch", "transformers")
)

app = modal.App("vllm-llama3-70b-instruct", image=image)

@app.function(gpu="A100-80GB:6", timeout=3600)
def run_vllm():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
    ]

    subprocess.run(cmd)
