# Start a jupyter kernel on Modal as a sandbox
# See: https://modal.com/docs/examples/jupyter_sandbox
#
# Right now the image is composed of necessary dependencies for running Whisper models
# via FasterWhisper and huggingface transformers. Add other libraries to the kernel as needed
# or install them from the running notebook.
#
# run with:
# python start_jupyter_kernel.py
#
# then see sandbox dashboard:
# https://modal.com/sandboxes/personalizedmodels/main

###########################
# Adjust these
#
JUPYTER_PORT = 8888
TIMEOUT = 3600 # seconds
GPU_TYPE = 'L4' # choose according to: https://modal.com/pricing
###########################


import json
import secrets
import time
import urllib.request

import modal

app = modal.App.lookup("jupyter_kernel", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04", add_python="3.11")
    .apt_install(
        "wget",
        "git",
        "libsndfile1",
        "libsndfile1-dev",
        "ffmpeg",
        "pkg-config",
	"build-essential",)
    .pip_install(
        "jupyter~=1.1.0",
        "numpy",
	"itables",
        "librosa",
        "soundfile",
        "audioread",
        "datasets[audio]==3.6.0", # use 3.6.0 as latest version (4.0.0) has breaking changes
        "matplotlib",
        "evaluate",
	"jiwer",
        "huggingface_hub",
        "torch",
        "torchaudio",
        "ctranslate2",
        "faster_whisper",
        "transformers",
    )
)


token = secrets.token_urlsafe(13)
token_secret = modal.Secret.from_dict({"JUPYTER_TOKEN": token})





print("🏖️  Creating sandbox")

with modal.enable_output():
    sandbox = modal.Sandbox.create(
        "jupyter",
        "notebook",
        "--no-browser",
        "--allow-root",
        "--ip=0.0.0.0",
        f"--port={JUPYTER_PORT}",
        "--NotebookApp.allow_origin='*'",
        "--NotebookApp.allow_remote_access=1",
        encrypted_ports=[JUPYTER_PORT],
        secrets=[token_secret],
        timeout=TIMEOUT,
        image=image,
        app=app,
        gpu=GPU_TYPE, 
    )

print(f"🏖️  Sandbox ID: {sandbox.object_id}")

tunnel = sandbox.tunnels()[JUPYTER_PORT]
url = f"{tunnel.url}/?token={token}"
print(f"🏖️  Jupyter notebook is running at: {url}")


def is_jupyter_up():
    try:
        response = urllib.request.urlopen(f"{tunnel.url}/api/status?token={token}")
        if response.getcode() == 200:
            data = json.loads(response.read().decode())
            return data.get("started", False)
    except Exception:
        return False
    return False


# timeout for startup
startup_timeout = 60  # seconds
start_time = time.time()
while time.time() - start_time < startup_timeout:
    if is_jupyter_up():
        print("🏖️  Jupyter is up and running!")
        break
    time.sleep(1)
else:
    print("🏖️  Timed out waiting for Jupyter to start.")    
