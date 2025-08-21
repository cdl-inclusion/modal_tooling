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
#
# Will mount a volume for permanente storage called STORAGE_VOLUME_NAME.
# If you adjust the storage name, ensure that other tools accessing the volume use the right name (eg tensorboard-server.py)
#
#

###########################
# Adjust these
#
JUPYTER_PORT = 8888
TIMEOUT = 3600 # seconds
# TIMEOUT = 86400  # 24 hours maximum for Modal sandbox -- if training longer, consider using a Modal function!
# -> when you use that, don't forget to stop after you're done!
GPU_TYPE = 'l4' # choose according to: https://modal.com/pricing
NUM_CPUS = 1 # for training want more than 1 (4 is good)
MEM = 2048 # for training you need more (16384 is a good default)
###########################


import json
import secrets
import time
import urllib.request

import modal

STORAGE_VOLUME_NAME = "jupyter_kernel"

app = modal.App.lookup(STORAGE_VOLUME_NAME, create_if_missing=True)

volume = modal.Volume.from_name(STORAGE_VOLUME_NAME, create_if_missing=True)


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
	"accelerate>=0.26.0",
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
        "transformers==4.52.0", # to avoid some issues with the latest version as discussed here: https://huggingface.co/openai/whisper-large-v3/discussions/201
	"tensorboard"
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
        cpu=NUM_CPUS,
        memory=MEM,
        volumes={f"/{STORAGE_VOLUME_NAME}": volume}
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
