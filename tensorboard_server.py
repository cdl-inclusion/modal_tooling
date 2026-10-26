"""
TensorBoard Server for Modal

A multi-experiment TensorBoard server that can monitor multiple training runs simultaneously.
Based on: https://modal.com/docs/examples/tensorflow_tutorial

Features:
- Single server handles multiple experiments
- Live monitoring of ongoing training
- Automatic volume reloading with graceful error handling
- URL-based experiment selection

Usage:
1. Development/Testing:
   modal serve tensorboard_server.py

2. Production Deployment:
   modal deploy tensorboard_server.py

Then access your experiments via URLs with logdir parameters:

Example:
https://your-modal-url/?logdir=trained_models/training_run1


URL Parameters:
    logdir - Path under /STORAGE_VOLUME_NAME to the experiment directory
             (required, can be nested directories)

Multiple Experiments:
    Start the server ONCE and access different experiments by changing the logdir parameter.
    Each unique logdir creates a separate TensorBoard instance that's cached for performance.
    You can monitor multiple training runs simultaneously in different browser tabs.

Volume Setup:
    Expects a Modal Volume named "STORAGE_VOLUME_NAME" containing your training logs.
    Ensure, that you are setting the sam STORAGE_VOLUME_NAME in start_jupyter_kernel.py!

Container Lifecycle:
    The server automatically stays alive while TensorBoard tabs are open (due to auto-refresh).
    It shuts down after 10 minutes of no requests (when all tabs are closed, computer sleeps, 
    or internet disconnects). Cold start takes ~10-30 seconds when accessing after shutdown.
"""

import modal
import os
import shutil
import threading
import time

# Configuration constants
STORAGE_VOLUME_NAME = "jupyter_kernel"
LOGDIR = f"/{STORAGE_VOLUME_NAME}"

app = modal.App("tensorboard-server")

# Volume to share logs with training 
volume = modal.Volume.from_name(STORAGE_VOLUME_NAME, create_if_missing=False)

# Image with TensorBoard dependencies
tensorboard_image = modal.Image.debian_slim().pip_install([
    "tensorboard",
    "tensorflow",
    "protobuf",
])

# TensorBoard must NOT read directly from the volume mount: it keeps event files open,
# and Modal refuses to reload a volume while files on it are open -> logs never update.
# Instead we mirror the event files to local disk and let TensorBoard read the mirror.
MIRROR_DIR = "/tmp/tb_mirror"
RELOAD_MIN_INTERVAL = 10  # seconds; each page load triggers ~30 requests, don't reload on every one

_sync_lock = threading.Lock()
_last_reload = 0.0


def sync_logdir(logdir_param):
    """Reload the volume (throttled) and append new event-file bytes to the local mirror."""
    global _last_reload
    src_root = os.path.join(LOGDIR, logdir_param)
    dst_root = os.path.join(MIRROR_DIR, logdir_param.lstrip("/"))
    with _sync_lock:
        if time.time() - _last_reload >= RELOAD_MIN_INTERVAL:
            try:
                volume.reload()
            except Exception as e:
                print(f"Volume reload failed: {e}")
            _last_reload = time.time()

        if not os.path.exists(src_root):
            print(f"WARNING: Directory {src_root} does not exist!")
            return dst_root
        for dirpath, _, filenames in os.walk(src_root):
            for name in filenames:
                if "tfevents" not in name:
                    continue  # skip checkpoints etc.
                src = os.path.join(dirpath, name)
                dst = os.path.join(dst_root, os.path.relpath(src, src_root))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                # Event files are append-only: copy only the new tail so TensorBoard's
                # open file handles stay valid and it picks up the new records.
                have = os.path.getsize(dst) if os.path.exists(dst) else 0
                if os.path.getsize(src) <= have:
                    continue
                with open(src, "rb") as fsrc, open(dst, "ab") as fdst:
                    fsrc.seek(have)
                    shutil.copyfileobj(fsrc, fdst)
    return dst_root


class SyncMiddleware:
    def __init__(self, app, logdir_param):
        self.app = app
        self.logdir_param = logdir_param

    def __call__(self, environ, start_response):
        sync_logdir(self.logdir_param)
        return self.app(environ, start_response)

@app.function(
    image=tensorboard_image,
    volumes={LOGDIR: volume},
    scaledown_window=600,  # 10 minutes
    max_containers=1,
)
@modal.concurrent(max_inputs=100)  # High limit needed: each TensorBoard page load generates ~30-40 HTTP requests (assets + API calls)
@modal.wsgi_app()
def tensorboard_app():
    import tensorboard
    from urllib.parse import parse_qs
    
    # Cache TensorBoard instances
    tensorboard_cache = {}
    
    def get_tensorboard_app(logdir_param):
        if logdir_param not in tensorboard_cache:
            mirror_logdir = sync_logdir(logdir_param)
            print(f"Creating new TensorBoard instance for logdir: {logdir_param} (mirrored at {mirror_logdir})")
            os.makedirs(mirror_logdir, exist_ok=True)

            board = tensorboard.program.TensorBoard()
            board.configure(logdir=mirror_logdir, reload_interval=5)
            (data_provider, deprecated_multiplexer) = board._make_data_provider()
            wsgi_app = tensorboard.backend.application.TensorBoardWSGIApp(
                board.flags,
                board.plugin_loaders,
                data_provider,
                board.assets_zip_provider,
                deprecated_multiplexer,
                experimental_middlewares=[lambda app, p=logdir_param: SyncMiddleware(app, p)],
            )
            tensorboard_cache[logdir_param] = wsgi_app
        return tensorboard_cache[logdir_param]
    
    def create_app(environ, start_response):
        # Parse URL parameters to get logdir
        query_string = environ.get('QUERY_STRING', '')
        params = parse_qs(query_string)
        logdir_param = params.get('logdir', [None])[0]
        
        # URL decode the logdir parameter (fixes %2F -> / conversion)
        if logdir_param:
            from urllib.parse import unquote
            logdir_param = unquote(logdir_param)
        
        # Check if this is an asset or API request (doesn't need logdir in URL)
        path_info = environ.get('PATH_INFO', '/')
        is_asset_or_api = (path_info.startswith('/font-') or 
                          path_info.startswith('/data/') or
                          path_info.startswith('/experiment/') or
                          path_info.endswith('.js') or 
                          path_info.endswith('.css') or 
                          path_info.endswith('.woff2') or 
                          path_info.endswith('.woff') or
                          path_info.endswith('.svg') or
                          path_info.endswith('.png') or
                          path_info.endswith('.ico'))
        
        if not logdir_param:
            if is_asset_or_api:
                # For assets/API requests, try to get logdir from Referer header
                referer = environ.get('HTTP_REFERER', '')
                if 'logdir=' in referer:
                    import re
                    from urllib.parse import unquote
                    match = re.search(r'logdir=([^&]+)', referer)
                    if match:
                        logdir_param = unquote(match.group(1))  # URL decode
                        print(f"Extracted logdir from referer: {logdir_param}")
                
                # Fallback: use the first available TensorBoard or default
                if not logdir_param:
                    if tensorboard_cache:
                        logdir_param = list(tensorboard_cache.keys())[0]
                    else:
                        logdir_param = "train_test_1"  # Default
            else:
                # Return simple landing page for root requests without logdir
                landing_html = """
                <html>
                    <head><title>TensorBoard Server</title></head>
                    <body>
                        <h1>TensorBoard Server</h1>
                        <p>Multi-experiment TensorBoard monitoring server</p>
                        <p><strong>Usage:</strong> <code>&lt;modal-url&gt;/?logdir=path/to/training_dir</code></p>
                    </body>
                </html>
                """
                start_response('200 OK', [('Content-Type', 'text/html')])
                return [landing_html.encode()]
        
        # Get or create TensorBoard app for this logdir
        wsgi_app = get_tensorboard_app(logdir_param)
        return wsgi_app(environ, start_response)
    
    return create_app