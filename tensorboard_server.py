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

# Middleware to reload volume data on requests
class VolumeMiddleware:
    def __init__(self, app, volume, logdir):
        self.app = app
        self.volume = volume
        self.logdir = logdir

    def __call__(self, environ, start_response):
        # Try to reload volume, but don't fail if files are in use
        try:
            print(f"Attempting to reload volume for {self.logdir}...")
            self.volume.reload()
            print(f"Volume reload successful for {self.logdir}")
        except RuntimeError as e:
            if "open files preventing" in str(e):
                print(f"Volume reload skipped (files in use) for {self.logdir}: {str(e)[:100]}...")
                pass
            else:
                print(f"Volume reload failed with unexpected error for {self.logdir}: {e}")
                raise
        except Exception as e:
            print(f"Volume reload failed for {self.logdir}: {e}")
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
            full_logdir = os.path.join(LOGDIR, logdir_param)
            print(f"Creating new TensorBoard instance for logdir: {full_logdir}")
            
            # Check if directory exists and what's in it
            if os.path.exists(full_logdir):
                try:
                    files = os.listdir(full_logdir)
                    print(f"Directory {full_logdir} contains: {files}")
                    # Find event files
                    event_files = [f for f in files if 'tfevents' in f.lower()]
                    print(f"Event files found: {event_files}")
                except Exception as e:
                    print(f"Error listing directory {full_logdir}: {e}")
            else:
                print(f"WARNING: Directory {full_logdir} does not exist!")
            
            board = tensorboard.program.TensorBoard()
            board.configure(logdir=full_logdir)
            (data_provider, deprecated_multiplexer) = board._make_data_provider()
            wsgi_app = tensorboard.backend.application.TensorBoardWSGIApp(
                board.flags,
                board.plugin_loaders,
                data_provider,
                board.assets_zip_provider,
                deprecated_multiplexer,
                experimental_middlewares=[lambda app: VolumeMiddleware(app, volume, logdir_param)],
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