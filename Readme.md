# Run a jupyter notebook kernel on Modal

* install modal: `pip install modal`
* make sure you have a Modal account setup (create and account on [Modal](http://modal.com) and set it up via `modal setup`)

* launch the jupyter kernel on Modal first: `python start_jupyter_kernel.py`
* in your notebook, connect the remote kernel as described
    * see `test_kernel.ipynb` for info on how to connect
* tested in VSCode -- but should run elsewhere as well

## TensorBoard Monitoring

Monitor your training runs with the multi-experiment TensorBoard server:

* **Deploy:** `modal deploy tensorboard_server.py`
* **Usage:** `https://your-modal-url/?logdir=path/to/training_dir`
* **Features:** Single server handles multiple experiments, live monitoring, automatic scaling