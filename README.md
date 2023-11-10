# Hook exercise : music tempo detection

This code base is a reproduction of a tempo classification and beat detection model and its training with the provided GTZAN dataset. 

This work is based on two papers from Sebatian Böck et al. :


- Böck, S., Davies, M. E., & Knees, P. (2019, November). Multi-Task Learning of Tempo and Beat: Learning One to Improve the Other. In ISMIR (pp. 486-493).
- Böck, S., & Davies, M. E. (2020, October). Deconstruct, Analyse, Reconstruct: How to improve Tempo, Beat, and Downbeat Estimation. In ISMIR (pp. 574-582).

I chose to work with the Pytorch Lightning framework, benefiting from higher level APIs for training while keeping some flexibility on the dataset and training logic side.

### Organisation of the repository

Short explanation of the different parts of the repo :

- `bock_network.py` : definition of the actual neural network model
- `data.py` : definition of the GTZAN dataset logic, together with Pytorch Lightning's DataModule.
- `lightning_model.py` : the definition of the PytorchLightning Module, containing all the training logoc : optimization, training step, validation step, etc. I appended a method to it `get_tempo()` allowing to use it directly for "standalone" inference.
- `train.py` : main script that starts the training, instantiating modules, datasets, callbacks, etc
- `metrics.py` and `utils.py`: utility functions: losses, audio / feature processing, metrics
- `get_tempo.py` a standalone script for direct inference
- `config` folder regroups different experiment config files
- `checkpoint` folder regroups trained models
- `tb_logs` folder regroups the TensorBoard logs for 4 different experiments

### Test a trained model

Ino order to perform inference, one can use the provided script.

First, install dependencies :

```pip install -r requirements.txt```

Then run the script :

```python get_tempo.py <PATH_TO_AUDIO_FILE>```
