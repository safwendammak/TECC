import logging, os, random, argparse
from utils.tf_utils import set_tf_loglevel, str2bool

set_tf_loglevel(logging.ERROR)

import json
import numpy as np
import tensorflow as tf

from preprocessing.preprocess import preprocess
from preprocessing.sub_preprocess import preprocess_sub_disease
from utils.dataloader import DataGen
from utils.callbacks import model_checkpoint

# Random seed
seed = 42
random.seed(seed)
np.random.seed(seed)

def train(
    model,
    path: str = "data/ptb",
    batch_size: int = 32, # Reduce this to 16 if there is any memory problem
    epochs: int = 60, 
    loggr=None,
    name: str = "imle_net",
) -> None:
    """Data preprocessing and training of the model.

    Parameters
    ----------
    model: tf.keras.Model
        Model to be trained.
    path: str, optional
        Path to the directory containing the data. (default: 'data/ptb')
    batch_size: int, optional
        Batch size. (default: 32)
    epochs: int, optional
        Number of epochs. (default: 60)
    loggr: wandb, optional
        To log wandb metrics. (default: None)
    name: str, optional
        Name of the model. (default: 'imle_net')


    """

    metric = "val_auc"
    checkpoint_filepath = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(checkpoint_filepath, exist_ok=True)

    X_train_scale, y_train, _, _, X_val_scale, y_val = preprocess(path=path)
    train_gen = DataGen(X_train_scale, y_train, batch_size=batch_size)
    val_gen = DataGen(X_val_scale, y_val, batch_size=batch_size)
    checkpoint = model_checkpoint(
        checkpoint_filepath,
        val_gen,
        loggr=loggr,
        monitor=metric,
        name=name,
    )

    # Early Stopping
    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor=metric,
        min_delta=0.0001,
        patience=20,
        mode="max",
        restore_best_weights=True,
        verbose=1,
    )
    callbacks = [checkpoint, stop_early]

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
    )
    logs_path = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_path, exist_ok=True)

    with open(os.path.join(logs_path, f"{name}_train_logs.json"), "w") as json_file:
        json.dump(history.history, json_file)


if __name__ == "__main__":
    """Main function to run the training of the model."""

    # Set the GPU to allocate only used memory at runtime.
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        print(device)
        tf.config.experimental.set_memory_growth(device, True)

    # Args parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", type=int, default=32, help="Batch size (Choose smaller batch size if available GPU memory is less)")
    parser.add_argument("--epochs", type=int, default=60, help="Number of epochs")

    args = parser.parse_args()

    from models.TECC import build_tecc
    from configs.tecc_config import Config
    model = build_tecc(Config())
    train(
        model,
        path='data/ptb',
        batch_size=args.batchsize,
        epochs=args.epochs,
        loggr=None,
        name='tecc'
    )
