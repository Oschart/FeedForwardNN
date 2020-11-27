# %%
import os
import pathlib
import warnings
from pathlib import Path

import numpy as np
import PIL
import PIL.Image
import plotly.graph_objects as go
import tensorflow as tf
from sklearn.model_selection import train_test_split

import ArtNeuralNetwork as ANN
from ann_utils import (load_dataset, plot_loss_vs_epochs, plot_tuning_results,
                       print_scores, save_arrays)

warnings.filterwarnings('ignore')


dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = 'flower_photos'
# check if dataset folder exists (to save download time)
if os.path.isdir(data_dir) is False:
    tf.keras.utils.get_file(origin=dataset_url,
                            fname='flower_photos',
                            untar=True)
# Image parameters
IMG_HEIGHT = 64
IMG_WIDTH = 64
USE_LOADED_DATASET = True
SEED = 73
RUN_EXP = False

np.random.seed(SEED)

as_gray = False

mode_dir = 'gray' if as_gray else 'rgb'
array_dir = 'array_store/%s' % mode_dir
plot_dir = 'plots/%s' % mode_dir

try:
    os.makedirs(array_dir)
except:
    print('array_store folder already exists')
try:
    os.makedirs(plot_dir)
except:
    print('plots folder already exists')

if USE_LOADED_DATASET and os.path.isfile('%s/Xtr.npy' % array_dir):
    Xtr, ytr = np.load('%s/Xtr.npy' %
                       array_dir), np.load('%s/ytr.npy' % array_dir)
    Xtest, ytest = np.load('%s/Xtest.npy' %
                           array_dir), np.load('%s/ytest.npy' % array_dir)
    categories = np.load('%s/categories.npy' % array_dir)
else:
    Xtr, ytr, Xtest, ytest, categories = load_dataset(
        data_dir, (IMG_HEIGHT, IMG_WIDTH), as_gray=as_gray)
    save_arrays(array_dir, Xtr=Xtr, ytr=ytr, Xtest=Xtest,
                ytest=ytest, categories=categories)
print('Dataset Loading Complete.')

X_train, X_val, y_train, y_val = train_test_split(Xtr, ytr,
                                                  test_size=0.2,
                                                  random_state=SEED)

# Final model hyperparameters
TOPOLOGY = [X_train.shape[1], 64, 32, len(categories)]
BATCH_SIZE = 32
MAX_EPOCHS = 1500
ACTIVATION = 'relu'
LOSS_T = "nll"
OPTIMIZER = 'adam'
BEST_PER_EPOCH = True
L_RATE = 0.05
BETA1 = 0.9
BETA2 = 0.99


def create_standard_ann():
    """Create an ANN with the standard fine-tuned parameters

    Returns:
        [ArtNeuralNetwork]: [a fine-tuned ANN]
    """
    return ANN.ArtNeuralNetwork(topology=TOPOLOGY,
                                batch_size=BATCH_SIZE,
                                max_epochs=MAX_EPOCHS,
                                activation=ACTIVATION,
                                optimizer=OPTIMIZER,
                                learning_rate=L_RATE,
                                beta1=BETA1,
                                beta2=BETA2,
                                loss_t=LOSS_T,
                                X_val=X_val,
                                y_val=y_val,
                                best_per_epoch=BEST_PER_EPOCH,
                                )


def fine_tune_param(**params):
    param_name = list(params.keys())[0]
    values = params[param_name]
    val_accrs = []
    for v in values:
        ann = create_standard_ann()
        param = dict()
        param[param_name] = v
        ann.set_params(**param)
        _, _, val_accr = ann.train(X_train, y_train)
        val_accrs.append(val_accr)
    return values, val_accrs


def run_fine_tune_experiments():
    epochs = list(range(MAX_EPOCHS))

    print("Start NN Architechture fine-tuning experiments...")
    nn_archs = [[64], [64, 32], [64, 32, 16]]
    nn_full_archs = [[X_train.shape[1], *arch,
                      len(categories)] for arch in nn_archs]
    _, val_accrs = fine_tune_param(toplogy=nn_full_archs)
    disp_archs = [str(arch) for arch in nn_archs]
    plot_tuning_results(disp_archs, val_accrs, epochs, "Hidden Layers")

    print("Start Loss Function fine-tuning experiments...")
    _, val_accrs = fine_tune_param(loss_t=['hinge', 'nll'])
    plot_tuning_results(['Hinge', 'NLL'], val_accrs, epochs, "Loss Function")

    print("Start Activation Function fine-tuning experiments...")
    _, val_accrs = fine_tune_param(activation=['relu', 'sigmoid'])
    plot_tuning_results(['ReLU', 'Sigmoid'], val_accrs,
                        epochs, "Activation Function")

    print("Start Optimizer fine-tuning experiments...")
    _, val_accrs = fine_tune_param(optimizer=['basic', 'adam'])
    plot_tuning_results(['Basic', 'Adam'], val_accrs, epochs, "Optimizer")

    print("Start Learning Rate fine-tuning experiments...")
    learning_rates = [0.0001, 0.001, 0.01, 0.05, 0.1]
    _, val_accrs = fine_tune_param(learning_rate=learning_rates)
    plot_tuning_results(learning_rates, val_accrs, epochs, "Learning Rate")

    print("Start Beta1 fine-tuning experiments...")
    beta1s = [0.6, 0.7, 0.8, 0.9, 0.95]
    _, val_accrs = fine_tune_param(beta1=beta1s)
    plot_tuning_results(beta1s, val_accrs, epochs, "Beta1")

    print("Start Beta2 fine-tuning experiments...")
    beta2s = [0.9, 0.93, 0.96, 0.99, 0.999]
    _, val_accrs = fine_tune_param(beta2=beta2s)
    plot_tuning_results(beta2s, val_accrs, epochs, "Beta2")


def main():

    print("Start Main Model Training...")
    ann = create_standard_ann()
    loss_train, loss_val, _ = ann.train(X_train, y_train)
    print("Main Model Finished.")
    plot_loss_vs_epochs(loss_train, loss_val, list(
        range(MAX_EPOCHS)), title="Loss vs. Epochs")
    Ypred = ann.predict(Xtest)
    CRRns, ACCR = ann.detailed_score(Ypred, ytest)
    print("Testing set results:")
    print_scores(CRRns, ACCR, categories)

    if RUN_EXP:
        run_fine_tune_experiments()

    return


# %%
print('Start run for RGB mode:')
main()
