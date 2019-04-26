import argparse
import os
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.python.keras.datasets import fashion_mnist
from model_example.model import create_model
from model_example.utils import root_dir


def train(model_name):
    # load the data
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # create a TensorBoard callback
    model_dir = root_dir(os.path.join('training', model_name))
    tensorboard_callback = TensorBoard(log_dir=model_dir)

    # create a checkpoint callback
    checkpoint_path = root_dir(os.path.join(model_dir, 'weights.ckpt'))
    checkpoint_callback = ModelCheckpoint(checkpoint_path, save_weights_only=True)

    # train the model
    model = create_model()
    model.fit(train_images, train_labels, epochs=5,
              callbacks=[checkpoint_callback, tensorboard_callback],
              validation_data=(test_images, test_labels))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--model-name', help='Model name', required=True)

    args = parser.parse_args()

    train(args.model_name)
