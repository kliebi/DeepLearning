"""
## Setup
"""

import numpy as np
from keras import regularizers
from tensorflow import keras
import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
#from tensorflow.keras.models import Model
from keras.models import Model
from keras.layers import Input, Reshape, Dense, Flatten
from keras.models import load_model
from keras.models import model_from_json

def preprocess(array):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """

    array = array.astype("float32") / 255.0
    array = np.reshape(array, (len(array), 28, 28, 1))
    return array
def display(array1, array2, name='temp'):
    """
    Displays ten random images from each one of the supplied arrays.
    """

    n = 10

    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]

    plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()
    plt.savefig(name+'.jpg')
    plt.close('all')
class Autoencoder:
  def __init__(self, img_shape=(28, 28, 1 ), latent_dim=2, n_layers=3, n_units=128):
    if not img_shape: raise Exception('Please provide img_shape')

    # create the encoder
    i = h = Input(img_shape) # the encoder takes as input images
    h = Flatten()(h) # flatten the image into a 1D vector
    for _ in range(n_layers): # add the "hidden" layers
      h = Dense(n_units, activation="relu")(h) # add the units in the ith hidden layer
      n_units = n_units/2
    o = Dense(latent_dim,activation="relu")(h) # this layer results in latent space with 2 dimensional size
    self.encoder = Model(inputs=[i], outputs=[o])

    # create the decoder
    i = h = Input((latent_dim,)) # the decoder takes as input lower dimensional vectors

    for _ in range(n_layers): # add the "hidden" layers
      h = Dense(n_units,activation="relu")(h) # add the units in the ith hidden layer
      n_units = n_units*2
    h = Dense(img_shape[0] * img_shape[1], activation='sigmoid')(h) # one unit per pixel in inputs
    o = Reshape(img_shape)(h) # create outputs with the shape of input images
    self.decoder = Model(inputs=[i], outputs=[o])

    # combine the encoder and decoder into a full autoencoder
    i = Input(img_shape) # take as input image vectors
    z = self.encoder(i) # push observations into latent space
    o = self.decoder(z) # project from latent space to feature space
    self.model = Model(inputs=[i], outputs=[o])
    self.model.compile(loss='mse', optimizer='adam')
def plot_latent_space(decoder, name, n=20, figsize=15):
    # display a n*n 2D manifold of digits
    digit_size = 28
    space =8

    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-2, 60, n)
    grid_y = np.linspace(-2, 60, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()
    plt.savefig(name)
    plt.close('all')
def plot_label_clusters(encoder,name, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean = encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()
    plt.savefig(name)
    plt.close('all')
def plot_loss(h, n_epochs):
    N = np.arange(0, n_epochs)
    plt.figure()
    plt.plot(N, h.history['loss'], label='train_loss')
    #plt.plot(N, h.history['val_loss'], label='val_loss')
    plt.title('Training Loss and Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.legend(loc='upper right')
    plt.savefig('mnist_ae_latent_space_Loss2.jpg')
    plt.close('all')
def store_model(model, name):
    # serialize  Model
    model_json = model.to_json()
    with open(name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights of  Model to HDF5
    model.save_weights(name + "_weights.h5")
    print("Saved to disk:  ")
    print(name)
def load_model(name):
    # load json and create model
    json_file = open(name+".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(name +"_weights.h5")
    print("Loaded model from disk")
    print(name)
    return loaded_model
def prepare_data():
    # Since we only need images from the dataset to encode and decode, we
    # won't use the labels.
    (train_data, y_train), (test_data, y_test) = mnist.load_data()

    # Normalize and reshape the data
    train_data = preprocess(train_data)
    test_data = preprocess(test_data)
    mnist_digits = np.concatenate([train_data, test_data], axis=0)
    return train_data, y_train, test_data, y_test, mnist_digits



def train_and_store_AE(n_epochs):
    train_data, y_train, test_data, y_test, mnist_digits = prepare_data()

    """
    Now we can train our autoencoder using `mnist_data` as both our input data
    and target. 
    """
    autoencoder = Autoencoder()
    h = autoencoder.model.fit(
        x=mnist_digits,
        y=mnist_digits,
        batch_size=64,
        epochs=n_epochs,  # 50
        shuffle=True,
        # validation_data=(normalized_test_data, normalized_test_data)
    )
    plot_loss(h, n_epochs)

    # store decoder Model
    store_model(autoencoder.encoder, "ae_encoder")
    store_model(autoencoder.decoder, "ae_decoder")
    store_model(autoencoder.model, "ae_model")

    # do predictions with train data to visualize model performance
    predictions = autoencoder.model.predict(train_data)
    display(train_data, predictions, name='AE_mnist-train_data-predictions')
    # plot latent space
    plot_latent_space(autoencoder.decoder, 'AE_mnist_latent_space.jpg')

    # Display how the latent space clusters
    plot_label_clusters(autoencoder.encoder, 'AE_mnist_latent_space-clusters.jpg', train_data, y_train)


def load_and_predict_AE():
    train_data, y_train, test_data, y_test, mnist_digits = prepare_data()
    # load full model
    loaded_model = load_model("ae_model")
    y = loaded_model.predict(mnist_digits)
    display(mnist_digits, y, name='AE_mnist-train_data-predictions_loaded')
    print('Display prediction with train data with loaded model done')

    # load encoder
    encoder_model = load_model("ae_encoder")
    # Display how the latent space clusters
    plot_label_clusters(encoder_model, 'AE_mnist_latent_space-clusters_loaded.jpg', train_data,
                        y_train)
    print('Display latent space clusters with loaded encoder done')

    # load decoder
    decoder_model = load_model("ae_decoder")
    plot_latent_space(decoder_model, 'AE_mnist_latent_space_loaded.jpg')
    print('Visualization of latent space  with loaded decoder done')

    print('DONE')

#train_and_store_AE(50)
#load_and_predict_AE()
