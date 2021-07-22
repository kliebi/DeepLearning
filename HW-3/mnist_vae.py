"""
Title: Variational AutoEncoder
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/05/03
Last modified: 2020/05/03
Description: Convolutional Variational AutoEncoder (VAE) trained on MNIST digits.
"""

"""
## Setup
"""
import os
import sys
import h5py

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
from keras.models import load_model
from keras.models import model_from_json

import cv2
from sklearn import manifold
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
matplotlib.use('svg')
import matplotlib.pyplot as plt


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
def plot_latent_space(decoder, name, n=20, figsize=15 ):
    # display a n*n 2D manifold of digits
    digit_size = 28
    scalex = 4.0
    scaley =3.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-3, 3, n)
    grid_y = np.linspace(-1.5, 2.5, n)[::-1]

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
    plt.savefig(name+'.jpg')
    plt.close('all')
    print("Display latent space as a grid of sampled digits - done")
def plot_label_clusters(encoder, data, labels, name):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()
    plt.savefig(name + '.jpg')
    plt.close('all')
    print("Display latent space clusters done")
def store_model(model, name):
    # serialize  Model
    tf.keras.models.save_model(model,name,save_format="tf")

    print("Saved to disk:  ")
    print(name)

def load_model(name):
    # load model
    loaded_model= tf.keras.models.load_model(name)

    print("Loaded model from disk")
    print(name)
    return loaded_model

def prepare_data():
    # prepare and get data
    (x_train, y_train), (x_test, _) = keras.datasets.mnist.load_data()
    mnist_digits = np.concatenate([x_train, x_test], axis=0)
    mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
    x_train = np.expand_dims(x_train, -1).astype("float32") / 255
    return mnist_digits, x_train,y_train



def train_and_store_VAE(n_epochs):
    """
    ## Build the encoder
    """

    latent_dim = 2

    encoder_inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    """
    ## Build the decoder
    """

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    """
    ## Train the VAE
    """
    # prepare and get data
    mnist_digits, x_train,y_train = prepare_data()

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())
    #vae.fit(mnist_digits, epochs=n_epochs, batch_size=128)
    vae.fit(mnist_digits, epochs=n_epochs, batch_size=128)
    # store decoder Model
    store_model(vae.encoder, "vae_encoder")
    store_model(vae.decoder, "vae_decoder")
    #store_model(vae, "vae_model")


    # Display how the latent space clusters different digit classes

    plot_label_clusters(vae.encoder, x_train, y_train, "mnist_vae_label_clusters")

    # Display how the latent space clusters different digit classes

    plot_latent_space(vae.decoder, "mnist_vae_latent_space")

    print('DONE')



def load_and_predict_VAE():
    mnist_digits, x_train,y_train = prepare_data()
    # load full model
    #loaded_model = load_model("vae_model")

    # load encoder
    encoder_model = load_model("vae_encoder")
    # Display how the latent space clusters
    plot_label_clusters(encoder_model, x_train, y_train, 'VAE_mnist_latent_space-clusters_loaded.jpg')
    print('Display latent space clusters with loaded encoder done')

    # load decoder
    decoder_model = load_model("vae_decoder")
    plot_latent_space(decoder_model, 'VAE_mnist_latent_space_loaded.jpg')
    print('Visualization of latent space  with loaded decoder done')

    print('DONE')


train_and_store_VAE(30)
load_and_predict_VAE()










