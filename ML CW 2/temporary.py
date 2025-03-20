# Dependencies cell.
import re
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import f
import random
import os
import pandas as pd
from pandas.core.base import NoNewAttributesMixin
from pandas.core.missing import clean_interp_method
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.applications import VGG16
from keras_tuner import HyperModel, Hyperband
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import InputSpec


# Pre-processing: dataset loading.

## Load and combine subsets to produce full dataset.
dset1 = np.load("subset_1.npy")
dset2 = np.load("subset_2.npy")
dset3 = np.load("subset_3.npy")
full = np.concatenate((dset1, dset2, dset3), axis=0)

## Obtain reconstructed colour image representations for convolutional models.
full_conv = np.reshape(full, (1196, 150, 225, 3))

# Pre-processing: normalise pixel values.
full_conv_normalised = (full_conv.astype('float32') / 127.5) - 1
full_conv_normalised_relu = full_conv.astype('float32') / 255

# Constants: flattened and standard image shapes.
flattened_shape = full.shape[1:]
image_shape = full_conv.shape[1:]

# Utilities: loss functions.
def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def kl_loss(z_log_var, z_mean):
    return -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))

def perceptual_loss(y, y_pred, perceptual_feature_extractor):
    true_features = perceptual_feature_extractor(y)
    pred_features = perceptual_feature_extractor(y_pred)
    
    loss = 0.0
    for true_feature, pred_feature in zip(true_features, pred_features):
        # Normalize features
        true_feature = tf.nn.l2_normalize(tf.reshape(true_feature, [tf.shape(true_feature)[0], -1]), axis=1)
        pred_feature = tf.nn.l2_normalize(tf.reshape(pred_feature, [tf.shape(pred_feature)[0], -1]), axis=1)
        
        # Compute mean squared error
        loss += tf.reduce_mean(tf.square(true_feature - pred_feature))
    
    return loss

def mse_loss(y, y_pred):
    return tf.keras.losses.MeanSquaredError()(y, y_pred)

def mae_loss(y, y_pred):
    return tf.keras.losses.MeanAbsoluteError()(y, y_pred)

def huber_loss(y, y_pred):
    return tf.keras.losses.Huber()(y, y_pred)


# Models: deep convolutional autoencoder cell with tunable hyperparameters.
class HyperCAE(HyperModel):
    def __init__(self, shape, loss="mse", downsampling="max_pool"):
        self.shape = shape
        self.loss = loss
        self.downsampling = downsampling

    def build(self, hp):
        encoder_layers = []
        encoder_layers.append(layers.ZeroPadding2D(padding=((1, 1), (3, 4)), input_shape=self.shape))
        
        num_layers = hp.Choice('num_conv_layers', values=[1, 2, 3])
        filters = []

        for i in range(num_layers):
            filters.append(hp.Choice(f'filters_{i + 1}', values=[16, 32, 64]))

        for i in range(num_layers):
            if self.downsampling == "max_pool":
                encoder_layers.append(layers.Conv2D(filters[i], (3, 3), activation='leaky_relu', padding='same', name=f'conv{i+1}', kernel_initializer=HeNormal()))
                encoder_layers.append(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
            elif self.downsampling == "strides":
                encoder_layers.append(layers.Conv2D(filters[i], (3, 3), activation='leaky_relu', padding='same', strides=2, name=f'conv_{i+1}_strides',kernel_initializer=HeNormal())) 
        
        self.encoder = tf.keras.Sequential(encoder_layers)
        
        bottleneck_filters = hp.Choice('bottleneck_filters', values=[32, 64, 128])
        self.bottleneck = layers.Conv2D(bottleneck_filters, (3, 3), padding='same', name='bottleneck')
        
        decoder_layers = []
        
        for i in range(num_layers):
            decoder_layers.append(layers.UpSampling2D((2, 2)))  
            decoder_layers.append(
                layers.Conv2D(filters[i], (3, 3), activation='leaky_relu', padding='same', name=f'dec_conv_{i+1}', kernel_initializer=HeNormal())
            )
        
        decoder_layers.append(
            layers.Conv2D(3, (3, 3), activation='tanh', padding='same')
        ) 

        decoder_layers.append(layers.Cropping2D(cropping=((1, 1), (3, 4))))

        self.decoder = tf.keras.Sequential(decoder_layers)
        
        self.model = tf.keras.Sequential([
            self.encoder,
            self.bottleneck,
            self.decoder
        ])

        opt = tf.keras.optimizers.Adam()

        match self.loss:
            case "huber":
                self.loss = tf.keras.losses.Huber()
            case "mse":
                self.loss = tf.keras.losses.MeanSquaredError()
            case "mae":
                self.loss = tf.keras.losses.MeanAbsoluteError()

        self.model.compile(optimizer=opt, loss=self.loss)
        
        return self.model
    
    def call(self, x):
        return self.model(x)

class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        elif isinstance(padding, tuple) and len(padding) == 2:
            if isinstance(padding[0], int) and isinstance(padding[1], int):
                self.padding = ((padding[0], padding[0]), (padding[1], padding[1]))
            elif isinstance(padding[0], tuple) and isinstance(padding[1], tuple):
                self.padding = padding
            else:
                raise ValueError("Invalid padding format")
        else:
            raise ValueError("`padding` should be an int, a tuple of 2 ints, or a tuple of 2 tuples of 2 ints.")

    def compute_output_shape(self, input_shape):
        return (input_shape[0],
                input_shape[1] + self.padding[0][0] + self.padding[0][1],
                input_shape[2] + self.padding[1][0] + self.padding[1][1],
                input_shape[3])

    def call(self, inputs):
        paddings = [[0, 0],  # Batch dimension: no padding
                    [self.padding[0][0], self.padding[0][1]],  # Height dimension
                    [self.padding[1][0], self.padding[1][1]],  # Width dimension
                    [0, 0]]  # Channels dimension: no padding
        return tf.pad(inputs, paddings=paddings, mode='REFLECT')

    def get_config(self):
        config = super().get_config()
        config.update({'padding': self.padding})
        return config


class InertLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(InertLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs

# Models: standard deep convolutional autoencoder (without hyperparameter optimisation) cell.
class CAE(Model):
    def __init__(self, shape, loss="mse", downsampling="max_pool", padding='zero', bottleneck_filters=16):
        super().__init__()
        self.shape = shape
        self.loss = loss
        self.padding = padding
        self.bottleneck_filters = bottleneck_filters
        self.initial_padding_layer = None
        self.intermediate_padding = None
        
        if self.padding == 'zero':
            self.initial_padding_layer = layers.ZeroPadding2D(padding=((1, 1), (3, 4)), input_shape=self.shape)
            self.intermediate_padding = 'same'
            self.extra_padding_layer = InertLayer()
        elif self.padding == 'replication':
            self.initial_padding_layer = ReflectionPadding2D(padding=((1, 1), (3, 4)), input_shape=self.shape)
            self.intermediate_padding = 'valid'
            self.extra_padding_layer = ReflectionPadding2D(padding=(1, 1))
 
        if downsampling == "max_pool":
            self.encoder = tf.keras.Sequential([
                self.initial_padding_layer,
                layers.Conv2D(64, (3, 3), activation='relu', padding=self.intermediate_padding, name='conv1'),
                self.extra_padding_layer,
                layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding=self.intermediate_padding),
                layers.BatchNormalization(),
                layers.Conv2D(32, (3, 3), activation='relu', padding=self.intermediate_padding, name='conv2'),
                self.extra_padding_layer,
                layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding=self.intermediate_padding),
                layers.BatchNormalization(),
                layers.Conv2D(self.bottleneck_filters, (3, 3), activation='relu', padding='same', name='bottleneck')
            ])
        
        elif downsampling == "strides":
            self.encoder = tf.keras.Sequential([
                self.initial_padding_layer,
                layers.Conv2D(32, (3, 3), activation='relu', padding=self.intermediate_padding, name='conv1'),
                layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding=self.intermediate_padding),  
                layers.BatchNormalization(),
                layers.Conv2D(16, (3, 3), activation='relu', padding=self.intermediate_padding, name='conv2'),
                layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding=self.intermediate_padding),
                layers.BatchNormalization(),
                layers.Conv2D(self.bottleneck_filters, (3, 3), activation='relu', padding=self.intermediate_padding, name='bottleneck')
            ])

        self.decoder = tf.keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', padding=self.intermediate_padding),
            self.extra_padding_layer,
            layers.UpSampling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding=self.intermediate_padding),
            self.extra_padding_layer,
            layers.UpSampling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Conv2D(3, (3, 3), activation='sigmoid', padding=self.intermediate_padding),
            self.extra_padding_layer,
            layers.Cropping2D(cropping=((1, 1), (3, 4)))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def compute_loss(self, x, y, y_pred, sample_weight=None):
        match self.loss:
            case "mse":
                return mse_loss(y, y_pred)
            case "mae":
                return mae_loss(y, y_pred)
            case 'ssim':
                return ssim_loss(y, y_pred)
            case 'huber':
                return huber_loss(y, y_pred)
 
    def train_step(self, data):
        x, y = data
        
        print("Received input x: ", x)
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            print("Y predicted: ", y_pred)
            loss = self.compute_loss(x, y, y_pred)
            print("Obtained loss: ", loss)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        
        return {m.name: m.result() for m in self.metrics}


class Sampling(layers.Layer):
    def call(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class AdvancedVEncoder(Model):
    def __init__(self, shape, latent_dim):
        super().__init__()
        self.shape = shape
        self.latent_dim = latent_dim

        self.convolutions = tf.keras.Sequential([
            #layers.ReplicationPadding2D(padding=(2, 7), input_shape=(150, 225, 3)),
            layers.ZeroPadding2D(padding=((1, 1), (3, 4)), input_shape=self.shape),
            layers.Conv2D(128, (3, 3), activation='leaky_relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),  
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='leaky_relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),  
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='leaky_relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),  
            layers.BatchNormalization(),
            layers.Flatten()
        ])
        
        self.z_mean = layers.Dense(latent_dim, name='z_mean')
        self.z_log_var = layers.Dense(latent_dim, name='z_log_var')
        self.sampling = Sampling()

    def call(self, x):
        conv = self.convolutions(x)
        z_mean = self.z_mean(conv)
        z_log_var = self.z_log_var(conv)
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z

class AdvancedVDecoder(Model):
    def __init__(self, shape, latent_dim):
        super().__init__()
        self.shape = shape
        self.latent_dim = latent_dim

        self.decoder = tf.keras.Sequential([
            layers.Dense(19 * 29 * 32, activation='leaky_relu'),
            layers.Reshape((19, 29, 32)),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(64, 3, activation='leaky_relu', padding='same'),
            layers.BatchNormalization(),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(32, 3, activation='leaky_relu', padding='same'),
            layers.BatchNormalization(),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(3, 3, activation='tanh', padding='same'), 
            layers.Cropping2D(cropping=((1, 1), (3, 4)))
        ])

    def call(self, x):
        return self.decoder(x)


class VAE(Model):
    def __init__(self, shape, alpha=1, beta=1, advanced=True, latent_dim=256, loss='mse'):
        super().__init__()
        self.advanced = advanced

        self.encoder = AdvancedVEncoder(shape, latent_dim)
        self.decoder = AdvancedVDecoder(shape, latent_dim)
        self.perceptual_model = VGG16(weights='imagenet', include_top=False, input_shape=shape)
        self.perceptual_layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']
        self.perceptual_model_outputs = [self.perceptual_model.get_layer(name).output for name in self.perceptual_layer_names]
        self.perceptual_feature_extractor = Model(inputs=self.perceptual_model.input, outputs=self.perceptual_model_outputs)
        self.perceptual_feature_extractor.trainable = False

        self.z_mean = None
        self.z_log_var = None
        self.reconstruction_losses = []

        self.alpha = alpha
        self.beta = beta

        self.reconstruction_loss = loss
        self.kl_loss = kl_loss
        
    def call(self, inputs):
        self.z_mean, self.z_log_var, z = self.encoder(inputs)
        decoded = self.decoder(z)
        return decoded

    def compute_loss(self, x, y, y_pred, sample_weight=None):
        reconstruction_loss = 0.0

        match self.reconstruction_loss:
            case 'perceptual':
                reconstruction_loss = perceptual_loss(y, y_pred, self.perceptual_feature_extractor)
            case 'mse':
                reconstruction_loss = mse_loss(y, y_pred)
            case 'mae':
                reconstruction_loss = mae_loss(y, y_pred)
            case 'huber':
                reconstruction_loss = huber_loss(y, y_pred)

        kl_loss = self.kl_loss(self.z_log_var, self.z_mean)

        print("Obtained reconstruction loss: ", reconstruction_loss)
        
        # TO DO: Automatically optimise alpha and beta.
        return (self.alpha * reconstruction_loss) + (self.beta * kl_loss)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        
        return {m.name: m.result() for m in self.metrics}

# Experiment: simple train-test split to test number of epochs needed for convergence.
'''
X_train, X_test = train_test_split(full_conv_normalised, test_size=0.2, random_state=42)

cae_model = CAE(shape=image_shape, loss='huber', downsampling='max_pool')

cae_model.compile(optimizer='adam')

# History will store the training loss for each epoch
history = cae_model.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test))

# Plot the training and validation loss to observe convergence
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
'''

# Post-processing: rescaling.
def rescale_image(image):
    return np.uint8(((image + 1) / 2) * 255)

def get_compression_ratio(encoder):
    # Extract the bottleneck layer
    bottleneck_layer = encoder.get_layer('bottleneck')
    
    # Get the output shape of the bottleneck layer
    bottleneck_shape = bottleneck_layer.output_shape  
    
    # Compute the size of the bottleneck layer (channels * height * width)
    bottleneck_size = bottleneck_shape[1] * bottleneck_shape[2] * bottleneck_shape[3]
    
    # Get the input shape of the encoder
    input_shape = encoder.input_shape  # (batch_size, height, width, channels)
    
    # Compute the size of the input image (channels * height * width)
    input_size = input_shape[1] * input_shape[2] * input_shape[3]
    
    # Calculate the compression ratio
    compression_ratio = input_size / bottleneck_size
    
    return compression_ratio

# Experiment: train four models (MSE, MAE, Huber, SSIM) and compare reconstruction quality on 1 image.
X_train, X_test = train_test_split(full_conv_normalised_relu, test_size=0.2, random_state=42)
X_test, X_val = train_test_split(X_test, test_size=0.1, random_state=42)

# losses = ['huber', 'mse', 'ssim']
# paddings = ['zero', 'replication']
bottleneck_filters = [8, 16, 32]
models = []

#for loss in losses:
# Replace with get_best_cae().
for filter in bottleneck_filters:
    cae = CAE(shape=image_shape, loss='huber', padding='zero', bottleneck_filters=filter)
    cae.compile(optimizer='adam')
    cae.encoder.summary()
    cae.decoder.summary()
    history = cae.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test))
    models.append(cae)

plt.figure(figsize=(12, 8))

original_image = full_conv[15]

# Display the original image in the first position
plt.subplot(2, 3, 1)  # 2 rows, 3 columns, first position
plt.imshow(original_image.reshape(image_shape))  
plt.title("Original Image")
plt.axis('off')

# Display the reconstructed images for each model in the next subplots
for i, model in enumerate(models):
    reconstructed_image = model.predict(full_conv_normalised_relu[15].reshape(1, *image_shape))  # Make prediction for the test image
    plt.subplot(2, 3, i + 2)  # Adjusting position for each reconstructed image (2 rows, 3 columns)
    #reconstructed_image = rescale_image(reconstructed_image)
    plt.imshow(reconstructed_image.reshape(image_shape))  
    plt.title(f"Reconstructed with {bottleneck_filters[i]} bottleneck filters\nCompression Ratio = {get_compression_ratio(model.encoder)} ")
    plt.axis('off')

plt.tight_layout()
plt.show()


'''
# Assuming model is already defined
model = models[0]
test_indices = [1, 15, 50, 72]

# Loss function - You can use any loss function that's available in your model (e.g., MSE)
for i, index in enumerate(test_indices):
    # Fetch the original test image and the reconstructed image
    test_image = full_conv[index]
    reconstructed_image = model.predict(full_conv_normalised_relu[index].reshape(1, *image_shape))
    
    # Evaluate the model using model.evaluate()
    # Reshape both test_image and reconstructed_image to ensure they match the expected input/output shape
    test_image_reshaped = test_image.reshape(1, *image_shape)  # Shape for evaluation
    reconstructed_image_reshaped = reconstructed_image.reshape(1, *image_shape)  # Shape for evaluation

    # Evaluate model on the pair (original and prediction)
    loss = model.evaluate(test_image_reshaped, reconstructed_image_reshaped, verbose=0)  # We set verbose=0 to suppress the output
    
    # Plot original image in the first row
    plt.subplot(2, len(test_indices), i + 1)  # 2 rows, len(test_indices) columns, original image
    plt.imshow(test_image.reshape(image_shape))
    plt.title(f"Original image {i}")
    plt.axis('off')

    # Plot reconstructed image in the second row
    plt.subplot(2, len(test_indices), len(test_indices) + i + 1)  # 2 rows, len(test_indices) columns, reconstructed image
    plt.imshow(reconstructed_image.reshape(image_shape))
    plt.title(f"Reconstructed image {i}\nSSIM Loss: {loss:.4f}")
    plt.axis('off')

# Adjust layout for better display
plt.tight_layout()
plt.show()
'''

# Experiment: Plot MSE loss distribution on random sample for outlier detection
'''
def plot_loss_distribution():
    X_train, X_test = train_test_split(full_conv_normalised, test_size=0.2, random_state=42)

    cae_model = CAE(shape=image_shape, loss='mse', downsampling='max_pool')
    cae_model.compile(optimizer='adam')
    cae_model.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test))

    num_samples = 200
    random_indices = np.random.choice(X_test.shape[0], num_samples, replace=False)
    random_images = X_test[random_indices]

    # Calculate MSE for each image in the random sample
    mse_values = []

    for image in random_images:
        reconstructed_image = cae_model.predict(image.reshape(1, *image_shape))  # Use MSE model (index 1 in the models list)
        mse_value = tf.keras.losses.MeanSquaredError()(image, reconstructed_image.reshape(image_shape))
        mse_values.append(mse_value)

    # Plot the MSE distribution
    plt.figure(figsize=(8, 6))
    plt.hist(mse_values, bins=20, edgecolor='black')
    plt.title('MSE Loss Distribution for Random Sample')
    plt.xlabel('MSE')
    plt.ylabel('Frequency')
    plt.show()

    # Optional: Identify potential outliers
    threshold = np.percentile(mse_values, 95)  # Use the 95th percentile as an outlier threshold
    outliers = np.array(random_images)[np.array(mse_values) > threshold]

    # Plot outliers (if any)
    if len(outliers) > 0:
        plt.figure(figsize=(12, 8))
        for i, outlier in enumerate(outliers):
            plt.subplot(1, len(outliers), i + 1)
            plt.imshow(outlier.reshape(image_shape))
            plt.title(f"Outlier {i+1}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print("No outliers detected based on MSE threshold.")

'''

'''
# Training: KerasTuner for deep CAE. 
def get_best_model(loss='huber'):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    fold_test_losses = []
    fold_best_models = []  

    for train_index, test_index in kf.split(full_conv_normalised):
        X_train, X_test = full_conv_normalised[train_index], full_conv_normalised[test_index]

        # Define a KerasTuner model
        cae_model = HyperVAE(shape=image_shape, loss=loss, downsampling="max_pool")

        # Inner folds.
        inner_kf = KFold(n_splits=3, shuffle=True, random_state=42)

        # Define a RandomSearch tuner for hyperparameter optimization
        tuner = Hyperband(
            cae_model,
            objective='val_loss',
            max_epochs=10,
            factor=5,
            hyperband_iterations=1
        )

        best_val_loss = float("inf")
        best_model = None

        for inner_train_index, inner_test_index in inner_kf.split(X_train):
            inner_X_train, inner_X_test = X_train[inner_train_index], X_train[inner_test_index]
            tuner.search(inner_X_train, inner_X_train, epochs=10, validation_data=(inner_X_test, inner_X_test))
            best_inner_model = tuner.get_best_models(num_models=1)[0]
            val_loss = best_inner_model.evaluate(inner_X_test, inner_X_test, verbose=0)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = best_inner_model
        
        test_loss = best_model.evaluate(X_test, X_test, verbose=0)
        fold_test_losses.append(test_loss)  # Store the validation loss for this fold
        fold_best_models.append(best_model)  # Store the best model for this fold

    # Calculate the average validation loss across all folds
    average_val_loss = np.mean(fold_test_losses)
    std_val_loss = np.std(fold_test_losses)
    print(f"Average Validation Loss: {average_val_loss}")

    # Select the best model (with the lowest validation loss) across all folds
    best_fold_index = np.argmin(fold_test_losses)
    best_model = fold_best_models[best_fold_index]

    # Print the summary of the best model
    # print("Best model selected from fold", best_fold_index + 1)
    # best_model.summary()

    # Retrain the best model on the entire dataset
    X_train, X_test = train_test_split(full_conv_normalised, test_size=0.2, random_state=42)
    best_model.fit(X_train, X_train, batch_size=32, validation_data=(X_test, X_test))
    
    
    best_model.summary()
    for layer in best_model.layers:
        if isinstance(layer, tf.keras.Sequential):
            layer.summary()
    

get_best_model()
'''

# Experiment: feature map comparison cell between stride-based and pooling-based downsampling.








