import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import numpy as np
import os

param_model_path = "models/vae/"
param_result_path = "result/vae/"
if not os.path.exists(param_model_path):
    os.makedirs(param_model_path)
if not os.path.exists(param_result_path):
    os.makedirs(param_result_path)

param_img_width, param_img_heigth = 28, 28
param_intermediate_size = 512
param_latent_size = 20

param_batch_size = 512
param_epochs = 100


class Sampling(Layer):
    def __init__(self, **kwargs):
        super(Sampling, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        mu, logvar = inputs
        epsilon = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * logvar) * epsilon


class VAELOSS(Layer):
    def __init__(self, **kwargs):
        super(VAELOSS, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        output, realimg, mu, logvar = inputs

        bceloss = binary_crossentropy(realimg, output)

        klloss = -0.5 * (logvar + 1 - mu ** 2 - tf.exp(logvar))
        klloss = tf.reduce_sum(klloss, axis=-1)

        loss = tf.reduce_mean(param_img_heigth * param_img_width * bceloss + klloss)
        self.add_loss(loss)

        return output


class USER:
    def build_model(self, summary=False, plot=False):
        realimg = Input(shape=[param_img_heigth * param_img_width, ], name="realimg")
        x = Dense(param_intermediate_size, activation="relu", name="interdense")(realimg)
        mu = Dense(param_latent_size, name="mu")(x)
        logvar = Dense(param_latent_size, name="logvar")(x)
        z = Sampling(name="sampling")(inputs=(mu, logvar))

        encoder = Model(inputs=realimg, outputs=[mu, logvar, z], name="encoder")

        latentimg = Input(shape=[param_latent_size, ], name="latentimg")
        x = Dense(param_intermediate_size, activation="relu", name="invinterdense")(latentimg)
        outputimg = Dense(param_img_width * param_img_heigth, activation="sigmoid", name="invimg")(x)

        decoder = Model(inputs=latentimg, outputs=outputimg, name="decoder")

        output1 = decoder(encoder(realimg)[2])
        output = VAELOSS(name="vaeloss")(inputs=(output1, realimg, mu, logvar))

        vae = Model(inputs=realimg, outputs=output, name="vae")

        if summary:
            encoder.summary()
            decoder.summary()
            vae.summary()

        if plot:
            plot_model(encoder, param_model_path + "encoder.png", show_shapes=True)
            plot_model(decoder, param_model_path + "decoder.png", show_shapes=True)
            plot_model(vae, param_model_path + "vae.png", show_shapes=True)

        return encoder, decoder, vae

    def train(self):
        (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

        x_train = np.reshape(x_train, [-1, param_img_width * param_img_heigth])
        x_test = np.reshape(x_test, [-1, param_img_width * param_img_heigth])
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        encoder, decoder, vae = self.build_model(summary=True, plot=True)

        vae.compile(optimizer=Adam(0.01))

        vae.fit(x_train, epochs=param_epochs, batch_size=param_batch_size, validation_data=(x_test, None))

        encoder.save_weights(param_model_path + "encoder.h5")
        decoder.save_weights(param_model_path + "decoder.h5")
        vae.save_weights(param_model_path + "vae.h5")

    def predict(self):
        _, _, vae = self.build_model()
        vae.load_weights(param_model_path + "vae.h5")

        (_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

        x_test = np.reshape(x_test, [-1, param_img_width * param_img_heigth])
        x_test = x_test.astype('float32') / 255

        x_test = x_test[:11]
        y_predict = vae.predict(x_test)

        realimages = np.reshape(x_test * 255, [-1, param_img_width, param_img_heigth, 1])
        fakeimages = np.reshape(y_predict * 255, [-1, param_img_width, param_img_heigth, 1])

        for i in range(11):
            realimg = tf.image.encode_jpeg(realimages[i], quality=100)
            fakeimg = tf.image.encode_jpeg(fakeimages[i], quality=100)

            with tf.io.gfile.GFile(param_result_path + 'realimg' + str(i) + '.jpg', 'wb') as file:
                file.write(realimg.numpy())

            with tf.io.gfile.GFile(param_result_path + 'fakeimg' + str(i) + '.jpg', 'wb') as file:
                file.write(fakeimg.numpy())

    def test(self):
        _, decoder, _ = self.build_model()
        decoder.load_weights(param_model_path + "decoder.h5")

        latentimgs = tf.random.normal([11, param_latent_size])
        fakeimgs = decoder.predict(latentimgs)
        fakeimgs = np.reshape(fakeimgs * 255, [-1, param_img_width, param_img_heigth, 1])

        for i in range(11):
            fakeimg = tf.image.encode_jpeg(fakeimgs[i], quality=100)

            with tf.io.gfile.GFile(param_result_path + 'testfakeimg' + str(i) + '.jpg', 'wb') as file:
                file.write(fakeimg.numpy())


if __name__ == "__main__":
    user = USER()
    user.train()
    user.predict()
    user.test()
