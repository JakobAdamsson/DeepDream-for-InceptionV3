import tensorflow as tf
import numpy as np
import matplotlib as mpl
import IPython.display as display
import PIL.Image
from tensorflow.keras.utils import plot_model
from typing import Tuple
import sys
class DeepDream(tf.Module):
    """DeepDream model class"""
    def __init__(self, model, stepsize, steps):
        self.model = model
        self.step_size = stepsize
        self.steps = steps
        self.hej = 1

    def perform_octave_scaling(self, octave_scale: int, img: np.array) -> None:
        """
        Args: [Octave scale: int, img: np.array]

        Description: Performs octave scaling on provided image with a given octave scale.
        Octaves in this case is simply the process of making the image smaller or bigger for each gradient calculated.

        A high octave value, the more dreamy image
        A low octave value, the less dreamy and blurry image

        Returns: None.
        """

        base_shape = tf.shape(img)[:-1]
        float_base_shape = tf.cast(base_shape, tf.float32)
        for i in range(-2, 3):
            new_shape = tf.cast(float_base_shape * (octave_scale ** i), tf.int32)
            img = tf.image.resize(img, new_shape).numpy()
            img = self.run_simple_dream_modification(img)

        #self.show(img)

    def deprocess(self, img: np.array) -> np.array:
        """
        Args: [img: np.array]

        Description: Image is normalized to work for the model, to get the correct print of the image we need to go back to values between [0, 255]

        Returns: np.array(uint8)
        """
        img = 255 * (img + 1.0) / 2.0
        return tf.cast(img, tf.uint8)

    def show(self, img) -> None:
        """
        Args: [img: np.array]

        Description: Displays the given image

        Returns: None
        """
        display.display(PIL.Image.fromarray(np.array(img)))

    def run_simple_dream_modification(self, img: np.array) -> tf.TensorArray:
        """
        Args: [img: np.array]

        Description: Main loop that performs the gradient ascent untill the steps are 0. Important to keep in mind,
        The model expects a tensor, therefor we need to convert the stepsize and the image into a tensor.

        The reason for the if and else statements in the whileloop is to ensure that steps_remaning always is > 0


        """
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        img = tf.convert_to_tensor(img)
        step_size = tf.convert_to_tensor(self.step_size)
        steps_remaining = self.steps
        step = 0
        while steps_remaining:
            if steps_remaining > 100:
                run_steps = tf.constant(100)
            else:
                run_steps = tf.constant(steps_remaining)
            steps_remaining -= run_steps
            step += run_steps

            # This line was missing inside the while loop
            loss, img = self.run_gradient_ascent(img, run_steps, tf.constant(step_size))

        display.clear_output(wait=True)
        self.show(self.deprocess(img))
        print("Step {}, loss {}".format(step, loss))

        result = self.deprocess(img)
        display.clear_output(wait=True)
        self.show(result)

        return result

    def calc_loss(self, img: tf.TensorArray) -> tf.TensorArray:
        """
        Args: [img: tf.TensorArray]

        Description: First an expansion of dimensions is performed, this is to tell the model that we have a batch of one image,
        then a forwardpass is performed, followed by that it Iterates through each Feature window(11x14 pixels) in the image,
        calculating the mean for each window then sums it all up and returns a tensor for all windows.

        Returns tf.TensorArray
        """
        # expand dims, set axis=0 for to convert the 3d tensor into a 4d tensor(tensorflow förväntar sig 4d)
        new_img = tf.expand_dims(img, axis=0)

        # forward pass through the model
        layer_activations = self.model(new_img)

        losses = []

        # Räkna ut loss genom mean för varje filter i bilden
        losses = tf.map_fn(tf.reduce_mean, layer_activations)
        print(tf.reduce_sum(losses))

        return tf.reduce_sum(losses)

    @tf.function(
        input_signature=(
                # Image
                tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
                # steps
                tf.TensorSpec(shape=[], dtype=tf.int32),
                # step_size
                tf.TensorSpec(shape=[], dtype=tf.float32),)
    )
    def run_gradient_ascent(self, img: tf.TensorArray, steps: int, step_size: int) -> Tuple[tf.constant, tf.TensorArray]:
        """
        Args: [img: tf.TensorArray, steps: int, step_size: int]

        Description: Iterates each step in provided as "steps", then using gradienttape we can record how the gradient
        behaves during the iterations. For each gradient, we calculate the loss for that specific gradient.

        Returns: Tuple(tf.constant, tf.TensorArray)
        """
        loss = tf.constant(0.0)
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                tape.watch(img)
                loss = self.calc_loss(img)

            # Calculate the gradient of the loss with respect to the input image.(en tensor med gradienter där vi maximerar loss)
            gradients = tape.gradient(loss, img)

            # Normalisera gradienten
            gradients /= tf.math.reduce_std(gradients) + 1e-8

            # Klipp ut gradienten efter bilden blivit manipulerad för att maximera loss
            img = img + gradients * step_size
            img = tf.clip_by_value(img, -1, 1)

        return loss, img
