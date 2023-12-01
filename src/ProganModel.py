import tensorflow as tf
import numpy as np
import imageio
import os


class ModelProcessor():
    """Класс для работы с моделью Progan-128"""
    latent_dim = int(512)

    def __init__(self, model):
        self.progan = model

    def __interpolate_hypersphere(self, v1, v2, num_steps):
        v1_norm = tf.norm(v1)
        v2_norm = tf.norm(v2)
        v2_normalized = v2 * (v1_norm / v2_norm)
        vectors = []
        for step in range(num_steps):
            interpolated = v1 + (v2_normalized - v1) * step / (num_steps - 1)
            interpolated_norm = tf.norm(interpolated)
            interpolated_normalized = interpolated * \
                (v1_norm / interpolated_norm)
            vectors.append(interpolated_normalized)
        return tf.stack(vectors)

    def interpolate_between_vectors(self, seed=None):
        """Создаёт два фото, переход между ними и возвращает серию изображений перехода."""
        if seed is not None:
            tf.random.set_seed(seed)
        v1 = tf.random.normal([self.latent_dim])
        v2 = tf.random.normal([self.latent_dim])
        vectors = self.__interpolate_hypersphere(v1, v2, 50)
        interpolated_images = self.progan(vectors)['default']
        return interpolated_images

    def animate(self, images):
        """Создаёт gif файл анимации перехода из серии изображений и возвращает путь к файлу."""
        images = np.array(images)
        converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
        if not os.path.exists('./resources'):
            os.makedirs('./resources')
        path = './resources/animation.gif'
        imageio.mimsave(path, converted_images)
        return path
