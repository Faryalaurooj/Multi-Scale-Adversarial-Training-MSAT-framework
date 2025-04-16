import torch
from singan import train as singan_train
from singan import generate as singan_generate

class SinGANAugmentor:
    def __init__(self, config):
        self.config = config

    def train_singan(self, image_path):
        """
        Trains SinGAN on a single image and returns generator.
        """
        Gs, Zs, reals, NoiseAmp = singan_train(self.config, image_path)
        return Gs, Zs, reals, NoiseAmp

    def generate_samples(self, Gs, Zs, reals, NoiseAmp, num_samples=5):
        """
        Generates synthetic images using the trained SinGAN.
        """
        samples = []
        for _ in range(num_samples):
            sample = singan_generate(self.config, Gs, Zs, reals, NoiseAmp)
            samples.append(sample)
        return samples

