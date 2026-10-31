"""Model tests on random tensors, so CI doesn't need the 170 MB CIFAR-10 download.

    pip install pytest && pytest -q
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supervised_autoencoder import PlotUtility, SupervisedAutoencoder, SupervisedAutoencoderTrainer
from unsupervised_autoencoder import CNN_Autoencoder, CNN_AutoencoderTrainer

CLASSES = [5, 0, 6]  # dog, airplane, frog
torch.manual_seed(0)
IMAGES = [torch.rand(3, 32, 32) for _ in range(12)]
LABELS = [CLASSES[i % 3] for i in range(12)]


def test_unsupervised_rebuilds_the_input_shape():
    model = CNN_Autoencoder(3)
    batch = torch.stack(IMAGES)
    assert model(batch).shape == batch.shape
    assert model.encode(batch).shape == (12, 2)  # the 2-D embedding that gets plotted


def test_supervised_returns_reconstruction_and_class_scores():
    model = SupervisedAutoencoder(3)
    decoded, scores = model(torch.stack(IMAGES))
    assert decoded.shape == (12, 3, 32, 32)
    assert scores.shape == (12, 3)


def test_both_trainers_reduce_the_loss():
    for model, trainer_cls in [(CNN_Autoencoder(3), CNN_AutoencoderTrainer),
                               (SupervisedAutoencoder(3), SupervisedAutoencoderTrainer)]:
        trainer = trainer_cls(model, IMAGES, LABELS, CLASSES)
        before = _loss(model, trainer)
        trainer.train(epochs=30)
        assert _loss(model, trainer) < before


def test_embedding_plot_is_saved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    PlotUtility.plot_embeddings(np.random.rand(13, 2), LABELS + ["new images"], names, CLASSES)
    assert (tmp_path / "supervised_embeddings.png").stat().st_size > 0


def _loss(model, trainer):
    with torch.no_grad():
        out = model(trainer.selected_images)
        decoded = out[0] if isinstance(out, tuple) else out
        return trainer.criterion_ae(decoded, trainer.selected_images).item()
