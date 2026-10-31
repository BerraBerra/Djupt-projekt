"""Run training with python training.py. Edit settings.py to change the experiment."""
from settings import MANIFEST, OUTPUT, TRAINING
from image_classifier.training import train

if __name__ == '__main__':
    train(MANIFEST, TRAINING, OUTPUT)
