"""Run training with python training.py. Edit settings.py to change the experiment."""
from settings import MANIFEST, OUTPUT, training_config
from image_classifier.training import train

if __name__ == '__main__':
    train(MANIFEST, training_config(), OUTPUT)
