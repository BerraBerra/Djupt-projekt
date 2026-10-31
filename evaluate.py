import argparse
from settings import MANIFEST, CHECKPOINT, OUTPUT
from image_classifier.evaluation import evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default=MANIFEST)
    parser.add_argument('--checkpoint', default=CHECKPOINT)
    parser.add_argument('--output', default=OUTPUT)
    args = parser.parse_args()
    evaluate(args.manifest, args.checkpoint, args.output)
