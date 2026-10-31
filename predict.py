import argparse
from settings import CHECKPOINT, OUTPUT
from image_classifier.inference import predict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify an image or folder; softmax is not proof of origin')
    parser.add_argument('input', help='Image file or folder')
    parser.add_argument('--checkpoint', default=CHECKPOINT)
    parser.add_argument('--output', default=OUTPUT / 'predictions.csv')
    parser.add_argument('--review-threshold', type=float, default=.7)
    args = parser.parse_args()
    predict(args.input, args.checkpoint, args.output, args.review_threshold)
