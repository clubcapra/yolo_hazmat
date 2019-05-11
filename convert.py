import os
import sys
import argparse
import random

from multiprocessing import Pool

import numpy as np
import imgaug as ia
from PIL import Image, ImageDraw
from imgaug import augmenters as iaa


def parse_arguments():
    """
    Parse the CLI arguments
    """

    parser = argparse.ArgumentParser(description='Convert your data to Darknet format')

    parser.add_argument(
        "--data-dir",
        "-dd",
        type=str,
        nargs="?",
        help="The folder containing the raw training and testing data"
    )

    parser.add_argument(
        "--output-dir",
        "-od",
        type=str,
        nargs="?",
        help="The folder were the samples should be saved"
    )

    parser.add_argument(
        "--train-test-ratio",
        "-ttr",
        type=float,
        nargs="?",
        default=0.7,
        help="Floating point value representing the proportion of samples that we will use for training"
    )

    parser.add_argument(
        "--aug-factor",
        "-af",
        type=int,
        nargs="?",
        default=5,
        help="How many synthetic samples will be generated for each legitimate sample"
    )

    parser.add_argument(
        "--thread-count",
        "-tc",
        type=int,
        nargs="?",
        default=8,
        help="How many threads to use for the conversion, the maximum value is the number of categories you have"
    )

    return parser.parse_args()

def build_augmenter(seed):
    """
    Use imgaug to build the augmenter, that will augment (duh) our dataset.
    """

    ia.seed(seed)
    return iaa.SomeOf((1, 4), [
        iaa.GaussianBlur((0, 3.0)),
        iaa.AverageBlur(k=(2, 7)),
        iaa.MedianBlur(k=(3, 11)),
        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
        iaa.AddToHueAndSaturation((-20, 20)),
        iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
        iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
        iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2)
    ])


def bbox2(img):
    """
    Convert the given mask to a box
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return cmin, cmax, rmin, rmax

def convert(size, box):
    """
    Convert a box to the darknet format (Which expects a centered? value)
    """
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def parse_folder(i, d, args):
    """
    Takes a folder, convert if to trainable sample.
    Apply augmentation if needed.
    """
    try:
        os.makedirs(os.path.join(args.output_dir, d))
    except:
        pass # Only god can judge me
    files = os.listdir(os.path.join(args.data_dir, d))
    files.sort()
    for j, f in enumerate(files):
        # Every one in two is the mask/label file
        if j % 2 != 0:
            continue
        if j % 1000 == 0:
            print(f"{j}/{len(files)}")
        try:
            prefix = "test" if i % int((1 - args.train_test_ratio) * 100) == 0 else "train"
            img = Image.open(os.path.join(args.data_dir, d, f)).convert('RGB')
            img.thumbnail((1200, 1200))
            img.copy().save(os.path.join(args.output_dir, d, f"{prefix}_{i}_{j}_0.jpg"))
            try:
                mask = Image.open(os.path.join(args.data_dir, d, f[:-4] + '_mask.png'))
                mask.thumbnail((1200, 1200))
                x, y, w, h = convert(img.size, bbox2(np.array(mask)))
            except Exception as inner_ex:
                # Probably a .txt mask
                with open(os.path.join(args.data_dir, d, f[:-4] + '.txt')) as f:
                    _, x, y, w, h = f.read().replace('\n', '').split(' ')

            with open(os.path.join(args.output_dir, d, f"{prefix}_{i}_{j}_0.txt"), 'w') as f:
                f.write(f"{i} {x} {y} {w} {h}\n")

            # Do we need to run augmentations
            if args.aug_factor <= 0:
                continue

            # Time to create some augmentations
            augmenter = build_augmenter(random.randint(0, 100000000))
            # Converting to OpenCV2 format
            img = np.array(img)[:, :, ::-1].copy()
            res = augmenter.augment_images([img for _ in range(args.aug_factor)])
            for k, ai in enumerate(res):
                Image.fromarray(ai[:, :, ::-1]).save(os.path.join(args.output_dir, d, f"train_{i}_{j}_{k + 1}.jpg"))
                with open(os.path.join(args.output_dir, d, f"train_{i}_{j}_{k + 1}.txt"), 'w') as f:
                    f.write(f"{i} {x} {y} {w} {h}\n")

        # Some images are odd and I'd rather not spend time fixing them
        except Exception as ex:
            print(f"Exception thrown: {ex}")

def main():
    """
    We will go through the dataset we have a convert the groundtruths
    to .txt files, as needed by darknet.
    """

    args = parse_arguments()

    p = Pool(args.thread_count)
    p.starmap(
        parse_folder,
        [
            (i, d, args)
            for i, d in enumerate(os.listdir(args.data_dir))
        ]
    )

    with open('test.txt', 'w') as test_file, open('train.txt', 'w') as train_file:
        for d in os.listdir(args.output_dir):
            print(d)
            for i, f in enumerate(os.listdir(os.path.join(args.output_dir, d))):
                if f[-3:] not in ('jpg', 'png'):
                    continue
                if f[0:4] == 'test':
                    test_file.write(os.path.join(args.output_dir, d, f) + "\n")
                else:
                    train_file.write(os.path.join(args.output_dir, d, f) + "\n")

    print('Done')

if __name__=='__main__':
    main()
