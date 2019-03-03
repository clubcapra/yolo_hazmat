# YOLO Hazmat

This repository contains the code to train Yolo using darknet, including configuration files, data processor, and augmenter.

## Quickstart

Dependencies are managed using [pipenv](https://github.com/pypa/pipenv), and a lock file is provided for reproducability.

To install dependencies, run `pipenv install`

The steps below assumes that your computer already has [CUDA](https://developer.nvidia.com/cuda-downloads) installed.

### Prepare the data

Run the `convert.py` script to convert the [Hazmat Label Dataset](https://osf.io/b5dap/) to the Yolo format if the
`--aug-factor` flag is given, the data will also be augmented, which is recommended considering the dataset's size.

Use `python3 convert.py -h` to see all options.

Models are trained with data prepared using this command: `python3 convert.py -dd data/ -od out/ -af 5 -tc 8`. Where
`-tc` is the number of threads.

Once the processing is done, there should be a new `out/` folder and two files `train.txt` and `test.txt`.

### Training

1. Edit `Makefile` and set `GPU=1` to enable CUDA.
2. Run `make`
3. Move the previously create files and folder to the `darknet` folder.
4. Run `./darknet detector train cfg/hazmat-yolov3.data cfg/hazmat-yolov3.cfg ./darknet53.conv.74`
5. Let it run for at least 1000 iterations (about 20 minutes on a GTX 1080Ti)
6. Test your model with `./darknet detector test cfg/hazmat-yolov3.data cfg/hazmat-yolov3.cfg ./backup/hazmat-yolov3_last.weights image-to-test.jpg`
    - You can pick any weights in the `backup/` folder
7. `predictions.jpg` file will appear in your working directory and you can see if the results make sense.

## Pretrained models 

All pretrained models are available here: https://drive.google.com/drive/u/0/folders/1lv1DpMYBFLBVi6YZIYVqv55xVzfU1zXi

You need to have access to CAPRA's Google Drive.

## TODOs

- Retrain with Yolo-tiny