# Deepfake Detection Model
A neural network model to detect the presence of deepfakes in videos. Trained on the FaceForensics++ dataset, achieving over 95% test accuracy.
Written in Spring 2026 for CS400-01 Computer Vision.

## Conda Environment

We experienced some difficulties getting the exported environment to work across both Conda and pip, and thus the installation may take awhile.

Install the included `environment.yml` conda environment to ensure you have the appropriate Python libraries & packages installed.   
Run `conda env create -f environment.yml` to install, then `conda activate df` to activate the environment.
- Include `-n new_name` if you want to use a different environment name (default is `df`)   

Run `conda install torchvision` to install Torchvision.
Run `pip install facenet-pytorch==2.6.0` to install Facenet, a library used for preprocessing and cropping videos from the dataset.
- Facenet is not currently available in conda environments.   

Ensure both `opencv-python` and `opencv-python-headless` are of version 4.13.0 (`pip list | grep opencv`). If they are not already installed, install them.   



## Downloading the Dataset

To download the dataset, you need to install Kaggle

Dataset Kaggle URL: https://www.kaggle.com/datasets/xdxd003/ff-c23/data
Follow this tutorial: https://www.youtube.com/watch?v=gkEbaMgvLs8

1. `conda install -c conda-forge kagglehub=0.3.13`
2. Create new directory `~/.kaggle`
3. Insert Kaggle API token into `~/.kaggle/access_token`
	- `cat ~/.kaggle/access_token` should return `KGAT_377f8a15...`
4. Run `download_data.py` to automatically download dataset into kagglehub's `.cache` folder
5. `mv ~/.cache/kagglehub/datasets/xdxd003/ff-c23/ ~/project/src/video/`

The filepath for the dataset after downloading & moving should match as follows:

```
project/   
├── src/   
│   └── video/   
|		└── ff-c23/   
|			|	1.complete   
|			└── versions/   
|				└── 1/   
|					└── FaceForensics++_C23/   
|						└── csv/   
|						└── Deepfakes/   
|						└── original/   
|						└── ...   
```

You can also customize the directory organization by editing the config at the top of `loader.py` with DATASET_DEEPFAKE and DATASET_ORIGINAL.   


The FaceForensics++ dataset (original + Deepfakes) contains a total of 2,000 videos. When running the model, the data is cropped down to the individual's face, duplicated & flipped to increase the dataset size.   
The train, validation and test splits are 60%, 20%, 20%. The final dataset size after augmentation is 1,200(2)+400+400, totalling **3,200 videos**.   
Videos after augmentation are cached to reduce computation costs across epochs.   
  
Cached videos can be regenerated on run, but will not change unless loadVideos(seed) is changed to use a different seed. The default seed is 42, which can be changed in `model/model.py`.





## Running the model

> [!WARNING]
> Ensure the conda environment is active & the dataset is located in the correct directory before running

Run `python3 model/model.py` to begin training the model.
- Create a new video cache if one does not exist already (Option 1)
	- This can take some time, depending on your CPU speeds. This process crops and outputs 3,200 videos around the subject's face, storing it in a .pth file within `cache/[train|test|validation]`
	- The cache will contain the exact same videos on subsequent generations. Regenerating the cache will have no effect unless the seed is changed.
		- The generation seed is located at the top of `model/model.py`

- Train a new model and select the specified number of epochs. From our testing, around 10 epochs will get the best results. Models trained further than 10 epochs will gain slight but minimal accuracy increases.

In order to load a previous model, copy the relative path of a save directory and input it after selecting "Load Previous Model" (ex. `results/model_paths/r8 2026-04-28 22:02:41.391219`)


After training or loading a model, it will then evaluate it on the Train, Validation and Test datasets and return a final accuracy for each set.


## Tensorboard

To view graphs for a model, copy the relative path for a result and run:
- `tensorboard --logdir='results/tensorboard/r8 2026-04-28 22:02:41.391219'`
- If the default port is already in use, add the `--port=6009` flag using another port.



The best recorded run is listed here:
-   Train Accuracy: 99.88% (2397 Correct, 2400 Total) (Predicting: 1219 Fake, 1181 Real)
-   Validation Accuracy: 97.25% (778 Correct, 800 Total) (Predicting: 386 Fake, 414 Real)
-   Test Accuracy: 98.50% (788 Correct, 800 Total) (Predicting: 380 Fake, 420 Real)