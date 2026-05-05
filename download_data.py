import kagglehub

# Using kagglehub 0.3.13
#
#	1) Insert Kaggle's API token in `~/.kaggle/access_token`
#	1) Run this .py file
#	2) Move new downloaded folder into designated project folder using the `mv` terminal command below.



path = kagglehub.dataset_download("xdxd003/ff-c23")
print("Path", path)
# mv ~/.cache/kagglehub/datasets/xdxd003/ff-c23/ ~/repo/deepfake_detection/src/video/
