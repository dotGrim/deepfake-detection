import torch
import os
from torch.utils.data import Dataset, random_split
from torchvision import transforms

from crop import crop_video


DATASET_DEEPFAKE = 'src/video/ff-c23/versions/1/FaceForensics++_C23/Deepfakes'
DATASET_ORIGINAL = 'src/video/ff-c23/versions/1/FaceForensics++_C23/original'


class DeepfakeVideoDataset(Dataset):
    def __init__(self, video_paths, labels, seq_length=16, transform=None, cache_dir="cache"):
        self.video_paths = video_paths
        self.seq_length = seq_length
        self.transform = transform

        if "test" in cache_dir or "validation" in cache_dir:
            augment = False
            length = len(video_paths)
        else:
            augment = True
            length = len(video_paths) * 2
        video_counter = 0

        # Create a flipped version of the transform for data augmentation
        flipped_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=1.0),
            # transforms.RandomVerticalFlip(p=1.0),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # cache cropped videos
        self.cache_dir = cache_dir

        # Preprocess all videos once
        self.cached_paths = []
        new_labels = []
        for i, path in enumerate(video_paths):
            cache_path = os.path.join(cache_dir, f"{i}.pt")
            cache_path_flipped = os.path.join(cache_dir, f"{i}_flipped.pt")
            
            if not os.path.exists(cache_dir):
                os.mkdir(cache_dir)
            
            # Cache original version
            if not os.path.exists(cache_path):
                video_counter += 1
                print(f"Caching {video_counter}/{length} ({cache_path}): {path}")
                tensor = crop_video(path, seq_length=seq_length, face_size=224, stride=1, use_center_crop=False, transform=transform)
                torch.save(tensor, cache_path)
            
            # Cache flipped version
            if not os.path.exists(cache_path_flipped) and augment:
                video_counter += 1
                print(f"Caching {video_counter}/{length} ({cache_path_flipped}): {path}")
                tensor = crop_video(path, seq_length=seq_length, face_size=224, stride=1, use_center_crop=False, transform=flipped_transform)
                torch.save(tensor, cache_path_flipped)
            
            if os.path.exists(cache_path):
                self.cached_paths.append(cache_path)
                new_labels.append(labels[i])
            if os.path.exists(cache_path_flipped):
                self.cached_paths.append(cache_path_flipped)
                new_labels.append(labels[i])
        
        self.labels = new_labels

    def __len__(self):
        return len(self.cached_paths)

    def __getitem__(self, idx):

        video_tensor = torch.load(self.cached_paths[idx])
        return video_tensor, torch.tensor(self.labels[idx])





def loadVideos(seed=42):

    path = {}
    dir = {}

    # Get dataset directories
    path['deepfake'] = DATASET_DEEPFAKE
    path['original'] = DATASET_ORIGINAL

    # Get full list of files in each directory ++ Assign labels
    dir['fake'] = list(map(lambda x: f'{path["deepfake"]}/{x}', os.listdir(path["deepfake"])))
    dir['real'] = list(map(lambda x: f'{path["original"]}/{x}', os.listdir(path["original"])))
    videos = dir['fake'] + dir['real']
    labels = [1] * (len(dir['fake'])) + [0] * (len(dir['real']))

    # Split dataset based on seed
    generator = torch.Generator().manual_seed(seed)
    train_videos, val_videos, test_videos = random_split(videos, [0.6, 0.2, 0.2], generator=generator)
    
    train_labels = [labels[i] for i in train_videos.indices]
    val_labels = [labels[i] for i in val_videos.indices]
    test_labels = [labels[i] for i in test_videos.indices]

    return list(train_videos), list(val_videos), list(test_videos), train_labels, val_labels, test_labels



# Test basic dataloading capabilities
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train_videos, val_videos, test_videos, train_labels, val_labels, test_labels = loadVideos(42)

    # standard EfficientNetV2-M normalization & size
    transform = transforms.Compose([
        
        transforms.ToPILImage(),
        transforms.Resize((224, 224)), # resolution
        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Initialize the Dataset
    train_dataset = DeepfakeVideoDataset(train_videos, train_labels, seq_length=16, transform=transform, cache_dir='cache/train')
    val_dataset = DeepfakeVideoDataset(val_videos, val_labels, seq_length=16, transform=transform, cache_dir='cache/validation')
    test_dataset = DeepfakeVideoDataset(test_videos, test_labels, seq_length=16, transform=transform, cache_dir='cache/test')

    # 3. Create the DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=6, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False)

    print(len(train_loader)*8, len(val_loader)*6, len(test_loader)*6)