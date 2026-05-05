import torch
import torch.nn as nn
from torchvision import models
import os
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import shutil

from loader import DeepfakeVideoDataset, loadVideos







DEBUG = True
SEED = 42
SAVE_PERIOD = 1 # How many epochs between saving? (1 = every epoch, 3 = every 3 epochs)





class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2, lstm_hidden_dim=128, lstm_layers=1):
        super(DeepfakeDetector, self).__init__()
        
        # 1. Load pretrained EfficientNet
        self.backbone = models.efficientnet_v2_s(weights='DEFAULT')
        self.feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity() # Remove the original head

        # 2. LSTM Layer
        #   batch_first=True means input is [Batch, Seq, Features]
        self.lstm = nn.LSTM(self.feature_dim, lstm_hidden_dim, num_layers=lstm_layers, batch_first=True)

        # 2.5. Batch Normalization
        self.batch_norm = nn.BatchNorm1d(self.feature_dim)

        # 2.6. Dropout
        self.dropout = nn.Dropout(p=0.6)

        # 3. Final Classification Head
        self.classifier = nn.Linear(lstm_hidden_dim, num_classes)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, C, H, W]
        batch_size, seq_len, c, h, w = x.shape
        
        # Flatten batch and sequence to pass through CNN
        #   New shape: [Batch * Seq_Len, C, H, W]
        x = x.view(batch_size * seq_len, c, h, w)
        
        # Extract features (e.g., shape [Batch * Seq_Len, 1280])
        features = self.backbone(x)
        
        # Reshape back for LSTM: [Batch, Seq_Len, 1280]
        features = features.view(batch_size, seq_len, -1)

        # Apply batch normalization
        features = self.batch_norm(features.transpose(1, 2)).transpose(1, 2)

        # Pass through LSTM
        # out: [Batch, Seq_Len, Hidden_Dim]
        # hn: [Layers, Batch, Hidden_Dim] (final hidden state)
        out, (hn, cn) = self.lstm(features)

        # Apply dropout
        #   Use the final hidden state to classify
        prediction = self.classifier(self.dropout(hn[-1]))
        
        return prediction



def train_one_epoch(dataloader, debug=True):
    model.train()

    length = len(dataloader)
    total_loss = 0
    correct = 0
    total = 0

    # 300 iterations for 1 epoch
    for batch_idx, (videos, labels) in enumerate(dataloader):
        # videos: [Batch, Seq, C, H, W]
        videos, labels = videos.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Track accuracy during training
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Broadcast mid-training loss
        if (batch_idx % 10 == 0) and debug:
            print(f"   ({(batch_idx)}/{length}) » Loss: {loss.item():.4f} (Videos {batch_idx * len(labels):.0f}/{length * len(labels)})")

    avg_loss = total_loss / length
    train_accuracy = 100 * correct / total
    return avg_loss, train_accuracy



def evaluate_model(model, dataloader, device, type=""):
    if type != "":
        type = type+" "

    model.eval() # Set to evaluation mode
    correct = 0
    total = 0
    total_loss = 0
    all_preds = []
    all_labels = []

    guess_real = 0
    guess_fake = 0

    with torch.no_grad(): # Disable gradient tracking
        for videos, labels in dataloader:
            videos, labels = videos.to(device), labels.to(device)

            outputs = model(videos)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # predictions
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # added to ensure it doesn't guess entirely FAKE/REAL
            guess_fake += predicted.sum().item()
            guess_real += (len(predicted) - predicted.sum().item())

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    print(f'{type}Accuracy: {accuracy:.2f}% ({correct} Correct, {total} Total) (Predicting: {guess_real} Fake, {guess_fake} Real)')
    return accuracy, avg_loss, all_preds, all_labels








if __name__ == "__main__":

    # Video cache for cropped tensors
    choice = int(input("\nCropped video cache >>\nIt can take a long time to generate a new cache.\n\n1: Reset video cache\n2: Use pre-existing video cache\n\nOption > "))
    if choice == 1:
        if os.path.exists('cache'):
            shutil.rmtree('cache')
        os.mkdir('cache')
    else:
        if not os.path.exists('cache'):
            os.mkdir('cache')
    
    train_videos, val_videos, test_videos, train_labels, val_labels, test_labels = loadVideos(SEED)

    # standard EfficientNetV2-M normalization & size
    transform = transforms.Compose([
        
        transforms.ToPILImage(),
        transforms.Resize((224, 224)), # resolution
        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    # 2. Initialize the Dataset -- data is cropped & augmented
    train_dataset = DeepfakeVideoDataset(train_videos, train_labels, seq_length=16, transform=transform, cache_dir='cache/train')
    val_dataset = DeepfakeVideoDataset(val_videos, val_labels, seq_length=16, transform=transform, cache_dir='cache/validation')
    test_dataset = DeepfakeVideoDataset(test_videos, test_labels, seq_length=16, transform=transform, cache_dir='cache/test')

    # 3. Create the DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=6, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False)

    # 4. Main model features
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepfakeDetector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-1)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
 
    # DATA: End training early if validation loss doesn't decrease/have significant impact over <patience> epochs
    best_val_loss = float('inf')
    patience = 3
    epochs_without_improvement = 0

    # TRAIN / RUN MODEL
    choice = int(input("1: Train new model\n2: Load previous model\n\nOption > "))
    
    # Train new model
    if choice == 1:
        print("\n")
    
        timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        uuid = f"{input('Enter a prefix message for this run > ')} {timestamp}"
        save_dir = f'results/model_paths/{uuid}'

        print(f"Running on UUID >> {uuid}")
        # Create saves directory; error if fail
        try:
            print(f"Created new directory {save_dir}")
            os.makedirs(save_dir)
        except:
            print("Saves directory already exists!")

        log_dir = f'results/tensorboard/{uuid}'
        writer = SummaryWriter(log_dir)

        print("\nThe model takes around 30m to train for 10 epochs.")
        epoch_count = int(input("How many epochs? > "))
        for i in range(1, epoch_count+1):
            print(f"Training... [Epoch {i}]")
            avg_loss, train_accuracy = train_one_epoch(train_loader, debug=DEBUG)
            print(f"Train Accuracy: {train_accuracy:.2f}%")

            # Evaluate on validation set
            val_accuracy, val_loss, _, _ = evaluate_model(model, val_loader, device=device, type="Validation")
           
            # Save to Tensorboard
            writer.add_scalars(f"accuracy", {'train': train_accuracy, 'val': val_accuracy}, i)
            writer.add_scalars(f"loss", {'train': avg_loss, 'val': val_loss}, i)

            # Decay learning rate
            scheduler.step(val_accuracy)

            # Save model
            if i % SAVE_PERIOD == 0:
                torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch_{i}.pth'))
                print(f"Saved model at epoch {i}")

            # Break early if model doesn't improve
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                #torch.save(model.state_dict(), os.path.join(save_dir, f'model_best_epoch_{i}.pth'))
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {i}, no improvement for {patience} epochs.")
                    break
            

            print("")



    # Load previous model
    if choice == 2:
        

        save_dir = input("Enter the target .pth relative path (ex. 'results/model_paths/2026-04-20 15:14:17.557884/model_epoch_3.pth') > ")
        while os.path.isdir(save_dir) == True:
            print(f"Error: Requires .pth file, given a directory.")
            save_dir = input("Enter the target .pth relative path (ex. 'results/model_paths/2026-04-20 15:14:17.557884/model_epoch_3.pth') > ")

        uuid = save_dir.split("/")[-1] # Can cause issues with models not being held within `model_paths/UUID/.pth`

        
        model.load_state_dict(torch.load(save_dir, weights_only=True))
        epoch_count = 1 # necessary for final evaluation -> write to tensor

        log_dir = f'results/tensorboard/{uuid}'
        writer = SummaryWriter(log_dir)
        # CMD: tensorboard --logdir=runs/


    # Final model evaluation -- return train, validation & test accuracy+loss

    print("\n\nEvaluating model...\n")

    train_accuracy, train_loss, _, _ = evaluate_model(model, train_loader, device=device, type="Train")
    val_accuracy, val_loss, _, _ = evaluate_model(model, val_loader, device=device, type="Validation")
    test_accuracy, test_loss, _, _ = evaluate_model(model, test_loader, device=device, type="Test")
    
    writer.add_scalars(f"accuracy", {'train': train_accuracy, 'val': val_accuracy, 'test': test_accuracy}, epoch_count)
    writer.add_scalars(f"loss", {'train': train_loss, 'val': val_loss, 'test': test_loss}, epoch_count)
    
    writer.close()

