import torch
from torch.utils.data import DataLoader
from train import train_one_epoch
from evaluate import evaluate_model
from model import VGG
from dataset import VGGdata
import urllib.request
import io
import numpy as np
from torchvision.transforms import Resize
from sklearn.model_selection import train_test_split

# In this part I wanted to download and load the dataset
url = "https://zenodo.org/records/10519652/files/organcmnist_128.npz?download=1"
response = urllib.request.urlopen(url)
npz_file = io.BytesIO(response.read())
data = np.load(npz_file)

# Here I preprocess data and do the resizing to fit the VGG input size
test_images = torch.from_numpy(data['test_images']).float().unsqueeze(1)
test_labels = torch.from_numpy(data['test_labels']).float().squeeze()

resize = Resize((224, 224))
resized_images = resize(test_images)

data_train, data_test, targets_train, targets_test = train_test_split(
    resized_images, test_labels, test_size=0.2, shuffle=True
)

# preparing test and train datasets VGGdata 
train_dataset = VGGdata(data_train, targets_train)
test_dataset = VGGdata(data_test, targets_test)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# model, optimizer, and loss function
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VGG().to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

# Training for 8 epochs and evaluation of each training loop 
EPOCHS = 8
for epoch in range(EPOCHS):
    print(f"Starting Epoch {epoch + 1}/{EPOCHS}")
    
    train_one_epoch(
        model=model,
        dataloader=train_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epoch_number=epoch,
        device=device
    )
    
    train_acc, train_f1 = evaluate_model(model, train_dataloader, device)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Train Accuracy: {train_acc:.4f}, Train F1 Score: {train_f1:.4f}")

# As final step, I evaluate my model using the test dataset 
test_acc, test_f1 = evaluate_model(model, test_dataloader, device)
print(f"Final Test Accuracy: {test_acc:.4f}, Test F1 Score: {test_f1:.4f}")
