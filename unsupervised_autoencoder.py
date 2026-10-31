#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  

import torch  
import torch.nn as nn 
import torch.optim as optim  
import matplotlib.pyplot as plt
from torchvision import datasets, transforms  
import random  

class DataLoaderUtility:
    def __init__(self):
        # Convert images to PyTorch tensors
        transform = transforms.ToTensor()  
        # Load the CIFAR10 training dataset
        self.train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    def get_selected_data(self):
        # Map class labels to their respective indices in the CIFAR10 dataset
        class_labels = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
        selected_classes = [class_labels['dog'], class_labels['airplane'], class_labels['frog']]
        # Initialise a counter for each selected class
        class_counts = {cls: 0 for cls in selected_classes}
        # Create a DataLoader for iterating through the dataset
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=128, shuffle=True)
        selected_images = []
        selected_labels = []
        # Iterate over the dataset to select specific classes of images
        for _, (imgs, labels) in enumerate(train_loader):
            for img, label in zip(imgs, labels):
                if label.item() in selected_classes and class_counts[label.item()] < 100:
                    selected_images.append(img)
                    selected_labels.append(label.item())
                    class_counts[label.item()] += 1
            # Break the loop once 100 images of each selected class are collected
            if all([count == 100 for count in class_counts.values()]):
                break
        return selected_images, selected_labels, selected_classes

    @staticmethod
    def load_test_data():
        # Convert images to tensors
        transform = transforms.ToTensor()
        # Load the CIFAR10 test dataset
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        return test_dataset

class CNN_Autoencoder(nn.Module):
    def __init__(self, num_classes):
        super(CNN_Autoencoder, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2, 8, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.ReLU()
        )
    
    def encode(self, x):
        # Encode an input image using the encoder
        encoded = self.encoder(x)
        encoded = nn.functional.adaptive_avg_pool2d(encoded, (1, 1))
        return encoded.view(encoded.size(0), -1)

    def forward(self, x):
        # Define the forward pass of the autoencoder
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class CNN_AutoencoderTrainer:
    def __init__(self, model, selected_images, selected_labels, selected_classes):
        self.model = model
        # Prepare the selected images and labels for training
        self.selected_images = torch.stack(selected_images).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.selected_labels = torch.tensor([selected_classes.index(l) for l in selected_labels]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # Set up the optimizer
        self.criterion_ae = nn.MSELoss()  # Set up the loss function for autoencoding

    def train(self, epochs=1000):
        # Training loop for 1000 epochs
        for epoch in range(epochs):
            self.optimizer.zero_grad()  # Clear old gradients
            decoded = self.model(self.selected_images) 
            loss = self.criterion_ae(decoded, self.selected_images)  
            # Backpropagate the error
            loss.backward()  
            self.optimizer.step()
            # Print loss
            if (epoch + 1) % 100 == 0:  
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

class PlotUtility:
    @staticmethod
    def plot_embeddings(embeddings, labels, class_names, selected_classes):
        # Define plotting parameters
        symbols = ['o', 'v', 's', 'x']
        colors = ['#F39C12', '#6495ED', '#9FE2BF', '#E74C3C']
        labels_map = {cls: class_names[cls] for cls in selected_classes}
        labels_map["new images"] = "new images"
        # Plot embeddings for each class
        for label_name, marker, color in zip(labels_map.keys(), symbols, colors):
            idx = [index for index, label in enumerate(labels) if label == label_name]
            if len(idx) > 0:
                if marker == 'x':
                    plt.scatter(embeddings[idx, 0], embeddings[idx, 1], marker=marker, color=color, label=labels_map[label_name])
                else:
                    plt.scatter(embeddings[idx, 0], embeddings[idx, 1], marker=marker, edgecolors=color, facecolors='none', label=labels_map[label_name])
        plt.legend(loc='upper left')  
        plt.show()  


if __name__ == "__main__":
    data_utility = DataLoaderUtility()
    # Get selected data (images and labels) from the DataLoaderUtility
    selected_images, selected_labels, selected_classes = data_utility.get_selected_data()

    # Initialise the CNN autoencoder model and moves it to the appropriate device (GPU or CPU)
    model = CNN_Autoencoder(len(selected_classes)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # Initialise the trainer with the model and selected data
    trainer = CNN_AutoencoderTrainer(model, selected_images, selected_labels, selected_classes)
    trainer.train()

    # Load test data
    test_dataset = DataLoaderUtility.load_test_data()
    # Randomly selects some images from the test dataset
    random_selected_indices = random.sample(range(len(test_dataset)), 5)
    random_selected_images = [test_dataset[i][0] for i in random_selected_indices]
    random_selected_labels = ["new images" for _ in range(5)]

    # Prepare data for embedding visualisation
    all_selected_images = selected_images + random_selected_images
    all_labels = selected_labels + random_selected_labels

    # Generate embeddings for the images
    with torch.no_grad():
        embeddings = model.encode(torch.stack(all_selected_images).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))).view(len(all_selected_images), -1).cpu().numpy()
    
    # Plot the embeddings
    PlotUtility.plot_embeddings(embeddings, all_labels, data_utility.train_dataset.classes, selected_classes)
