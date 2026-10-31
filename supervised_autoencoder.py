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
        # Load the CIFAR10 training dataset applying the transformation
        self.train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        # Extract unique classes from the dataset
        self.unique_classes = list(set([y for _, y in self.train_dataset]))

    def get_selected_data(self):
        # Randomly select 3 unique classes from the dataset
        selected_classes = random.sample(self.unique_classes, 3)
        # Initialize a counter for each selected class
        class_counts = {cls: 0 for cls in selected_classes}
        # DataLoader to iterate through the dataset in batches
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=128, shuffle=True)
        selected_images = []
        selected_labels = []
        # Iterate to collect images and labels of the selected classes
        for _, (imgs, labels) in enumerate(train_loader):
            for img, label in zip(imgs, labels):
                if label.item() in selected_classes and class_counts[label.item()] < 100:
                    selected_images.append(img)
                    selected_labels.append(label.item())
                    class_counts[label.item()] += 1
            # Break the loop once 100 images for each class are collected
            if all([count == 100 for count in class_counts.values()]):
                break
        return selected_images, selected_labels, selected_classes

    @staticmethod
    def load_test_data():
        # Load and return the CIFAR10 test dataset
        transform = transforms.ToTensor()
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        return test_dataset

class SupervisedAutoencoder(nn.Module):
    def __init__(self, num_classes):
        super(SupervisedAutoencoder, self).__init__()
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
        # Linear layer for classification
        self.classifier = nn.Linear(2 * 4 * 4, num_classes)
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
        flattened_encoded = encoded.view(encoded.size(0), -1)
        # Pass encoded images through the classifier
        classification = self.classifier(flattened_encoded)
        # Decode the encoded images
        decoded = self.decoder(encoded)
        return decoded, classification

class SupervisedAutoencoderTrainer:
    def __init__(self, model, selected_images, selected_labels, selected_classes):
        self.model = model
        # Prepare the images and labels for training
        self.selected_images = torch.stack(selected_images).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.selected_labels = torch.tensor([selected_classes.index(l) for l in selected_labels]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # Initializes the optimizer
        # Loss functions for autoencoding and classification
        self.criterion_ae = nn.MSELoss()
        self.criterion_cls = nn.CrossEntropyLoss()

    def train(self, epochs=1000):
        # Training loop for 1000 epochs
        for epoch in range(epochs):
            self.optimizer.zero_grad()  # Clear gradients
            decoded, classifications = self.model(self.selected_images)
            # Calculate losses for autoencoding and classification
            loss_ae = self.criterion_ae(decoded, self.selected_images)
            loss_cls = self.criterion_cls(classifications, self.selected_labels)
            # Total loss
            loss = loss_ae + loss_cls 
            # Backpropagation
            loss.backward()  
            self.optimizer.step()
            # Prints loss
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

class PlotUtility:
    @staticmethod
    def plot_embeddings(embeddings, labels, class_names, selected_classes):
        # Plotting parameters for embeddings
        symbols = ['o', 'v', 's', 'x']
        colors = ['#F39C12', '#6495ED', '#9FE2BF', '#E74C3C']
        # Map class indices to names
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
    # Retrieve selected data (images and labels) 
    selected_images, selected_labels, selected_classes = data_utility.get_selected_data()
    model = SupervisedAutoencoder(len(selected_classes)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # Initialize and start training the model
    trainer = SupervisedAutoencoderTrainer(model, selected_images, selected_labels, selected_classes)
    trainer.train()
    # Load test data
    test_dataset = DataLoaderUtility.load_test_data()
    # Randomly select some images from the test dataset
    random_selected_indices = random.sample(range(len(test_dataset)), 5)
    random_selected_images = [test_dataset[i][0] for i in random_selected_indices]
    random_selected_labels = ["new images" for _ in range(5)]
    # Prepare data for embedding visualization
    all_selected_images = selected_images + random_selected_images
    all_labels = selected_labels + random_selected_labels
    # Generate embeddings for the images without updating the model
    with torch.no_grad():
        embeddings = model.encode(torch.stack(all_selected_images).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))).view(len(all_selected_images), -1).cpu().numpy()
    # Plot the embeddings
    PlotUtility.plot_embeddings(embeddings, all_labels, data_utility.train_dataset.classes, selected_classes)
