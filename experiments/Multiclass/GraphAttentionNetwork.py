import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np
import os

class GAT(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def image_to_graph(image):
    """Convert an image to a graph format suitable for torch_geometric."""
    C, H, W = image.shape
    nodes = image.view(C, -1).T  # Reshape image to (H*W, C)
    edge_index = []
    
    for i in range(H):
        for j in range(W):
            idx = i * W + j
            if i > 0:
                edge_index.append([idx, (i-1) * W + j])
            if i < H - 1:
                edge_index.append([idx, (i+1) * W + j])
            if j > 0:
                edge_index.append([idx, i * W + (j-1)])
            if j < W - 1:
                edge_index.append([idx, i * W + (j+1)])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=nodes, edge_index=edge_index)

def main():
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    # Load dataset
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    num_classes = len(dataset.classes)

    # Convert dataset to graph data
    graph_data_list = [image_to_graph(image) for image, label in dataset]
    labels = torch.tensor([label for _, label in dataset], dtype=torch.long)

    # Split dataset into train and test sets
    train_indices, test_indices = train_test_split(np.arange(len(graph_data_list)), test_size=0.2, random_state=42)
    train_data = [graph_data_list[i] for i in train_indices]
    train_labels = labels[train_indices]
    test_data = [graph_data_list[i] for i in test_indices]
    test_labels = labels[test_indices]

    # Create DataLoader
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Training settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GAT(in_channels=3, out_channels=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    def train():
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out, train_labels[data.index])
            loss.backward()
            optimizer.step()

    def test(loader):
        model.eval()
        correct = 0
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            correct += pred.eq(test_labels[data.index]).sum().item()
        return correct / len(loader.dataset)

    # Training the model
    for epoch in range(1, 201):
        train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    # Evaluate the model
    test_acc = test(test_loader)
    print(f'Test Accuracy: {test_acc:.4f}')

if __name__ == '__main__':
    main()
