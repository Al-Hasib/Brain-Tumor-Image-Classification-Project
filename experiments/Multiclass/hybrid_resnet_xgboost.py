import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn import svm
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score

def main():
    # Define the device to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load a pre-trained model and modify it to output features
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Identity()  # Remove the last fully connected layer
    model = model.to(device)
    model.eval()

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load your dataset
    train_dataset = ImageFolder(root=r'C:\Users\abdullah\projects\Brain Tumor\Brain-Tumor-Image-Classification-Project\data\Brain Tumor Classification (MRI)\train', transform=transform)
    test_dataset = ImageFolder(root=r'C:\Users\abdullah\projects\Brain Tumor\Brain-Tumor-Image-Classification-Project\data\Brain Tumor Classification (MRI)\val', transform=transform)

    

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Extract features from the training set
    train_features = []
    train_labels = []

    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            train_features.append(outputs.cpu().numpy())
            train_labels.append(labels.numpy())

    train_features = np.concatenate(train_features)
    train_labels = np.concatenate(train_labels)

    # Train an XGBoost classifier on the extracted features
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(train_features, train_labels)

    # Train an SVM classifier on the extracted features
    svm_model = svm.SVC()
    svm_model.fit(train_features, train_labels)

    # Extract features from the test set
    test_features = []
    test_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            test_features.append(outputs.cpu().numpy())
            test_labels.append(labels.numpy())

    test_features = np.concatenate(test_features)
    test_labels = np.concatenate(test_labels)

    # Make predictions on the test set
    test_predictions = xgb_model.predict(test_features)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(test_labels, test_predictions)
    print(f'Test accuracy: {accuracy * 100:.2f}%')

    # Make predictions on the test set using SVM
    test_predictions_svm = svm_model.predict(test_features)
    accuracy_svm = accuracy_score(test_labels, test_predictions_svm)
    print(f'SVM Test accuracy: {accuracy_svm * 100:.2f}%')

    # Transform and extract features from a new image
    from PIL import Image
    img = Image.open(r'C:\Users\abdullah\projects\Brain Tumor\Brain-Tumor-Image-Classification-Project\data\Brain Tumor Classification (MRI)\val\meningioma_tumor\image(7).jpg')
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        img_features = model(img).cpu().numpy()

    # Predict the class label
    predicted_class = xgb_model.predict(img_features)
    print(f'Predicted class: {predicted_class[0]}')

    # Predict the class label using SVM
    predicted_class_svm = svm_model.predict(img_features)
    print(f'SVM predicted class: {predicted_class_svm[0]}')

if __name__ == '__main__':
    main()
