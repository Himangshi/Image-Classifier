import argparse
import torch
from torch import nn, optim
from torchvision import models, transforms, datasets
import json

def train_model(data_dir, save_path, epochs=5):
    # Define data directories
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define data transformations
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)

    # Create data loaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

    # Load class-to-name mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # Load pretrained VGG16 model
    model = models.vgg16(pretrained=True)

    # Freeze pretrained model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define custom classifier
    classifier = nn.Sequential(
        nn.Linear(25088, 1588),
        nn.ReLU(),
        nn.Linear(1588, 488),
        nn.ReLU(),
        nn.Linear(488, 102),
        nn.LogSoftmax(dim=1)
    )

    # Replace classifier in the model
    model.classifier = classifier

    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.002)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    for e in range(1, epochs + 1):
        batch_counter = 0
        training_loss = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            batch_counter += 1

            optimizer.zero_grad()
            prediction = model(images)
            loss = criterion(prediction, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

            if batch_counter % 30 == 0:
                model.eval()
                testing_accuracy = 0
                testing_loss = 0

                with torch.no_grad():
                    for images_test, labels_test in validloader:
                        images_test, labels_test = images_test.to(device), labels_test.to(device)
                        prediction_test = model(images_test)
                        testing_loss += criterion(prediction_test, labels_test)

                        prediction_exp_prob_test = torch.exp(prediction_test)
                        actual_highest_value_test, index_of_highest_values_test = prediction_exp_prob_test.topk(1)
                        comparison_test = (index_of_highest_values_test == labels_test.view(*index_of_highest_values_test.shape)).sum().item()
                        testing_accuracy += comparison_test

                print(f"Epoch: {e} | Batch: {batch_counter} | Training loss: {round(training_loss, 2)} | Validation loss: {round((testing_loss / len(validloader)).item(), 2)} | Validation Accuracy: {round(testing_accuracy / len(validloader.dataset) * 100, 2)}")
                print("----------------------------------------")
                training_loss = 0

    # Save the model checkpoint
    param_dict = {'input_size': 25088, 'output_size': 102, 'state_dict': model.state_dict()}
    torch.save(param_dict, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a flower classification model")
    parser.add_argument("data_dir", help="Path to the directory containing the flower dataset")
    parser.add_argument("save_path", help="Path to save the trained model checkpoint")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for training")

    args = parser.parse_args()
    train_model(args.data_dir, args.save_path, args.epochs)
