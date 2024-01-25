import argparse
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import matplotlib

# Use Agg backend for matplotlib (no display required)
matplotlib.use('Agg')


# TODO: Write a function that loads a checkpoint and rebuilds the model
def build_model_again(model_path):
    param_dict = torch.load(model_path)
    model = models.vgg16(pretrained=True)
    classifier = nn.Sequential(nn.Linear(param_dict["input_size"], 1588),
                                     nn.ReLU(),
                                     nn.Linear(1588, 488),
                                     nn.ReLU(),                                 
                                     nn.Linear(488, param_dict["output_size"]), 
                                     nn.LogSoftmax(dim=1))
    model.classifier = classifier
    model.cuda()
    model.load_state_dict(param_dict['state_dict'])
    return model

model = build_model_again('model.pth')

def process_image(image):
    input_image = Image.open(image).resize((256, 256))
    input_image_transformation = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    processed_image = input_image_transformation(input_image)
    return processed_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    model.eval().cpu()  # Set the model to evaluation mode
    image = process_image(image_path).unsqueeze(dim=0)
    with torch.no_grad():
        prediction = model(image)
        prediction_exp_prob = torch.exp(prediction) 
        actual_highest_value, index_of_highest_values = prediction_exp_prob.topk(topk)
    return [index_of_highest_values, actual_highest_value]

def plot_results(image):
    plt.subplot(2, 1, 1)
    imgplot = plt.imshow(mpimg.imread(image))
    prediction_result = predict(image)
    first_class = prediction_result[0][0]
    second_class = prediction_result[0][1]
    third_class = prediction_result[0][2]
    fourth_class = prediction_result[0][3]
    fifth_class = prediction_result[0][4]

    first_prob = prediction_result[1][0]
    second_prob = prediction_result[1][1]
    third_prob = prediction_result[1][2]
    fourth_prob = prediction_result[1][3]
    fifth_prob = prediction_result[1][4]

    y = [str(first_class), str(second_class), str(third_class), str(fourth_class), str(fifth_class)]
    x = [first_prob, second_prob, third_prob, fourth_prob, fifth_prob]

    plt.subplot(2, 1, 2)
    plt.barh(y, x)

    plt.show()

"""
def plot_results(image, output_path='prediction_plot.png'):
    fig, axs = plt.subplots(2, 1, figsize=(6, 9))
    
    axs[0].imshow(mpimg.imread(image))
    
    prediction_result = predict(image)
    classes = [str(idx) for idx in prediction_result[0][0]]
    probabilities = prediction_result[1][0]

    axs[1].barh(classes, probabilities)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()    
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict flower class from an image")
    parser.add_argument("image_path", help="Path to the image for prediction")
    parser.add_argument("model_path", help="Path to the trained model checkpoint")
    parser.add_argument("--topk", type=int, default=5, help="Number of top predictions to show")

    args = parser.parse_args()

    model = build_model_again(args.model_path)
    prediction_result = predict(args.image_path, model, args.topk)

    for i in range(args.topk):
        class_idx = prediction_result[0][0][i].item()  # Corrected index
        probability = prediction_result[1][0][i].item()  # Extract the probability value
        print(f"{class_idx}: {round(probability * 100, 2)}%")

    """plt.show()"""
    plot_results(args.image_path)


