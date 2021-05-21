"""

Template code for COMP5623M CW1 Question 2

python explore.py --image_path XX --use_pre_trained True


"""

import argparse
import cv2
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Set up training arguments and parse

parser = argparse.ArgumentParser(description='Explore pre-trained AlexNet')


parser.add_argument(
    '--image_path', type=str,
    help='Full path to the input image to load.')
parser.add_argument(
    '--use_pre_trained', type=bool, default=True,
    help='Load pre-trained weights?')


args = parser.parse_args()

# Device configuration - defaults to CPU unless GPU is available on device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


print("=======================================")
print("                PARAMS               ")
print("=======================================")
for arg in vars(args):
    print(F"{arg:>20} {getattr(args, arg)}")


#########################################################################
#
#        QUESTION 2.1.2 code here
# 
#########################################################################


# Read in image located at args.image_path

img = cv2.imread(args.image_path, 1)

# Normalisations expected by pre-trained net, to apply in the image transform
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean,norm_std)
])
image_input = data_transform(img).unsqueeze(0)
# Loads the model and downloads pre-trained weights if not already downloaded
model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)

# To see the AlexNet architecture
print(model)

model.eval()


# Pass image through a single forward pass of the network

output = model(image_input)






# layer indices of each conv layer in AlexNet
conv_layer_indices = [0, 3, 6, 8, 10]

#########################################################################
#
#        QUESTION 2.1.3 
# 
#########################################################################


def filter_normalization(layer):
    layer_max = layer.max()
    layer_min = layer.min()
    return (layer - layer_min) / (layer_max - layer_min)


def extract_filter(conv_layer_idx, model):
    """ Extracts a single filter from the specified convolutional layer,
		zero-indexed where 0 indicates the first conv layer.

		Args:
			conv_layer_idx (int): index of convolutional layer
			model (nn.Module): PyTorch model to extract from

	"""

    # Extract filter
    for layer in model.named_modules():
        if isinstance(layer[1], nn.Conv2d):
            layer_index = layer[0].replace('features.', ' ')
            layer_index = int(layer_index)
            if layer_index == conv_layer_idx:
                the_filter = layer[1].weight

    fig = plt.figure(figsize=(35, 35))
    for x in range(len(conv_layer_indices)):
        if x == 1 or x == 3:
            continue
        filter = extract_filter(conv_layer_indices[x], model)
        filter = filter_normalization(filter)
        filter_num = filter.detach().numpy()
        for i, layers in enumerate(filter_num):
            for j, kernal_layers in enumerate(layers):
                if(i + 1) > 64:
                    continue
                ax = fig.add_subplot(8, 8, i + 1)
                ax.imshow(kernal_layers, cmap='gray')
                ax.axis('off')
        fig.savefig('desk_filter_%s.jpg' % x)
    return the_filter


# for i in conv_layer_indices:
#     print("features" + str(i), extract_filter(conv_layer_idx=i, model=model))

# print(extract_filter(conv_layer_idx=6,model=model))





# plt.show()

#########################################################################
#
#        QUESTION 2.1.4
# 
#########################################################################


def extract_feature_maps(input, model):
    """ Extracts the all feature maps for all convolutional layers.

		Args:
			input (Tensor): input to model
			model (nn.Module): PyTorch model to extract from

	"""

	# Extract all feature maps
	# Hint: use conv_layer_indices to access
    conv_model=nn.Sequential()
    for layer in model.named_modules():
        if isinstance(layer[1], nn.Conv2d):
            conv_model.add_module(layer[0].replace('.', ' '), layer[1])
    feature_maps = [conv_model[0](input)]
    for x in range(1, len(conv_model)):
        feature_maps.append(conv_model[x](feature_maps[-1]))

    for x in range(len(feature_maps)):
        plt.figure(figsize=(30, 30))
        if x == 1 or x == 3:
            continue
        layers = feature_maps[x][0, :, :, :].detach()
        for i, filter in enumerate(layers):
            if i == 64:
                break
            plt.subplot(8, 8, i + 1)
            plt.imshow(filter, cmap='gray')
            plt.axis('off')
    plt.show()

    return feature_maps

# print(extract_feature_maps(image_input, model))




