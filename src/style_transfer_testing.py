
# coding: utf-8

# In[15]:


import logging

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
from skimage.transform import resize as resize_image
from scipy.misc import imsave

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import pyjet.backend as J

import copy

import data_utils as dsb

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# This is a bit of magic to make matplotlib figures appear inline in the notebook
# rather than in a new window.
# get_ipython().run_line_magic('matplotlib', 'inline')
# plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

# Set up some globals
SEED = 42
np.random.seed(SEED)



NUM_CHANNELS = 3

PATH_TO_TRAIN = '../input/train/'
PATH_TO_TEST = '../input/test/'

NUM_TRAIN = 670
NUM_TEST = 65


# In[4]:


# Load the clusters
train_cluster_ids = np.load("../clusters/train_clusters.npz")
test_cluster_ids = np.load("../clusters/test_clusters.npz")


in_set = np.vectorize(lambda a, s: a in s)


# In[5]:


# Load the training data
test_ids, X_test, sizes_test = dsb.load_test_data(path_to_test=PATH_TO_TEST, img_size=None,
                                                  num_channels=NUM_CHANNELS, mode='rgb')
train_ids, X_train, Y_train = dsb.load_train_data(path_to_train=PATH_TO_TRAIN, img_size=None,
                                                  num_channels=NUM_CHANNELS, mode='rgb')
print("Number of training samples: %s" % len(train_ids))
print("X-train shape: {}".format(X_train.shape))
print("Y-train shape: {}".format(Y_train.shape))
print("X-test shape: {}".format(X_test.shape))

# Get indexes from clusters
train_clusters = np.zeros(NUM_TRAIN, dtype=int)
test_clusters = np.zeros(NUM_TEST, dtype=int)
train_clusters[in_set(train_ids, {a for a in train_cluster_ids["cluster_0"]})]= 0
train_clusters[in_set(train_ids, {a for a in train_cluster_ids["cluster_1"]})]= 1
train_clusters[in_set(train_ids, {a for a in train_cluster_ids["cluster_2"]})]= 2

test_clusters[in_set(test_ids, {a for a in test_cluster_ids["cluster_0"]})]= 0
test_clusters[in_set(test_ids, {a for a in test_cluster_ids["cluster_1"]})]= 1
test_clusters[in_set(test_ids, {a for a in test_cluster_ids["cluster_2"]})]= 2

print(train_clusters)


# Get some image loading setup

# In[35]:


# Get the style images
style_images = np.concatenate([X_train[train_clusters == 0], X_test[test_clusters == 0]], axis=0)
content_images = np.concatenate([X_train[train_clusters != 0], X_test[test_clusters != 0]], axis=0)

# Save for the pytorch tutorial
# imsave("style.jpg", resize_image(style_images[np.random.randint(len(style_images))], content_images[0].shape[:3]))
# imsave("content.jpg", content_images[0])
# raise ValueError()


NUM_TRAIN_STYLE = np.count_nonzero(train_clusters == 0)
NUM_TRAIN_CONTENT = np.count_nonzero(train_clusters != 0)
NUM_TEST_STYLE = np.count_nonzero(test_clusters != 0)
NUM_TEST_CONTENT = np.count_nonzero(test_clusters != 0)



# In[41]:


# Content loss only has 1 target since there's only one content image
class ContentLoss(nn.Module):

    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        self.target = target.detach() * weight
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion((input * self.weight).expand_as(self.target), self.target)
        self.output = input
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

    
class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=B)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a, b, c * d)  # resise F_XL into \hat F_XL

        G = torch.bmm(features, features.transpose(1, 2))  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, weight):
        super(StyleLoss, self).__init__()
        self.G_input = None
        self.G_targets = None
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input, targets):
        self.output = input.clone()
        # Take gram of input and target
        self.G_input = self.gram(input)
        self.G_target = self.gram(targets).detach()
        # Apply the weight
        self.G_input.mul_(self.weight)
        self.G_target.mul_(self.weight)
        # First average target, or average criterion? Let's try average criterion
        # Since self.G_input has batch size of 1, broadcasting should happen
        self.loss = self.criterion(self.G_input.expand_as(self.G_target), self.G_target)
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


# In[8]:


cnn = models.vgg19(pretrained=True).features

# move it to the GPU if possible:
if J.use_cuda:
    cnn = cnn.cuda()


# In[9]:


print(repr(cnn))


# In[58]:


def get_input_param_optimizer(input_tensor):
    # this line to show that input is a parameter that requires a gradient
    input_param = nn.Parameter(input_tensor.data)
    optimizer = optim.Adam([input_param])
    return input_param, optimizer

class StyleTransferModel(nn.Module):
    
    def __init__(self, content_layers, style_layers, content_weight=1, style_weight=1000):
        super(StyleTransferModel, self).__init__()
        self.content_layers = content_layers
        self.style_layers = style_layers
        assert all("conv" in l for l in content_layers)
        assert all("conv" in l for l in style_layers)
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.content_losses = [None] * len(content_layers)
        self.style_losses = [None] * len(style_layers)
        self.model = None
    
    def compile(self, content_img, base_model):
        
        assert content_img.shape[3] == 3
        assert content_img.ndim == 4
        content_img = content_img.transpose(0, 3, 1, 2)
        
        self.model = nn.ModuleList()  # the new network
        content_count = 0
        style_count = 0
        i = 1
        content_hidden = Variable(J.from_numpy(content_img.astype(np.float32) / 255.))
        for layer in base_model:
            # Get the intermediate content values
            content_hidden = layer(content_hidden)
            # Add the layer to the model and create name
            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(i)
                self.model.add_module(name=name, module=layer)
                
                # Check if its a content layer
                if name in self.content_layers:
                    print("Adding layer", name, "to content losses")
                    content_loss = ContentLoss(content_hidden, self.content_weight)
                    self.model.add_module(name="content_loss_" + str(i), module=content_loss)
                    self.content_losses[content_count] = content_loss
                    content_count += 1
                # Check if its a style layer
                if name in self.style_layers:
                    print("Adding layer", name, "to content losses")
                    style_loss = StyleLoss(self.style_weight)
                    self.model.add_module(name="style_loss_" + str(i), module=style_loss)
                    self.style_losses[style_count] = style_loss
                    style_count += 1
                # Check if we're done
                if content_count == len(self.content_losses) and style_count == len(self.style_losses):
                    break
            
            if isinstance(layer, nn.ReLU):
                name = "relu_" + str(i)
                self.model.add_module(name=name, module=layer)
                i += 1
                
            if isinstance(layer, nn.MaxPool2d):
                name = "max_pool2d_" + str(i)
                self.model.add_module(name=name, module=layer)
                
        # Turn the losses into module lists
        self.content_losses = nn.ModuleList(self.content_losses)
        self.style_losses = nn.ModuleList(self.style_losses)
        
        print("Content_Losses")
        print(repr(self.content_losses))
        
        print("Style Losses")
        print(repr(self.style_losses))
        
        print("Model:")
        print(repr(self.model))
    
    def forward(self, x, targets):
        for name, layer in self.model._modules.items():
            # The only case we need to worry about it the style loss
            if "style_loss" in name:
                x = layer(x, targets)
            elif "content_loss" in name:
                x = layer(x)
            # All other layers, run everything through
            else:
                x = layer(x)
                targets = layer(targets)
        
    def backward(self):
        content_loss_value = 0
        style_loss_value = 0
        for loss in self.content_losses:
            content_loss_value += loss.backward()
        for loss in self.style_losses:
            style_loss_value += loss.backward()
        return content_loss_value, style_loss_value
    
    def run_model(self, content_img, style_imgs, num_steps=3000, batch_size=32):
        assert self.model is not None, "Compile the model first"
        print("Compiling the input image...")
        
        style_imgs = np.stack([resize_image(img, content_img.shape[1:3], mode='constant', preserve_range=True).astype(np.int8) 
                               for img in style_imgs])
        
        assert content_img.shape[3] == 3
        assert content_img.ndim == 4
        content_img = content_img.transpose(0, 3, 1, 2)
        assert style_imgs.shape[3] == 3
        assert style_imgs.ndim == 4
        style_imgs = style_imgs.transpose(0, 3, 1, 2)
        
        assert style_imgs.shape[1:] == content_img.shape[1:] == (3, 256, 320),             "content images: {}, style_images: {}".format(content_img.shape, style_imgs.shape)
        
        input_tensor = Variable(J.from_numpy(content_img.astype(np.float32) / 255.))
        input_param, optimizer = get_input_param_optimizer(input_tensor)
        
        # inds = np.random.randint(len(style_imgs), size=1)
        # style_tensors = Variable(J.from_numpy(style_imgs[inds].astype(np.float32) / 255.))
        # style_tensors = Variable(J.from_numpy(style_imgs.astype(np.float32) / 255.))
        
        print("Optimizing...")
        run = [0]
        while run[0] <= num_steps:
            
            # Get the next set of style images
            inds = np.random.randint(len(style_imgs), size=batch_size)
            style_tensors = Variable(J.from_numpy(style_imgs[inds].astype(np.float32) / 255.))
            
            def closure():
                # Correct the values of the updated input image
                input_param.data.clamp_(0, 1)

                optimizer.zero_grad()
                self(input_param, style_tensors)
                style_score = 0
                content_score = 0

                # Run the backward pass
                content_score, style_score = self.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.data[0], content_score.data[0]))
                    print()

                return style_score + content_score
            
            optimizer.step(closure)

        # a last correction...
        input_param.data.clamp_(0, 1)

        return (J.to_numpy(input_param.data) * 255).astype(np.int8)


# In[59]:


# Run on the first image
CONTENT_LAYERS = {'conv_4'}
STYLE_LAYERS =  {'conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'}
CONTENT_WEIGHT = 1
STYLE_WEIGHT = 1000

# Reset to put channels first
transfer_model = StyleTransferModel(CONTENT_LAYERS, STYLE_LAYERS, CONTENT_WEIGHT, STYLE_WEIGHT)
transfer_model.compile(content_images[0][np.newaxis], cnn)
new_image = transfer_model.run_model(content_images[0][np.newaxis], style_images)

plt.imshow(new_image.transpose(0, 2, 3, 1)[0])
plt.show()

