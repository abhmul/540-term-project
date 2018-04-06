import torch
import torch.nn as nn

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