import torch
import torch.nn as nn


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

        style_imgs = np.stack(
            [resize_image(img, content_img.shape[1:3], mode='constant', preserve_range=True).astype(np.int8)
             for img in style_imgs])

        assert content_img.shape[3] == 3
        assert content_img.ndim == 4
        content_img = content_img.transpose(0, 3, 1, 2)
        assert style_imgs.shape[3] == 3
        assert style_imgs.ndim == 4
        style_imgs = style_imgs.transpose(0, 3, 1, 2)

        assert style_imgs.shape[1:] == content_img.shape[1:] == (
        3, 256, 320), "content images: {}, style_images: {}".format(content_img.shape, style_imgs.shape)

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