from torch import tensor, Tensor
from torchvision import transforms
from PIL import Image
from torch_lr_finder import LRFinder
import numpy as np


def display_tensor_as_image(tensor: Tensor):
    '''
    Display a tensor of shape (1, 784) as a (1, 28, 28) image.
    '''
    to_img = transforms.ToPILImage()
    tensor = tensor.view(28, 28)
    img: Image.Image = to_img(tensor)
    img = img.resize((28*4, 28*4))
    display(img)


class FlattenImage():
    '''
    A callable class to be used in transforms.Compose
    Flattens an image tensor (1, 28, 28) -> (1, 784)
    '''
    def __call__(self, x):
        return x.view(28*28)

def set_optimal_learning_rate(model, optimizer, loss_fn, data_loader, device, end_lr, num_iter) -> None:
    '''
    Using torch_lr_finder.LRFinder, find, plot, and set the optimal learning rate.
    Updates the optimizer in place.
    '''

    lr_finder = LRFinder(model, optimizer, loss_fn, device=device)
    lr_finder.range_test(data_loader, end_lr=end_lr, num_iter=num_iter)
    min_grad_idx = (np.gradient(np.array(lr_finder.history['loss']))).argmin()
    optimal_lr = lr_finder.history['lr'][min_grad_idx]

    for param_group in optimizer.param_groups:
        param_group['lr'] = optimal_lr

    lr_finder.plot()
