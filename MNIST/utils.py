from torch import tensor, Tensor
from torchvision import transforms
from PIL import Image


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


transform = transforms.Compose([
    transforms.ToTensor(),
    FlattenImage()
])