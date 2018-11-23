from model import load_model, create_transform, load_classes
from PIL import Image, ImageOps
import optparse
import torch

parser = optparse.OptionParser()

parser.add_option('-i', '--image_file',
    action="store", dest="file_name",
    help="Input image file name")

options, args = parser.parse_args()

if options.file_name is None:
    print("Usage:")
    print("main.py -i <Image file complete path with name>")
    exit()

IMG_URL = options.file_name

classes=load_classes()
model = load_model()

img_pil = Image.open(IMG_URL).convert('L')
img_pil = ImageOps.invert(img_pil)
transform = create_transform()
img_tensor = transform(img_pil)
img_tensor.unsqueeze_(0)

outputs = model(img_tensor)

_, predicted = torch.max(outputs, 1)

print('Predicted Digit: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(1)))