'''
File used to do the prediction
'''
import argparse
import random
import os
import torch
import numpy as np
from torch.autograd import Variable
from pytorch.vision import utils
from pytorch.vision.model import net as net
from pytorch.vision.model import data_loader as data_loader
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

def predict(image_filename):
    # Process image
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(image_filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((64, 64), Image.BILINEAR)

    # pytorch provides a function to convert PIL images to tensors.
    pil2tensor = transforms.ToTensor()
    rgb_image = pil2tensor(image)
    # Numpy -> Tensor
    #image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    # Add batch of size 1 to image
    model_input = rgb_image.unsqueeze(0)

    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()  # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)

    # Define the model
    model = net.Net(params).cuda() if params.cuda else net.Net(params)
    checkpoint = torch.load("experiments/test/best.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    loss_fn = net.loss_fn
    metrics = net.metrics
    # Probs
    probs = torch.exp(model.forward(model_input))
    print(probs) # this will result a tensor
    probs_t = probs.detach().numpy().tolist()[0]
    print(probs_t) # this will result a nd array

    #get the softmax value from model
    output_batch = model(model_input)
    print(output_batch)
    #loss = loss_fn(output_batch, labels_batch)

    # extract data from torch Variable, move to cpu, convert to numpy arrays
    output_batch = output_batch.data.cpu().numpy()
    print(output_batch) #np array
    outputs = np.argmax(output_batch, axis=1) #get the prediction from softmax
    print(outputs)
    #labels_batch = labels_batch.data.cpu().numpy()

    # Top probs
    #top_probs, top_labs = probs.topk(top_num)
    #top_probs = top_probs.detach().numpy().tolist()[0]
    #top_labs = top_labs.detach().numpy().tolist()[0]

    # Convert indices to classes
    #idx_to_class = {val: key for key, val in
    #                model.class_to_idx.items()}
    #top_labels = [idx_to_class[lab] for lab in top_labs]
    #top_flowers = [label_map[idx_to_class[lab]] for lab in top_labs]
    #return top_probs, top_labels, top_flowers


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/SIGNS', help="Directory with the SIGNS dataset")
parser.add_argument('--output_dir', default='data/64x64_SIGNS', help="Where to write the new data")
parser.add_argument('--model_dir', default='experiments/test', help="Where to write the new data")

if __name__ == '__main__':
    args = parser.parse_args()
    image_filename = "data/64x64_SIGNS/test_signs/2_IMG_4607.jpg"
    predict(image_filename)