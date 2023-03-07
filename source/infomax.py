import torch
import torch.nn as nn
import os
import numpy as np
from pathlib import Path

fwd = Path(__file__).resolve()
path=str(Path(fwd).parents[1])


# Set global variable for the pytorch device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class Params():
    def __init__(self):
        self.lr = 1.
        # this learning rate is tuned to an input size of 3000, at other input sizes navigation may be succesful but is not optimal
        # initial observation - larger image requires smaller learning rate before saturation

        self.noEpochs = 1
        self.loadModel = None
        self.outputSize = None
        self.seed = None
        self.saveModel=True
        self.model_path = path+'/Models/infomaxModels/'
        self.algorithmName='Infomax'


class InfomaxNetwork(nn.Module):
    def __init__(self, infomaxParams, imgs, deg_range=(-180, 180), degree_shift=1):
        super(InfomaxNetwork, self).__init__()
        
        self.deg_range = deg_range
        self.deg_step = degree_shift
        self.num_of_rows = imgs[0].shape[0]
        self.num_of_cols = imgs[0].shape[1]
        self.num_of_cols_perdegree = self.num_of_cols / 360
        self.degrees = np.arange(*deg_range)
        self.total_search_angle = round((deg_range[1] - deg_range[0]) / self.deg_step)

        # prep imgs
        self.imgs = [torch.unsqueeze(torch.from_numpy(item).float(), 0) for item in imgs]

        self.size = self.imgs[0].flatten().size(0)
        self.params = infomaxParams


        if infomaxParams.outputSize == None:
            self.params.outputSize = self.size
        else:
            self.outputSize = infomaxParams.outputSize

        if infomaxParams.seed == None:
            torch.seed()
        else:
            self.params.seed = infomaxParams.seed
            torch.manual_seed(self.params.seed)

        # NN.flatten so we can use it for batches
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.size,
                             self.params.outputSize,
                             bias=False,device=device)

        nn.init.uniform_(self.fc1.weight, -0.5, 0.5)
        #print('1')

        weights=self.fc1.weight
        #print(weights)
        std_weights=self.Standardize(weights)
        #print('2')
        self.fc1.weight = torch.nn.Parameter(std_weights)
        #print('3')
        self.fc1.weight.requires_grad = False

        self.TrainNet(self.imgs)

    def Standardize(self, t):
        #print('try')
        #print(t.device)
        #print(torch.max(t))
        t = (t - torch.mean(t)) / torch.std(t)
        return t

    # Takes in rolled single image (dimensions = (no_rolls, image y axis, image x axis))
    # Outputs familiarity score for each roll
    # Essentially gets the RRFF
    def Familiarity(self, x):
        outputVector = self.Forward(x)
        x = torch.sum(torch.abs(outputVector), dim=-1)
        return (x)

    def Forward(self, x):
        # if using multiple flattened images then we nned to make the column vectors
        if (len(x.shape)) > 2:
            x = self.flatten(x)
        x = x.unsqueeze(0)
        
        # Amani's normalization for the use of constant learning rate
        #x = x / 10

        x = (self.fc1(x))
        return x

    def TrainNet(self, train_set):
        # Standarise the inputs
        train_set = [self.Standardize(img) for img in train_set]
        for epoch in range(self.params.noEpochs):
            for img in train_set:
                #x = torch.flatten(img.squeeze())
                u = self.Forward(img)
                h = u.squeeze()
                y = torch.tanh(u)
                for param in self.parameters():
                    W = param
                WH = torch.matmul(torch.transpose(W, 0, 1), h)
                update = torch.outer(torch.add(y, h).squeeze(), WH)
                dW = (W - update)
                # New normalisation of the eta by the netwrok size (inputs units * out units)
                change = (self.params.lr / (self.size*self.params.outputSize)) * (dW)
                newWeights = W + change
                self.fc1.weight = nn.Parameter(newWeights)

    def rotate(self, d, img):
        """
        Sister function to the the one in utils.rotate
        Converts the degrees into columns and rotates the image.
        Positive degrees rotate the image clockwise
        and negative degrees rotate the image counter clockwise
        :param d: number of degrees the agent will rotate its view
        :param image: An tensor that we want to shift.
        :return: Returns the rotated image.
        """
        cols_to_shift = int(round(d * self.num_of_cols_perdegree))
        return torch.roll(img, -cols_to_shift, dims=1)

    def get_heading(self, query_img):
        #TODO: here we need custom RMF or to use teh INfomax a matcher in the utils module
        # probabaly best to make a rotator here...?
        query_img = torch.unsqueeze(torch.from_numpy(query_img).float(), 0)
        query_img = self.Standardize(query_img)
        rot_qimgs = torch.empty((self.total_search_angle, self.num_of_rows, self.num_of_cols),  requires_grad=False)
        rsims = []
        for i, rot in enumerate(self.degrees):
            rot_qimgs[i] = self.rotate(rot, query_img)
            rsims.append(self.Familiarity((query_img)))
        rsim = self.Familiarity(rot_qimgs)
        return rsim.squeeze().detach().numpy()


def Train(modelName, trainDataset, infomaxParams):
    model_path=infomaxParams.model_path

    # Set up input size
    input_size = trainDataset[0].image.size()[1] * trainDataset[0].image.size()[2]

    # Create network
    if infomaxParams.loadModel == None:
        net = InfomaxNetwork(input_size, infomaxParams)

    # OR continue training on an existing model, will overwrite the passed in params with the original training params
    else:
        net = LoadModel(infomaxParams.model_path+infomaxParams.loadModel)

    for i in range(0, len(trainDataset)):
        net.TrainNet([trainDataset[i].image])

    #print('Trained on '+str(len(train_dataset))+ ' images')

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if infomaxParams.saveModel:
        torch.save((net.state_dict(),infomaxParams), model_path + modelName)
    return net

def LoadModel(modelPath):
    data = torch.load(modelPath)
    state=data[0]
    params=data[1]
    input_dimensions = state['fc1.weight'].size()[1]
    net = InfomaxNetwork(input_dimensions, params)
    net.load_state_dict(state)
    return (net)