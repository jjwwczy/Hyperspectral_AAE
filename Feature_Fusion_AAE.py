import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
import sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import scale
import os
import spectral
from torchsummary import summary
import copy
from torchvision import transforms, datasets
from torchvision.utils import save_image
from torch.autograd import Variable
import pandas as pd
import numpy as np
from temp import ConvBNRelu2D, ConvBNRelu3D, Decoder_net, Encoder_net
from visdom import Visdom

class Discriminant(nn.Module):
    def __init__(self, output_units=16):
        super(Discriminant, self).__init__()
        self.lin1 = nn.Linear(256, 512)
        self.lin2 = nn.Linear(512, 64)
        self.lin3 = nn.Linear(64,
                              1)  ## output if (1*2)  assuming  first neuron is giving out P(Y=1|X=x)  and Second P(Y=0 | X=x)
        ## Y=1 ==> True image   Y=0 ==> Fake images

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        # x = F.relu(self.lin3(x))
        return torch.tanh(x)


class ClassDiscriminant(nn.Module):
    def __init__(self, output_units=16):
        super(ClassDiscriminant, self).__init__()
        self.lin1 = nn.Linear(output_units, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128,
                              1)  ## output if (1*2)  assuming  first neuron is giving out P(Y=1|X=x)  and Second P(Y=0 | X=x)
        ## Y=1 ==> True image   Y=0 ==> Fake images

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        # x = F.relu(self.lin3(x))
        return torch.tanh(x)


def loadData(name):
    data_path = os.path.join(os.getcwd(), 'data')
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SA':
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    elif name == 'PU':
        data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
    elif name == 'KSC':
        data = sio.loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
        labels = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']

    return data, labels


def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca


def splitTrainTestSet(X, y, testRatio, randomState=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test


def padWithZeros(X, margin=2):
    newX = np.pad(X, pad_width=((margin, margin), (margin, margin), (0, 0)), mode='symmetric')
    return newX


def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]), dtype=np.float32)
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]), dtype=np.float32)
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels


class MYDataset(torch.utils.data.Dataset):#需要继承data.Dataset
    def __init__(self,Datapath,Labelpath,transform):
        # 1. Initialize file path or list of file names.
        self.Datalist=np.load(Datapath)
        self.Labellist=(np.load(Labelpath)).astype(int)
        self.transform=transform
    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data

        index=index
        Data=self.transform(self.Datalist[index])
        Data=Data.view(1,Data.shape[0],Data.shape[1],Data.shape[2])
        return Data ,self.Labellist[index]
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.Datalist)
def predict(model, Datapath, Labelpath):
    model.eval()
    model = model.cuda()
    test_data = MYDataset(Datapath, Labelpath,trans)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    prediction = []
    for data, label in test_loader:
        data = data.cuda()
        feature, out = model(data)
        for num in range(len(out)):
            prediction.append(np.array(out[num].cpu().detach().numpy()))
    return prediction


def evaluate(model, Datapath, Labelpath):
    model.eval()
    model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    test_data = MYDataset(Datapath, Labelpath,trans)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    score = np.zeros(2)
    train_loss = 0.
    train_acc = 0.
    index = 0
    for data, label in test_loader:
        data = data.cuda()
        feature, out = model(data)
        label = label.cuda()
        loss = criterion(out, label)
        train_loss += loss.item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred == label).sum()
        train_acc += train_correct.item()
    score[0] = train_loss / (len(test_data))
    score[1] = train_acc / (len(test_data))
    return score


TRAIN =0
dataset = 'IP'
test_ratio = 10201-48
windowSize = 25
X, y = loadData(dataset)
K = X.shape[2]
K = 30 if dataset == 'IP' else 15
trans = transforms.Compose(transforms = [
    transforms.ToTensor(),
    transforms.Normalize(np.zeros(K),np.ones(K))
])
output_units = 9 if (dataset == 'PU' or dataset == 'PC') else 16
X, pca = applyPCA(X, numComponents=K)
X, y = createImageCubes(X, y, windowSize=windowSize)
def feature_normalize(data):
    mu = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    return (data - mu)/std
X=feature_normalize(X)
np.save('X.npy', X)
np.save('y.npy', y)
Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X, y, test_ratio)
np.save('Xtrain.npy', Xtrain)
np.save('ytrain.npy', ytrain)
np.save('Xtest.npy', Xtest)
np.save('ytest.npy', ytest)
epochs =60
Threshold=0.8*epochs
batch_size = 128
encoder = Encoder_net(channel=K, windowSize=windowSize, output_units=output_units)
decoder = Decoder_net(channel=K, windowSize=windowSize, output_units=output_units)
discriminant = Discriminant(output_units=output_units)
classdiscriminant = ClassDiscriminant(output_units=output_units)
gen_lr = 1e-3
reg_lr = 5e-6
optim_dec = torch.optim.Adam(decoder.parameters(), lr=gen_lr)#1e-5
optim_enc = torch.optim.Adam(encoder.parameters(), lr=gen_lr)#1e-5
# regularizing optimizers ADAM
optim_enc_gen = torch.optim.SGD(encoder.parameters(), lr=1e-5)#1e-2 betas=[0.1,0.999]
optim_disc = torch.optim.SGD(discriminant.parameters(), lr=reg_lr)#1e-3   ,betas=[0.1,0.999]
optim_classdisc = torch.optim.SGD(classdiscriminant.parameters(), lr=reg_lr)#1e-3  ,betas=[0.1,0.999]
loss = nn.BCELoss()
loss1 = nn.MSELoss()
loss2 = nn.BCELoss()
loss3= nn.BCELoss()


## parameters  mean=[0,0] covariance matrix =[[1,0],[0,1]]
def generate_(batch_size):
    return torch.from_numpy(
        np.random.multivariate_normal(mean=np.zeros([256 ]), cov=np.diag(np.ones([256])),
                                      size=batch_size)).type(torch.float)


def select_sample(batch_size, output_units=16):
    label = np.random.randint(0, output_units, (batch_size))
    categorical = torch.nn.functional.one_hot(torch.LongTensor(label), num_classes=output_units)
    return categorical
viz = Visdom()
viz.line([0.], [0.], win='reco_loss', opts=dict(title='Reconstruction loss'))
viz.line([0.], [0.], win='classi_loss', opts=dict(title='Classification loss'))
if TRAIN == True:
    train_data = MYDataset('Xtest.npy', 'ytest.npy',trans)#unsupervised part
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    classi_data = MYDataset('Xtrain.npy', 'ytrain.npy',trans)#supervised part
    classi_loader = torch.utils.data.DataLoader(dataset=classi_data, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=gen_lr, weight_decay=0.00005)
    criterion = torch.nn.CrossEntropyLoss()

    best_loss = 10000
    best_model_wts = copy.deepcopy(encoder.state_dict())
    best_acc = 0
    vis_step=0
    vis_class_step=0
    for epoch in range(epochs):
        dl = 0
        rl = 0
        gl = 0
        classdl = 0
        classi_loss = 0
        train_acc = 0
        print(" Epoch No  {} ".format(epoch))

        for i, (data, label) in enumerate(classi_loader):#supervised part

            batch_size = data.shape[0]
            data = data.cuda()
            label = label.cuda()
            encoder = encoder.cuda()
            decoder = decoder.cuda()
            decoder.train()
            discriminant = discriminant.cuda()
            classdiscriminant = classdiscriminant.cuda()


            #####################################################
            ## resetting  grads to zero
            optim_dec.zero_grad()
            optim_enc.zero_grad()
            optim_disc.zero_grad()
            optim_enc_gen.zero_grad()
            optim_classdisc.zero_grad()
            ##############################
            ## Reconstruction Phase
            enc_out, classicode = encoder(data)
            dec_out = decoder(enc_out, classicode)
            reco = loss1(data, dec_out)
            reco.backward(retain_graph=True)
            optim_dec.step()
            optim_enc.step()
            decoder.eval()

            #######################
            ## Regularization Phase
            encoder.eval()
            discriminant.train()
            classdiscriminant.train()
            fake_data = torch.Tensor(generate_(batch_size)).cuda()
            fake_pred = discriminant(fake_data)
            true_data=  enc_out
            fake_sample=F.softmax(classicode,dim=1)
            fake_pred_class = classdiscriminant(fake_sample)
            true_data = true_data.view(batch_size, -1)
            true_sample = select_sample(batch_size, output_units=output_units).float().cuda()
            true_sample = true_sample.view(batch_size, -1)
            true_pred = discriminant(true_data)
            true_pred_class = classdiscriminant(true_sample)
            dis_loss=-(torch.mean(fake_pred) - torch.mean(true_pred))
            classdiscr_loss = -(torch.mean(true_pred_class) - torch.mean(fake_pred_class))
            dis_loss.backward(retain_graph=True)
            classdiscr_loss.backward(retain_graph=True)
            optim_disc.step()
            optim_classdisc.step()
            #Fool the discriminant
            # freeze the discriminant and train the encoder(generator)
            discriminant.eval()
            classdiscriminant.eval()
            encoder.train()
            # encl=-0.25*(3*torch.mean(true_pred)+torch.mean(fake_pred_class))
            encl=-(torch.mean(true_pred)+torch.mean(fake_pred_class))

            encl.backward(retain_graph=True)
            optim_enc_gen.step()
            classdl = classdl + classdiscr_loss.item()
            dl = dl + dis_loss.item()
            rl = rl + reco.item()
            gl = gl + encl.item()
            vis_step += 1

            viz.line([reco.item()], [vis_step], win='reco_loss', update='append')

            if (i % 16 == 0):
                print(
                    " Iteration No : {}  Reconsturction loss : {} Discriminant loss : {}   ClassDiscriminant loss : {}  Generator Loss {}".format(
                        i, reco.item(), dis_loss.item(),
                     classdiscr_loss.item(), encl.item()))
        ## semi-supervised classification Phase

            current_loss = criterion(classicode, label)
            classi_loss += current_loss.item()
            pred = torch.max(classicode, 1)[1]
            train_correct = (pred == label).sum()
            train_acc += train_correct.item()
            optimizer.zero_grad()
            current_loss.backward(retain_graph=True)
            optimizer.step()
        vis_class_step += 1
        viz.line([current_loss.item()], [vis_class_step], win='classi_loss', update='append')
        print(
            'Train Loss: {:.6f}, Acc: {:.6f}'.format(classi_loss / (len(classi_data)), train_acc / (len(classi_data))))
        acc = (train_acc / (len(classi_data)))
        cur_loss = (classi_loss / (len(classi_data)))+dis_loss.item()
        if (best_acc <= acc and best_loss > cur_loss):
            best_acc = acc
            best_loss = cur_loss
            best_model_wts = copy.deepcopy(encoder.state_dict())
            print("save a model with acc:{}".format(best_acc))
            torch.save(best_model_wts, dataset+'_'+'FF_AAE_encoder.pth')
        for i, (data, label) in enumerate(train_loader): #unsupervised part

            batch_size = data.shape[0]
            data = data.cuda()
            label = label.cuda()
            encoder = encoder.cuda()
            decoder = decoder.cuda()
            decoder.train()
            ## resetting  grads to zero
            optim_dec.zero_grad()
            optim_enc.zero_grad()
            optim_disc.zero_grad()
            optim_enc_gen.zero_grad()
            optim_classdisc.zero_grad()
            ##############################
            ## Reconstruction Phase
            enc_out, classicode = encoder(data)
            dec_out = decoder(enc_out, classicode)
            reco = loss1(data, dec_out)
            reco.backward(retain_graph=True)
            optim_dec.step()
            optim_enc.step()
            decoder.eval()

            #######################
            ## Regularization Phase
            encoder.eval()
            discriminant.train()
            classdiscriminant.train()
            fake_data = torch.Tensor(generate_(batch_size)).cuda()
            fake_pred = discriminant(fake_data)
            true_data=  enc_out
            fake_sample=F.softmax(classicode,dim=1)
            fake_pred_class = classdiscriminant(fake_sample)
            true_data = true_data.view(batch_size, -1)
            true_sample = select_sample(batch_size, output_units=output_units).float().cuda()
            true_sample = true_sample.view(batch_size, -1)
            true_pred = discriminant(true_data)
            true_pred_class = classdiscriminant(true_sample)
            # dis_loss = loss(fake_pred, fake_labels) + loss(true_pred, true_labels)
            dis_loss=-(torch.mean(fake_pred) - torch.mean(true_pred))
            classdiscr_loss = -(torch.mean(true_pred_class) - torch.mean(fake_pred_class))
            dis_loss.backward(retain_graph=True)
            classdiscr_loss.backward(retain_graph=True)
            optim_disc.step()
            optim_classdisc.step()
            #Fool the discriminant
            # freeze the discriminant and train the encoder(generator)
            discriminant.eval()
            classdiscriminant.eval()
            encoder.train()
            # encl=-0.25*(3*torch.mean(true_pred)+torch.mean(fake_pred_class))
            encl=-(torch.mean(true_pred)+torch.mean(fake_pred_class))
            encl.backward(retain_graph=True)
            optim_enc_gen.step()
            classdl = classdl + classdiscr_loss.item()
            dl = dl + dis_loss.item()
            rl = rl + reco.item()
            gl = gl + encl.item()
            vis_step += 1
            viz.line([reco.item()], [vis_step], win='reco_loss', update='append')
            if (i % 16 == 0):
                print(
                " Iteration No : {}  Reconsturction loss : {} Discriminant loss : {}   ClassDiscriminant loss : {}  Generator Loss {}".format(
                    i, reco.item(), dis_loss.item(),
                    classdiscr_loss.item(), encl.item()))

    print("Epoch : {} Complete".format(epoch))
    torch.save(best_model_wts, dataset+'_'+'FF_AAE_encoder.pth')
else:
    encoder.load_state_dict(torch.load(dataset+'_'+'FF_AAE_encoder.pth'))

Datapath = 'Xtest.npy'
Labelpath = 'ytest.npy'
encoder.load_state_dict(torch.load(dataset+'_'+'FF_AAE_encoder.pth'))
Y_pred_test = predict(encoder, Datapath, Labelpath)
y_pred_test = np.argmax(np.array(Y_pred_test), axis=1)
classification = classification_report(ytest.astype(int), y_pred_test)
print(classification)


def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def reports(model, X_test, y_test, name):
    # start = time.time()
    Datapath = 'Xtest.npy'
    Labelpath = 'ytest.npy'
    Y_pred = predict(model, Datapath, Labelpath)
    y_pred = np.argmax(np.array(Y_pred), axis=1)
    # end = time.time()
    # print(end - start)
    if name == 'IP':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
            , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif name == 'SA':
        target_names = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
                        'Fallow_smooth',
                        'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
                        'Vinyard_untrained', 'Vinyard_vertical_trellis']
    elif name == 'PU':
        target_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']
    elif name == 'KSC':
        target_names = ["Scrub", "Willow swamp",
                        "Cabbage palm hammock", "Cabbage palm/oak hammock",
                        "Slash pine", "Oak/broadleaf hammock",
                        "Hardwood swamp", "Graminoid marsh", "Spartina marsh",
                        "Cattail marsh", "Salt marsh", "Mud flats", "Wate"]

    classification = classification_report(y_test.astype(int), y_pred, target_names=target_names)
    oa = accuracy_score(y_test.astype(int), y_pred)
    confusion = confusion_matrix(y_test.astype(int), y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test.astype(int), y_pred)
    score = evaluate(model, Datapath, Labelpath)
    Test_Loss = score[0] * 100
    Test_accuracy = score[1] * 100
    return classification, confusion, Test_Loss, Test_accuracy, oa * 100, each_acc * 100, aa * 100, kappa * 100


classification, confusion, Test_loss, Test_accuracy, oa, each_acc, aa, kappa = reports(encoder, Xtest, ytest, dataset)
classification = str(classification)
file_name = "classification_report.txt"
with open(file_name, 'w') as x_file:
    x_file.write('{} Test loss (%)'.format(Test_loss))
    x_file.write('\n')
    x_file.write('{} Test accuracy (%)'.format(Test_accuracy))
    x_file.write('\n')
    x_file.write('\n')
    x_file.write('{} Kappa accuracy (%)'.format(kappa))
    x_file.write('\n')
    x_file.write('{} Overall accuracy (%)'.format(oa))
    x_file.write('\n')
    x_file.write('{} Average accuracy (%)'.format(aa))
    x_file.write('\n')
    x_file.write('\n')
    x_file.write('{}'.format(classification))
    x_file.write('\n')
    x_file.write('{}'.format(confusion.astype(str)))

# def Patch(data, height_index, width_index):
#     height_slice = slice(height_index, height_index + PATCH_SIZE)
#     width_slice = slice(width_index, width_index + PATCH_SIZE)
#     patch = data[height_slice, width_slice, :]
#
#     return patch
# #

# # load the original image
# X, y = loadData(dataset)
# height = y.shape[0]
# width = y.shape[1]
# PATCH_SIZE = windowSize
# X = padWithZeros(X, PATCH_SIZE // 2)
# # calculate the predicted image
# outputs = np.zeros((height, width))
# for i in range(height):
#     for j in range(width):
#         target = int(y[i, j])
#         if target == 0:
#             continue
#         else:
#             image_patch = Patch(X, i, j)
#             X_test_image = image_patch.reshape(1,image_patch.shape[0], image_patch.shape[1],image_patch.shape[2]).astype('float32')
#             np.save('patch.npy',X_test_image)
#             Datapath='patch.npy'
#             Labelpath='patch.npy'
#             prediction = predict(encoder,Datapath,Labelpath)
#             prediction=int(prediction[0])
#             outputs[i][j] = prediction + 1
# ground_truth = spectral.imshow(classes=y, figsize=(7, 7))
# predict_image = spectral.imshow(classes=outputs.astype(int), figsize=(7, 7))
# spectral.save_rgb("predictions.jpg", outputs.astype(int), colors=spectral.spy_colors)
# spectral.save_rgb(str(dataset) + "_ground_truth.jpg", y, colors=spectral.spy_colors)
# torch.cuda.empty_cache()
