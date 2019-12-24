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
class Encoder_net(nn.Module):
    def __init__(self, channel=200):
        super(Encoder_net, self).__init__()
        self.channel = channel
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=24, kernel_size=(20, 3, 3), stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(num_features=24)
        self.prelu1=nn.PReLU(num_parameters=1, init=0.25)
        self.conv2 = nn.Conv3d(in_channels=24, out_channels=48, kernel_size=(20, 3, 3), stride=1, padding=0)
        self.bn2 = nn.BatchNorm3d(num_features=48)
        self.prelu2=nn.PReLU(num_parameters=1, init=0.25)
        self.pool2 = nn.MaxPool3d(kernel_size=(18, 1, 1), stride=(18, 1, 1))
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x=self.prelu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x=self.prelu2(x)
        x = self.pool2(x)
        return x
class Decoder_net(nn.Module):
    def __init__(self, channel=200):
        super(Decoder_net, self).__init__()
        self.channel = channel
        self.deconv3 = nn.ConvTranspose3d(in_channels=48, out_channels=24, kernel_size=(18, 3, 3), stride=(18, 1, 1),
                                          padding=(0, 0, 0))  # H_{out}=(H_{in}-1)stride-2padding+kernel_size+output_padding
        self.bn3 = nn.BatchNorm3d(num_features=24)
        self.prelu3=nn.PReLU(num_parameters=1, init=0.25)
        self.deconv4 = nn.ConvTranspose3d(in_channels=24, out_channels=1, kernel_size=(39, 3, 3), stride=(1, 1, 1),
                                          padding=0)
        self.bn4 = nn.BatchNorm3d(num_features=1)

    def forward(self, x):
        x = self.deconv3(x)
        x = self.bn3(x)
        x=self.prelu3(x)
        x = self.deconv4(x)
        x = self.bn4(x)
        return x
class Discriminant(nn.Module):
    def __init__(self):
        super(Discriminant, self).__init__()
        self.lin1 = nn.Linear(432, 512)
        self.lin2 = nn.Linear(512, 512)
        self.lin3 = nn.Linear(512, 1)  ## output if (1*2)  assuming  first neuron is giving out P(Y=1|X=x)  and Second P(Y=0 | X=x)
                                      ## Y=1 ==> True image   Y=0 ==> Fake images
    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
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

    return data, labels


def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test

def padWithZeros(X, margin=2):
    newX=np.pad(X,pad_width=((margin,margin),(margin,margin),(0,0)),mode='symmetric')
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
def trainSVM(model,Datapath,Labelpath,test_ratio_SVM):
    train_data = MYDataset(Datapath, Labelpath,trans)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=32, shuffle=False)
    NewFeature=[]
    model.eval()
    model = model.cuda()
    for data, label in train_loader:
        data = data.cuda()
        feature = model(data)
        # feature = torch.sigmoid(feature)
        for num in range(len(feature)):
            NewFeature.append(feature.view(-1,432)[num].cpu().detach().numpy())
    ytrain_fea=np.load(Labelpath)
    Xtrain_fea=NewFeature

    class_weight = 'balanced'
    clf = sklearn.svm.SVC(class_weight=class_weight,probability=True,gamma='auto',kernel='linear')

    clf = sklearn.model_selection.GridSearchCV(clf, SVM_GRID_PARAMS, scoring=None, n_jobs=8, iid=True,
                                              refit=True, cv=3, verbose=3, pre_dispatch='2*n_jobs',
                                              error_score='raise', return_train_score=True)

    clf.fit(Xtrain_fea, ytrain_fea)
    print(clf.best_params_)
    joblib.dump(clf, 'SVM.model')
    return 0
def predict(model,model_SVM,Datapath,Labelpath):
    clf = model_SVM
    test_data = MYDataset(Datapath, Labelpath,trans)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=32, shuffle=False)
    NewFeature = []
    Prediction=[]
    model.eval()
    model = model.cuda()
    for data, label in test_loader:
        data = data.cuda()
        feature = model(data)
        # feature = torch.sigmoid(feature)
        for num in range(len(feature)):
            f=feature.view(-1, 432)[num].cpu().detach().numpy()
            NewFeature.append(f)
            Prediction.append(clf.predict(f.reshape(-1,432)))
    return Prediction
def evaluate(model,model_SVM,Datapath,Labelpath):
    clf = model_SVM
    test_data = MYDataset(Datapath, Labelpath,trans)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=32, shuffle=False)
    NewFeature = []
    model.eval()
    model = model.cuda()
    for data, label in test_loader:
        data = data.cuda()
        feature = model(data)
        # feature = torch.sigmoid(feature)
        for num in range(len(feature)):
            NewFeature.append(feature.view(-1,432)[num].cpu().detach().numpy())

    Xtest_fea = NewFeature
    score=np.zeros(2)
    score[0]=np.mean(clf.predict_proba(Xtest_fea))
    score[1]=clf.score(Xtest_fea,np.load(Labelpath).astype(int))
    return score
# SVM_GRID_PARAMS = [{'kernel': ['rbf'], 'gamma': np.arange(0.0003,0.0005,0.00001),'C': np.arange(610,630,1)}]
SVM_GRID_PARAMS = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3],
                    'C': [1, 10, 100, 1000]},
                   {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]},
                   {'kernel': ['poly'], 'degree': [3], 'gamma': [1e-1, 1e-2, 1e-3]}]
TRAIN=1
TRAINSVM=True
dataset = 'IP'
# test_ratio = 0.001
test_ratio_SVM=0.9#10201
windowSize = 5
X, y = loadData(dataset)
K=X.shape[2]
trans = transforms.Compose(transforms = [
    transforms.ToTensor(),
    transforms.Normalize(np.zeros(K),np.ones(K))
])
X, y = createImageCubes(X, y, windowSize=windowSize)
def feature_normalize(data):
    mu = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    return (data - mu)/std
X=feature_normalize(X)
np.save('Xtrain.npy',X)
np.save('ytrain.npy',y)

Xtrain_SVM, Xtest_SVM, ytrain_SVM, ytest_SVM=splitTrainTestSet(X, y, test_ratio_SVM)

np.save('Xtrain_SVM.npy',Xtrain_SVM)
np.save('ytrain_SVM.npy',ytrain_SVM)
np.save('Xtest_SVM.npy',Xtest_SVM)
np.save('ytest_SVM.npy',ytest_SVM)
epochs=100
batch_size =128
encoder = Encoder_net()
decoder = Decoder_net()
discriminant = Discriminant()
gen_lr = 0.001
reg_lr = 0.00001
optim_dec = torch.optim.Adam(decoder.parameters(), lr=gen_lr)
optim_enc = torch.optim.Adam(encoder.parameters(), lr=gen_lr)
#regularizing optimizers ADAM
optim_enc_gen = torch.optim.Adam(encoder.parameters(), lr=0.002,betas=[0.1,0.999])
optim_disc = torch.optim.SGD(discriminant.parameters(), lr=reg_lr)
loss= nn.BCELoss()
loss1 = nn.MSELoss()
loss2 = nn.BCELoss()

recon_loss=[]
discr_loss=[]
genra_loss=[]
## parameters  mean=[0,0] covariance matrix =[[1,0],[0,1]]
def generate_(batch_size):
    return torch.from_numpy(np.random.multivariate_normal(mean=np.zeros([432]),cov =np.diag(np.ones([432])),size=batch_size)).type(torch.float)

if TRAIN==True:
    train_data = MYDataset('Xtrain.npy','ytrain.npy',trans)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        dl = 0
        rl = 0
        gl = 0
        print(" Epoch No  {} ".format(epoch))
        for i, (data, label) in enumerate(train_loader):
            batch_size=data.shape[0]
            data=data.cuda()
            label=label.cuda()
            encoder = encoder.cuda()
            decoder = decoder.cuda()
            discriminant = discriminant.cuda()

            ## Getting True And Fake Lables Ready
            ###################################################

            ## True Data ==>  one coming out of encoder

            true_labels = np.zeros(shape=(batch_size, 2))
            true_labels[:,0] = 1.0  ## As defined in network architecture 1st neuron if for true images
            true_labels = torch.from_numpy(true_labels).type(torch.FloatTensor).cuda()

            ## Fake Data  ==> data samples from bivarient Gaussian  distribution

            fake_labels = np.zeros(shape=(batch_size, 2))
            fake_labels[:,1] = 1.0
            fake_labels = torch.from_numpy(fake_labels).type(torch.FloatTensor).cuda()

            #####################################################
            ## resetting  grads to zero
            optim_dec.zero_grad()
            optim_enc.zero_grad()
            optim_disc.zero_grad()
            optim_enc_gen.zero_grad()

            ##############################

            ## Reconstruction Phase
            enc_out = encoder(data)

            dec_out = decoder(enc_out)
            reco = loss1(data, dec_out)
            reco.backward()
            optim_dec.step()
            optim_enc.step()

            #######################

            ## Regularization Phase
            discriminant.train()
            fake_data = torch.Tensor(generate_(batch_size)).cuda()
            fake_pred = discriminant(fake_data)
            true_data = encoder(data)
            true_data=true_data.view(batch_size,-1)
            true_pred = discriminant(true_data)

            dis_loss=-(torch.mean(fake_pred) - torch.mean(true_pred))

            dis_loss.backward(retain_graph=True)
            optim_disc.step()
            discriminant.eval()  ## No Further training of discriminator

            encl=-(torch.mean(true_pred)) ## Since we are not updating dicriminator the encoder will want to resemble its distribution with the
            ## Bivarient Gaussian ==> Fooling the Discriminator
            encl.backward(retain_graph=True)
            optim_enc_gen.step()

            dl = dl + dis_loss.item()
            rl = rl + reco.item()
            gl = gl + encl.item()
            if (i % 8 == 0):
                print(" Iteration No : {}  Reconsturction loss : {}   Discrimant loss : {}".format(i, reco.item(),
                                                                                                   dis_loss.item()))

        print("Epoch : {} Complete".format(epoch))
        torch.save(encoder.state_dict(), 'AAE_encoder.pth')
else:
    encoder.load_state_dict(torch.load('AAE_encoder.pth'))





Datapath='Xtrain_SVM.npy'
Labelpath='ytrain_SVM.npy'
if TRAINSVM==True:
    trainSVM(encoder,Datapath,Labelpath,test_ratio_SVM)
    model_SVM = joblib.load('SVM.model')
else:
    model_SVM=joblib.load('SVM.model')
Datapath='Xtest_SVM.npy'
Labelpath='ytest_SVM.npy'
Y_pred = predict(encoder,model_SVM,Datapath,Labelpath)
classification = classification_report(np.load(Labelpath).astype(int), Y_pred)
print(classification)
def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def reports(model,model_SVM,Datapath, Labelpath, name):
    # start = time.time()
    y_pred = predict(model,model_SVM,Datapath,Labelpath)
    # y_pred = np.argmax(np.array(Y_pred), axis=1)
    # end = time.time()
    # print(end - start)
    Label=np.load(Labelpath).astype(int)
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

    classification = classification_report(Label, y_pred, target_names=target_names)
    oa = accuracy_score(Label, y_pred)
    confusion = confusion_matrix(Label, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(Label, y_pred)
    score = evaluate(model,model_SVM,Datapath,Labelpath)
    Test_Loss = score[0] * 100
    Test_accuracy = score[1] * 100
    return classification, confusion, Test_Loss, Test_accuracy, oa * 100, each_acc * 100, aa * 100, kappa * 100


classification, confusion, Test_loss, Test_accuracy, oa, each_acc, aa, kappa = reports(encoder,model_SVM,Datapath, Labelpath, dataset)
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


def Patch(data, height_index, width_index):
    height_slice = slice(height_index, height_index + PATCH_SIZE)
    width_slice = slice(width_index, width_index + PATCH_SIZE)
    patch = data[height_slice, width_slice, :]

    return patch
#

# load the original image
X, y = loadData(dataset)
height = y.shape[0]
width = y.shape[1]
PATCH_SIZE = windowSize
X = padWithZeros(X, PATCH_SIZE // 2)
# calculate the predicted image
outputs = np.zeros((height, width))
for i in range(height):
    for j in range(width):
        target = int(y[i, j])
        if target == 0:
            continue
        else:
            image_patch = Patch(X, i, j)
            X_test_image = image_patch.reshape(1,image_patch.shape[0], image_patch.shape[1],image_patch.shape[2]).astype('float32')
            np.save('WholePic.npy',X_test_image)
            Datapath='WholePic.npy'
            Labelpath='WholePic.npy'
            prediction = predict(encoder,model_SVM,Datapath,Labelpath)
            prediction=int(prediction[0])
            outputs[i][j] = prediction + 1
ground_truth = spectral.imshow(classes=y, figsize=(7, 7))
predict_image = spectral.imshow(classes=outputs.astype(int), figsize=(7, 7))
spectral.save_rgb("predictions.jpg", outputs.astype(int), colors=spectral.spy_colors)
spectral.save_rgb(str(dataset) + "_ground_truth.jpg", y, colors=spectral.spy_colors)
torch.cuda.empty_cache()