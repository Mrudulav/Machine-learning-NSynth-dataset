import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.utils.data as torch_data
import torchvision.transforms as transforms
import math
import os
import time
import matplotlib.ticker as ticker
import pytorch_nsynth.nsynth as NSynth
from sklearn.metrics import confusion_matrix

#defining the network
class Net(nn.Module):
    
     def __init__(self):
        super(Net, self).__init__()
        #1input image channel,6 output channels,5*5 square convolution
        #kernel
#        self.conv1 = nn.Conv1d(1, 6, 5) #in channles, out channels, kernel size
#        self.conv2 = nn.Conv1d(6, 16, 5)
        #an affine operation:y=Wx+b
        self.fc1 = nn.Linear(16000, 2048) #in features, out features
        self.fc2 = nn.Linear(2048, 256)
        self.fc3 = nn.Linear(256, 3)

     def forward(self, x):
        x=x.float()
        #max pooling over a (2,2) window
#        x=F.max_pool1d(F.relu(self.conv1(x)),2,2)
##        x = self.pool(F.relu(self.conv1(x)))
#        #If affine 
#        x=F.max_pool1d(F.relu(self.conv2(x)),2,2)
        
        x=x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x),dim=1)
        return x
#        
     def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
   
######################################################################################################    
def get_loader(path):
    #audio files are loaded as an int16 numpy array
    #rescaling intensity of float[-1,1]
    selectColmns=transforms.Lambda(lambda x: x[0:16000])
    toFloat=transforms.Lambda(lambda x: x/np.iinfo(np.int16).max)
    dataset=NSynth.NSynth(
            path,
            transform=transforms.Compose([selectColmns,toFloat]),#blaclkist string instrument
            categorical_field_list=["instrument_family","instrument_source"])
    return torch_data.DataLoader(dataset,batch_size=64,shuffle=True)
 
def plot (training_losses,validation_losses,EPOCHS):
    plt.figure(figsize=(20,10))
    x=np.linspace(1,EPOCHS,EPOCHS)
    training_losses=np.array(training_losses)
    validation_losses=np.array(validation_losses)
    plt.title("Learning curve over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Avg loss")
#    plt.annotate('Overfit point',
#                 xy=(min_loss_epoch, min_loss), xycoords='data',
#                 xytext=(0.8, 0.8), textcoords='axes fraction',
#                 arrowprops=dict(facecolor='black', shrink=0.05),
#                 horizontalalignment='left', verticalalignment='top')
    plt.plot(x, training_losses, color='purple', marker=".", label='Training loss')
    plt.plot(x, validation_losses, color='orange', marker=".", label='Validation loss')
    plt.legend()
    plt.savefig('./images/Learning_bonus.png')
#    plt.close()
    pass    
######################################################################################################    

def train(net,device):
    #train parameters
    EPOCHS = 20
    #BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    MOMENTUM = 0.6
#   
    #train_loader=get_loader("./nsynth-test")  
    train_loader=get_loader("/local/sandbox/nsynth/nsynth-train")
    #validation_loader=get_loader("./nsynth-test")
    validation_loader=get_loader("/local/sandbox/nsynth/nsynth-valid")
    print("Loading data complete")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    #####################################################################################
    training_losses=[]
    validation_losses=[]
    best_acc=0.0
    best_model_wts=None
    min_loss_epoch=0
    min_loss=math.inf
    labels_set=set()
    classes_found = []
    extreme_prob = []
    for epoch in range(1, EPOCHS + 1):
        print("Epoch:", epoch)
        #for phase in ['train', 'val']:
        for phase in ['train','val']:
            if phase == 'train':
                net.train()  # Set model to training mode
                loader = train_loader
                #dataset_size = len(training_set)
            else:
                net.eval()  # Set model to evaluate mode
                loader = validation_loader
            dataset_size = len(loader)

            running_loss = 0.0
            running_corrects = 0
            
            for samples, instrument_family_target,instrument_source_target, targets in loader:
                

            #for i, data in enumerate(loader, 0):
                # Get the inputs
                #inputs, labels = data
                inputs, labels = samples.to(device), instrument_source_target.to(device)
                inputs=inputs.unsqueeze(1)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    _, preds = torch.max(outputs, 1)
                    for index in range(len(outputs)):
                        if outputs[index][preds[index]]>0.6 or outputs[index][preds[index]] < 0.1:
                            if preds[index] not in classes_found:
                                extreme_prob.append(inputs[index])
                                classes_found.append(preds[index])
                    #loss = criterion(F.softmax(outputs, dim=1), labels)
                    loss=criterion(outputs,labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                for i in range(len(labels)):
                    labels_set.add(labels[i])

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = float(running_loss) / dataset_size
            epoch_acc = float(running_corrects) / dataset_size

            if phase == 'train':
                training_losses.append(epoch_loss)
            else:
                validation_losses.append(epoch_loss)
                if min_loss > epoch_loss:
                    min_loss = epoch_loss
                    min_loss_epoch = epoch

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                #best_model_wts = copy.deepcopy(net.state_dict())

        print("best accuracy", best_acc)
        #print("=" * 50)

    print('Finished Training')
    #print(extreme_prob)
    #print(classes_found)
    #plot(training_losses,validation_losses,EPOCHS,min_loss_epoch,min_loss,IMAGE_PATH+'training_learning_curve.png')
    ##########################################################################################
#Saving the weights of the trained network to a text file that are human readable

    weights = list(net.parameters())

    with open('Bonusweight.txt', 'w') as f:
        for item in weights:
            f.write("%s\n" % item)
            

def test(net,device):
    #test_loader=get_loader("./nsynth-test")
    test_loader=get_loader("/local/sandbox/nsynth/nsynth-test")
    
    correct = 0 
    total = 0 
    classes=set()
    with torch.no_grad():
        #for data in testloader:
        for samples, instrument_family_target,instrument_source_target, targets in test_loader:
            for label in instrument_source_target:
                classes.add(label.item())
            #images, labels = data
            inputs, labels = samples.to(device), instrument_source_target.to(device)
            inputs=inputs.unsqueeze(1)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print('\nAccuracy of the network on the test audio: %d %%\n' %(
            100 * correct / total))
    
    
    classes=list(classes)
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    
    pred = []
    for ind1 in range(3):
        predicted = []
        for ind2 in range(3):
            predicted.append(ind2)
        pred.append(predicted)
    #extreme_prob=[]
    
    #logit=model(x)
        #to calculate loss using probabilities
    #loss = torch.nn.functional.nll_loss(torch.log(p),y)
    
    
    with torch.no_grad():
        #4 tensors
        for samples, instrument_family_target,instrument_source_target, targets in test_loader:
            inputs, labels = samples.to(device), instrument_source_target.to(device)
            inputs=inputs.unsqueeze(1)
            outputs = net(inputs)
            #print(outputs)
           # p = torch.nn.functional.softmax(outputs,dim=1)
            #print(predicted)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                #plt.figure()
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                pred[label][predicted[i]]+=1
                
    classes_accuracy=[]            
    for i in range(len(classes)):
        #accuracy2.append(100 * class_correct[i] / class_total[i])
        print('Accuracy of %5s : %2d %%' % (classes[i],100 * class_correct[i] / class_total[i]))
        classes_accuracy.append(100 * class_correct[i] / class_total[i])
        
#    plt.imshow(pred, cmap='hot', interpolation='nearest')
#    plt.show()
    #torch.save(net.state_dict(),'Bonusweights')    
    
    #net = torch.load('q1_model')
    
    torch.save(net,'BonusModel') 
    #print(class_accuracy)

#C:/Users/Mrudula/.spyder-py3            
#####################################################################################
    
def main():
    
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(device)
    
    net=Net()
    net.to(device)
    
    t0=time.time()
    train(net,device)
    test(net,device)
    t1=time.time()
    print((t1-t0)/60)  
    #plot()
    pass



if __name__ == '__main__':
    main()


#C:\Users\Mrudula\.spyder-py3\Proj1



