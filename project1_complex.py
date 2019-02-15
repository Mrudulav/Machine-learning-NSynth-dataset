import os
import numpy as np
import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils.data as torch_data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import math
import matplotlib.ticker as ticker
import torch.utils.data as torch_data
import time
from pytorch_nsynth.nsynth import NSynth
#import pytorch_nsynth.nsynth as NSynth
from sklearn.metrics import confusion_matrix

#defining the network
class Net(nn.Module):
    
     def __init__(self):
        super(Net, self).__init__()
        #1input image channel,6 output channels,5*5 square convolution
        #kernel
        self.conv1 = nn.Conv1d(1, 8,5,2) #in channles, out channels, kernel size
        self.conv2 = nn.Conv1d(8,16, 5, 2)
        self.conv3 = nn.Conv1d(16,32,5,2) #in channles, out channels, kernel size
        self.conv4 = nn.Conv1d(32,64,3)
        self.bn1 = nn.BatchNorm1d(1)        
        self.bn2 = nn.BatchNorm1d(8)        
        self.bn3 = nn.BatchNorm1d(16)        
        self.bn4 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(7744, 4096) #in features, out features
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 10)

     def forward(self, x):
        x=x.float()
        #max pooling over a (2,2) window
        x=F.max_pool1d(F.relu(self.conv1(self.bn1(x))),4,2)
        x=F.max_pool1d(F.relu(self.conv2(self.bn2(x))),4,2)
        x=F.max_pool1d(F.relu(self.conv3(self.bn3(x))),4,2)
        x=F.max_pool1d(F.relu(self.conv4(self.bn4(x))),4,2)
        
        x=x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x),dim=1)
        return x
       
     def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
   
######################################################################################################    
#plot waves
def question2(data,name,title):
    plt.figure(figsize=(20,10))
    plt.plot(data)
    
    directory="./Compleximages/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.title(title)
    plt.savefig('./Compleximages/'+name)
    pass

#######################################################################################################
#gets the path
def loader_function(path):
    #audio files are loaded as an int16 numpy array
    #rescaling intensity of float[-1,1]
    selectColmns=transforms.Lambda(lambda x: x[0:16000])
    toFloat=transforms.Lambda(lambda x: x/np.iinfo(np.int16).max+1)
    dataset=NSynth(
            path,
            transform=transforms.Compose([selectColmns,toFloat]),
            blacklist_pattern=["synth_lead"],#blaclkist synth_lead instrument
            categorical_field_list=["instrument_family","instrument_source"])
    question2(dataset[0][0],"1-D_audio_waveform.png","1-D audio waveform")
    return dataset,torch_data.DataLoader(dataset,batch_size=256,shuffle=True,num_workers=16)
#########################################################################################################
#learning curves 
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
    plt.savefig('./Compleximages/Learning_curve.png')

    pass   
######################################################################################################    
#train & validate
def train(net,device):
    #train parameters
    EPOCHS = 40
    #BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    MOMENTUM = 0.8

            
    #loaders
    #train_loader=loader_function("./nsynth-test")
    train_data,train_loader=loader_function("/local/sandbox/nsynth/nsynth-train")
    #validation_loader=loader_function("./nsynth-test")
    validation_data,validation_loader=loader_function("/local/sandbox/nsynth/nsynth-valid")
    
    print("********************Loading data complete********************")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = LEARNING_RATE, momentum = MOMENTUM)
    #####################################################################################
    #train the network
    training_losses=[]
    validation_losses=[]
    
    for epoch in range(1, EPOCHS + 1):
        print("Epoch:", epoch)
        
        train_loss=0.0
        correct=0
        total=0
        for samples, instrument_family_target,instrument_source_target, targets in train_loader:
            #get the inputs
            inputs, labels = samples.to(device), instrument_family_target.to(device)
            inputs=inputs.unsqueeze(1)
            inputs=inputs.float()
            # zero the parameter gradients
            optimizer.zero_grad()
            
            outputs = net(inputs)
            loss=criterion(outputs,labels)
                    
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            
            
            total+=inputs.size(0)
            train_loss += loss.item() 
            correct += (preds == labels).sum().item()
    
        epoch_acc = float(correct)*100/ total
        epoch_loss = float(train_loss) / len(train_data)

       
        training_losses.append(epoch_loss)

        print('Training Loss: ' ,epoch_loss, 'Accuracy: ' , epoch_acc)
        
        validation_loss=0.0
        correct=0
        total=0
        
        with torch.no_grad():
            for samples, instrument_family_target,instrument_source_target, targets in validation_loader:
                inputs, labels = samples.to(device), instrument_family_target.to(device)
                inputs=inputs.unsqueeze(1)
                inputs=inputs.float()
                # zero the parameter gradients
                optimizer.zero_grad()
                
                val_outputs = net(inputs)
                loss=criterion(val_outputs,labels)
                        
                _, preds = torch.max(val_outputs, 1)
                                
                total += inputs.size(0)
                validation_loss += loss.item() 
                correct += (preds == labels).sum().item()
        
        epoch_acc = float(correct)*100/ total
        epoch_loss = float(validation_loss) / len(validation_data)

        validation_losses.append(epoch_loss)

        print('Validation Loss: ' ,epoch_loss, 'Accuracy: ' , epoch_acc)
    
    print('Finished Training')
    plot( training_losses,validation_losses,EPOCHS)
  #weights saving
    
    weights = list(net.parameters())

    with open('Complexweight.txt', 'w') as f:
        for item in weights:
            f.write("%s\n" % item)
    pass

#######################################################################################
#confusion red matrix
def question51(y_target,y_predict,instruments):
    cm = confusion_matrix(y_target,y_predict)
    np.set_printoptions(precision=2)
    cm = cm / cm.astype(np.float).sum(axis=1)
    print("Normalized confusion matrix: \n", cm)
    
    fig = plt.figure(figsize=(10,7))
    #cmap=plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap='hot')
    plt.colorbar()
    tick_marks=np.arange(len(instruments))
    plt.xticks(tick_marks, instruments, rotation=45)
    plt.yticks(tick_marks, instruments)
    
    
    fmt = '.2f'
    thresh = cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],fmt),
                 horizontalalignment = "center",
                 color="white" if cm[i,j] > thresh else "black")
        
    plt.title("Confusion matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    fig.savefig("./Compleximages/confusion")
    pass

###########################################################################################

#correct class probability
def question3(cmax_tensor,cmin_tensor,instruments):
    for i in range(len(cmax_tensor)):
        if cmax_tensor[i] is None:
            continue
        question2(cmax_tensor[i].data.cpu().numpy(), "class_" +instruments[i] +
                       "_max_probability_waveform", "class_" +instruments[i]+
                       "_max_probability_waveform")
    for i in range(len(cmin_tensor)):
        if cmin_tensor[i] is None:
            continue
        question2(cmin_tensor[i].data.cpu().numpy(), "class_" +instruments[i] +
                       "_min_probability_waveform", "class_" +instruments[i]+
                       "_min_probability_waveform")
        
    pass

###########################################################################################
    
#decision boundary
def question4(sample_wave,instruments):
    sample_wave.sort(key = lambda tup:abs(tup[2]))
    waveform_plotted_status=[False]*len(instruments)
    for label,nearest_class,difference,waveform in sample_wave:
        if not waveform_plotted_status[label]:
            question2(waveform, "Near decision boundary sample for class" +instruments[label],
                          "Label class:" +instruments[label]+ "Nearby class:" 
                          +instruments[nearest_class]+"Difference: " +str(difference))
            waveform_plotted_status[label] = True
    
    
    pass 
###################################################################################################
#testing
   
def test(net,device):
    #test_loader=loader_function("./nsynth-test")
    data,test_loader=loader_function("/local/sandbox/nsynth/nsynth-test")
    
    correct = 0 
    total = 0 
    pred = []
    for ind1 in range(10):
        predicted = []
        for ind2 in range(10):
            predicted.append(ind2)
        pred.append(predicted)
    classes=set()
    
    with torch.no_grad():
       
        for samples, instrument_family_target,instrument_source_target, targets in test_loader:
            for label in instrument_family_target:
                classes.add(label.item())
            
            inputs, labels = samples.to(device), instrument_family_target.to(device)
            inputs=inputs.unsqueeze(1)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print('\nAccuracy of the network on the test audio: %d %%\n' %(
            100 * correct / total))
    
   
    instruments=['bass','brass','flute','guitar','keyboard','mallet','organ','reed','string','vocal']
    classes=list(classes)
    print(classes)
    class_correct = [0]*len(classes)
    class_total = [0]*len(classes)
    
    y_target=[]
    y_predict=[]
    
    classes_max=[0]*len(classes)
    classes_min=[1]*len(classes)
    
    cmax_tensor=[None]*len(classes)
    cmin_tensor=[None]*len(classes)
    
    sample_wave=[]
    
    with torch.no_grad():
        #4 tensors
        for samples, instrument_family_target,instrument_source_target, targets in test_loader:
            inputs, labels = samples.to(device), instrument_family_target.to(device)
            inputs=inputs.unsqueeze(1)
            outputs = net(inputs)
          
            _, predicted = torch.max(outputs, 1)
            
            c = (predicted == labels).squeeze()
            
            y_target.extend(labels)
            y_predict.extend(predicted)
            
            for i in range(len(labels)):
                
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                pred[label][predicted[i]]+=1
                #max-min class prob
                correctclassprob = outputs[i][label].item()
                if correctclassprob > classes_max[label.item()]:
                    classes_max[label.item()] = correctclassprob
                    cmax_tensor[label.item()] = inputs[i][0]
                    
                if correctclassprob < classes_min[label.item()]:
                    classes_min[label.item()] = correctclassprob
                    cmin_tensor[label.item()] = inputs[i][0]
                    
                    
                #graphs for prbs near decision boundaries
                
                
                probabilities = outputs[i].data.cpu().numpy()
                target_probability = probabilities[label.item()]
                threshold = 0.3
                for index in range(len(probabilities)):
                    difference = abs(target_probability - probabilities[index])
                    if index != label.item() and abs(difference) < threshold:
                        sample_wave.append((label.item(), index, difference,
                                                 inputs[i][0].data.cpu().numpy()))
                        break
    
    classes_accuracy=[]            
    for i in range(len(classes)):
        
        print('Accuracy of %5s : %2d %%' % (classes[i],100 * class_correct[i] / class_total[i]))
        classes_accuracy.append(100 * class_correct[i] / class_total[i])
        
 
    question5(y_target,y_predict,instruments)
    question3(cmax_tensor,cmin_tensor,instruments)
    question4(sample_wave,instruments)
    question51(y_target,y_predict,instruments)
    
    torch.save(net,'ComplexModel') 
    pass

#C:/Users/Mrudula/.spyder-py3            
#####################################################################################
    
def main():
    
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(device)
    
    net=Net()
    net.to(device)
    
    t0=time.time()
    train(net,device)
    test(net,device)
    t1=time.time()
    print("Time taken in minutes: ",(t1-t0)/60)
    pass
    
if __name__ == '__main__':
    main()


#C:\Users\Mrudula\.spyder-py3\Proj1

