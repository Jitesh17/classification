
# In[10]:import os
import sys
import pyjeasy
import printj
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import matplotlib.pyplot as plt
from time import process_time 
import numpy as np
from data_preparation import get_dataset, convert_4ch_to_3ch, BoltDataset
# from my_models import TwoLayerNet
# from my_models import Net1
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
device = 'cuda'
IMG_SIZE = 256
BATCH_SIZE = 4
PATH = "/home/jitesh/3d/data/UE_training_results/bolt2/bolt_cropped"

classes = ['0', '1']
NUM_CLASSES =  len(classes)
# convert_4ch_to_3ch(PATH)  # Execute once only
dataset = get_dataset(PATH, ["b00", "b01"], IMG_SIZE)
trainloader = DataLoader(dataset["train"], batch_size = BATCH_SIZE, shuffle=True, num_workers=1)
dataiter = iter(trainloader)
images, labels = dataiter.next()
# images, labels = images.to(device), labels.to(device)
# print(trainloader)
print(labels)
# # print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


# In[10]:
device = 'cuda'
model = torchvision.models.densenet121(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 500),
    nn.Linear(500, NUM_CLASSES)
)
model = model.to(device)

from my_models import Net1
net = Net1(img_size=IMG_SIZE, num_classes=NUM_CLASSES)
net = net.to(device)
# print(net)
print(model)
# sys.exit()
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002, amsgrad=True)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500,1000,1500], gamma=0.5)

# len(trainloader)

# In[10]:
p_itr = 20 #200
loss_list = []
acc_list = []
t1_start = process_time()  
last_t1_stop = t1_start
total_epoch = 5
eta = np.inf
for epoch in range(total_epoch):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, (cpu_inputs, cpu_labels) in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        # inputs, labels = data
        inputs, labels = cpu_inputs.to(device), cpu_labels.to(device)

        # print(labels.shape)
        # print(labels)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        # outputs = net(inputs)
        outputs = model(inputs)
        # print(outputs.shape)
        # print(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # print statistics
        running_loss += loss.item()
        if i % p_itr == p_itr-1:    # print every 2000 mini-batches
            pred = torch.argmax(outputs, dim=1)
            correct = pred.eq(labels)
            acc = torch.mean(correct.float())
            loss_list.append(running_loss/p_itr)
            acc_list.append(acc)
            print('[%d, %5d] loss: %.3f,' %
                  (epoch + 1, i + 1, running_loss / p_itr), f'ETA: {eta} seconds')
            # print(f'ETA: {eta} seconds')
            running_loss = 0.0
        t1_stop = process_time()
        total_iters = total_epoch*len(trainloader)
        remaining_iters = (total_epoch-epoch + 1)*(len(trainloader) - i + 1)+0.01
        eta = (t1_stop-t1_start)/(total_iters-remaining_iters)*(remaining_iters)
        last_t1_stop = t1_stop
t1_stop = process_time()            
plt.plot(loss_list, label='loss')
plt.plot(acc_list, label='accuracy')
plt.legend()
plt.title('training loss and accuracy')
plt.show()
print(f'Finished Training\nTraining time: {t1_stop-t1_start} seconds.')

MODEL_WEIGHT_PATH = './bolt_Net1.pth'
torch.save(net.state_dict(), MODEL_WEIGHT_PATH)


# In[10]:
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
# In[10]:
# In[10]:
testloader = DataLoader(dataset["train"], batch_size=4,
                                         shuffle=True, num_workers=1)
dataiter = iter(testloader)
images, labels = dataiter.next()
# images, labels = images.to(device), labels.to(device)
print(testloader)
print(labels)
# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# In[10]:
labels
# # %%

# %%dataset = get_dataset(PATH, ["b00", "b01"], IMG_SIZE)
# model = model()
# MODEL_WEIGHT_PATH = './bolt_Net1.pth'
# model.load_state_dict(torch.load(MODEL_WEIGHT_PATH))
dataset = get_dataset(PATH, ["b00", "b11"], IMG_SIZE)
testloader = DataLoader(dataset["train"], batch_size=4,
                                         shuffle=False, num_workers=1)
for i, (cpu_inputs, cpu_labels) in enumerate(testloader
                                             , 0):
    inputs, labels = cpu_inputs.to(device), cpu_labels.to(device)
    # outputs = net(inputs)
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    imshow(torchvision.utils.make_grid(cpu_inputs))
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))
    print('GroundTruth: ', ' '.join('%5s' % classes[cpu_labels[j]] for j in range(4)))
# %%
model.eval()
fn_list = []
pred_list = []
for x, fn in testloader:
    with torch.no_grad():
        x = x.to(device)
        output = model(x)
        pred = torch.argmax(output, dim=1)
        fn_list += [n[:-4] for n in fn]
        pred_list += [p.item() for p in pred]

submission = pd.DataFrame({"id":fn_list, "label":pred_list})
submission.to_csv(f'preds_densenet121_{IMG_SIZE}.csv', 
                #   index=False
                  )

# %%
