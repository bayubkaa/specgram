import json
import matplotlib.pyplot as plt
import os  
plt.rcParams["figure.figsize"] = (7,5)


f = open('metrics/resnet50.json')

data = json.load(f)
data['train_acc'].insert(0, 0)
data['val_acc'].insert(0, 0)
data['train_loss'].insert(0, 1)
data['val_loss'].insert(0, 1)


#------------------------------------ sing ngisor yes
data['train_acc'].append(99.8997)
data['val_acc'].append(99.25)
data['train_loss'].append(0.0038)
data['val_loss'].append(0.0450)


plt.plot(data['train_acc'])
plt.plot(data['val_acc'])

# plt.plot(data['train_loss'])
# plt.plot(data['val_loss'])

plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend(['train acc resnet50', 'val acc resnet50'], loc ="lower right")
plt.title('resnet50 acc')


# plt.xlim([-1, 50])
# plt.ylim([0, 1])

plt.savefig("gambar/resnet50_acc.png")
plt.grid()
plt.show()

# metriks_resnet50 = open('metrics/resnet50.json')
# metriks_resnet50 = json.load(metriks_resnet50)  
# train_loss_resnet50 = metriks_resnet50['train_loss']    
# train_acc_resnet50 = metriks_resnet50['train_acc']
# val_loss_resnet50 = metriks_resnet50['val_loss']    
# val_acc_resnet50 = metriks_resnet50['val_acc']


# metriks_cred = open('metrics/cred.json')
# metriks_cred = json.load(metriks_resnet50)  
# train_loss_cred = metriks_resnet50['train_loss']    
# train_acc_cred = metriks_resnet50['train_acc']
# val_loss_cred = metriks_resnet50['val_loss']    
# val_acc_resnet50 = metriks_resnet50['val_acc']



# plt.plot()
# plt.show()

  
# Closing file
# metriks_resnet50.close()