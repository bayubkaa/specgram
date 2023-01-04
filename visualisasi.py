import json
import matplotlib.pyplot as plt
import os  
plt.rcParams["figure.figsize"] = (15,7)
analisa_metrik = 'val_acc'


metrics_folder = 'metrics'

list_file_json = os.listdir(metrics_folder)
list_file_json.sort()
legends = []
for filename in list_file_json:
    if filename in ['mobilenetv2.json']:
        continue
    # print(filename[:-5])
    full_path = os.path.join(metrics_folder, filename)
    f = open(full_path)
    data = json.load(f)

    for key in data:
        if key == analisa_metrik:
            legends.append(filename[:-5])
            plt.plot(data[key])


plt.xlabel('epoch')
plt.ylabel(analisa_metrik)
plt.legend(legends, loc ="lower right")
plt.savefig('gambar/'+analisa_metrik+".png")
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