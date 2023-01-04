from tabnanny import check
from modules.resnet import ResNet50
from utils.pruning_utils import get_new_conv, get_new_bn, get_new_dependant_conv, get_model_size
from utils.pruning_utils import pick_filter_to_prune

from utils.pruning_utils import get_new_dependant_nn
from utils.pruning_utils import pick_neuron_to_remove

import torch
model = ResNet50(num_classes=2)
checkpoint = torch.load('checkpoint/resnet50_ckpt.pth')
model.load_state_dict(checkpoint["net"])

#print(f'old model:{get_model_size(model)}')

list_module = [[(model, 'conv1'), 
                (model, 'bn1'),
                (model.layer1[0], 'conv1'),
                (model.layer1[0], 'conv_id')],

                #layer 1 block 0
                [(model.layer1[0], 'conv1'),
                (model.layer1[0], 'bn1'),
                (model.layer1[0], 'conv2')],
                
                [(model.layer1[0], 'conv2'),
                (model.layer1[0], 'bn2'),
                (model.layer1[0], 'conv3')],

                [(model.layer1[0], 'conv3'),
                (model.layer1[0], 'bn3')],

                [(model.layer1[0], 'conv_id'),
                (model.layer1[0], 'bn_id')],

                #layer 1 block 1
                [(model.layer1[1], 'conv1'),
                (model.layer1[1], 'bn1'),
                (model.layer1[1], 'conv2')],

                [(model.layer1[1], 'conv2'),
                (model.layer1[1], 'bn2'),
                (model.layer1[1], 'conv3')],

                [(model.layer1[1], 'conv3'),
                (model.layer1[1], 'bn3')],

                #layer 1 block 2
                [(model.layer1[2], 'conv1'),
                (model.layer1[2], 'bn1'),
                (model.layer1[2], 'conv2')],

                [(model.layer1[2], 'conv2'),
                (model.layer1[2], 'bn2'),
                (model.layer1[2], 'conv3')],

                [(model.layer1[2], 'conv3'),
                (model.layer1[2], 'bn3')],
                #------------------------------------------------------
                #layer 2 block 0
                [(model.layer2[0], 'conv1'),
                (model.layer2[0], 'bn1'),
                (model.layer2[0], 'conv2')],
                
                [(model.layer2[0], 'conv2'),
                (model.layer2[0], 'bn2'),
                (model.layer2[0], 'conv3')],

                [(model.layer2[0], 'conv3'),
                (model.layer2[0], 'bn3')],

                [(model.layer2[0], 'conv_id'),
                (model.layer2[0], 'bn_id')],

                #layer 2 block 1
                [(model.layer2[1], 'conv1'),
                (model.layer2[1], 'bn1'),
                (model.layer2[1], 'conv2')],

                [(model.layer2[1], 'conv2'),
                (model.layer2[1], 'bn2'),
                (model.layer2[1], 'conv3')],

                [(model.layer2[1], 'conv3'),
                (model.layer2[1], 'bn3')],

                #layer 2 block 2
                [(model.layer2[2], 'conv1'),
                (model.layer2[2], 'bn1'),
                (model.layer2[2], 'conv2')],

                [(model.layer2[2], 'conv2'),
                (model.layer2[2], 'bn2'),
                (model.layer2[2], 'conv3')],

                [(model.layer2[2], 'conv3'),
                (model.layer2[2], 'bn3')],

                #layer 2 block 3
                [(model.layer2[3], 'conv1'),
                (model.layer2[3], 'bn1'),
                (model.layer2[3], 'conv2')],

                [(model.layer2[3], 'conv2'),
                (model.layer2[3], 'bn2'),
                (model.layer2[3], 'conv3')],

                [(model.layer2[3], 'conv3'),
                (model.layer2[3], 'bn3')],
                #------------------------------------------------------
                #layer 3 block 0
                [(model.layer3[0], 'conv1'),
                (model.layer3[0], 'bn1'),
                (model.layer3[0], 'conv2')],
                
                [(model.layer3[0], 'conv2'),
                (model.layer3[0], 'bn2'),
                (model.layer3[0], 'conv3')],

                [(model.layer3[0], 'conv3'),
                (model.layer3[0], 'bn3')],

                [(model.layer3[0], 'conv_id'),
                (model.layer3[0], 'bn_id')],

                #layer 3 block 1
                [(model.layer3[1], 'conv1'),
                (model.layer3[1], 'bn1'),
                (model.layer3[1], 'conv2')],

                [(model.layer3[1], 'conv2'),
                (model.layer3[1], 'bn2'),
                (model.layer3[1], 'conv3')],

                [(model.layer3[1], 'conv3'),
                (model.layer3[1], 'bn3')],

                #layer 3 block 2
                [(model.layer3[2], 'conv1'),
                (model.layer3[2], 'bn1'),
                (model.layer3[2], 'conv2')],

                [(model.layer3[2], 'conv2'),
                (model.layer3[2], 'bn2'),
                (model.layer3[2], 'conv3')],

                [(model.layer3[2], 'conv3'),
                (model.layer3[2], 'bn3')],

                #layer 3 block 3
                [(model.layer3[3], 'conv1'),
                (model.layer3[3], 'bn1'),
                (model.layer3[3], 'conv2')],

                [(model.layer3[3], 'conv2'),
                (model.layer3[3], 'bn2'),
                (model.layer3[3], 'conv3')],

                [(model.layer3[3], 'conv3'),
                (model.layer3[3], 'bn3')],

                #layer 3 block 4
                [(model.layer3[4], 'conv1'),
                (model.layer3[4], 'bn1'),
                (model.layer3[4], 'conv2')],

                [(model.layer3[4], 'conv2'),
                (model.layer3[4], 'bn2'),
                (model.layer3[4], 'conv3')],

                [(model.layer3[4], 'conv3'),
                (model.layer3[4], 'bn3')],

                #layer 3 block 5
                [(model.layer3[5], 'conv1'),
                (model.layer3[5], 'bn1'),
                (model.layer3[5], 'conv2')],

                [(model.layer3[5], 'conv2'),
                (model.layer3[5], 'bn2'),
                (model.layer3[5], 'conv3')],

                [(model.layer3[5], 'conv3'),
                (model.layer3[5], 'bn3')],

                #------------------------------------------------

                #layer 4 block 0
                [(model.layer4[0], 'conv1'),
                (model.layer4[0], 'bn1'),
                (model.layer4[0], 'conv2')],
                
                [(model.layer4[0], 'conv2'),
                (model.layer4[0], 'bn2'),
                (model.layer4[0], 'conv3')],

                [(model.layer4[0], 'conv3'),
                (model.layer4[0], 'bn3')],

                [(model.layer4[0], 'conv_id'),
                (model.layer4[0], 'bn_id')],

                #layer 4 block 1
                [(model.layer4[1], 'conv1'),
                (model.layer4[1], 'bn1'),
                (model.layer4[1], 'conv2')],

                [(model.layer4[1], 'conv2'),
                (model.layer4[1], 'bn2'),
                (model.layer4[1], 'conv3')],

                [(model.layer4[1], 'conv3'),
                (model.layer4[1], 'bn3')],

                #layer 4 block 2
                [(model.layer4[2], 'conv1'),
                (model.layer4[2], 'bn1'),
                (model.layer4[2], 'conv2')],

                [(model.layer4[2], 'conv2'),
                (model.layer4[2], 'bn2'),
                (model.layer4[2], 'conv3')],

                [(model.layer4[2], 'conv3'),
                (model.layer4[2], 'bn3')],

                #add from layer 1
                ['add',
                (model.layer1[1], 'conv1'),
                (model.layer1[0], 'conv_id')],

                ['add',
                (model.layer1[2], 'conv1'),
                (model.layer1[0], 'conv_id')],

                ['add',
                (model.layer2[0], 'conv1'),
                (model.layer1[0], 'conv_id')],

                ['add',
                (model.layer2[0], 'conv_id'),
                (model.layer1[0], 'conv_id')],

                #----add from layer2
                ['add',
                (model.layer2[1], 'conv1'),
                (model.layer2[0], 'conv_id')],

                ['add',
                (model.layer2[2], 'conv1'),
                (model.layer2[0], 'conv_id')],

                ['add',
                (model.layer2[3], 'conv1'),
                (model.layer2[0], 'conv_id')],

                ['add',
                (model.layer3[0], 'conv1'),
                (model.layer2[0], 'conv_id')],

                ['add',
                (model.layer3[0], 'conv_id'),
                (model.layer2[0], 'conv_id')],

                #----add from layer3
                ['add',
                (model.layer3[1], 'conv1'),
                (model.layer3[0], 'conv_id')],

                ['add',
                (model.layer3[2], 'conv1'),
                (model.layer3[0], 'conv_id')],

                ['add',
                (model.layer3[3], 'conv1'),
                (model.layer3[0], 'conv_id')],

                ['add',
                (model.layer3[4], 'conv1'),
                (model.layer3[0], 'conv_id')],

                ['add',
                (model.layer3[5], 'conv1'),
                (model.layer3[0], 'conv_id')],

                ['add',
                (model.layer4[0], 'conv1'),
                (model.layer3[0], 'conv_id')],

                ['add',
                (model.layer4[0], 'conv_id'),
                (model.layer3[0], 'conv_id')],

                #add from layer 4
                ['add',
                (model.layer4[1], 'conv1'),
                (model.layer4[0], 'conv_id')],

                ['add',
                (model.layer4[2], 'conv1'),
                (model.layer4[0], 'conv_id')],
                ]


def prune_model(norm_ord, ratio=0.5):
    maximum_pruned_ratio = ratio
    threshold = 1 - maximum_pruned_ratio

    min_filter_in_layer = 10

    if True:
        for i_, paired_modules in enumerate(list_module):
            if i_ == 0:
                pass

            if paired_modules[0] in ['concat', 'add']:
                new_input_size = 0
                prune_module_tup = paired_modules[1]
                conv = getattr(prune_module_tup[0], prune_module_tup[1])
                multiplier = 1
                for mod in paired_modules[2:]:
                    if isinstance(mod, int):
                        multiplier = mod
                    else:
                        conv_ = getattr(mod[0], mod[1])
                        new_input_size += conv_.out_channels
                new_input_size *= multiplier
                for _ in range(conv.in_channels - new_input_size):
                    conv = getattr(prune_module_tup[0], prune_module_tup[1])
                    filter_index = pick_filter_to_prune(conv, norm_ord=norm_ord, dim=(0,2))
                    new_conv = get_new_dependant_conv(conv, filter_index)
                    setattr(prune_module_tup[0], prune_module_tup[1], new_conv)
                
            else:
                prune_module_tup = paired_modules[0]
                dependant_modules_list = paired_modules[1:] 
                
                modules_prune, attribute_prune = prune_module_tup
                initial_conv = getattr(modules_prune, attribute_prune)
                
                initial_total_filter = initial_conv.out_channels
                while True:
                    conv = getattr(modules_prune, attribute_prune)
                    filter_index = pick_filter_to_prune(conv, norm_ord=norm_ord)
                    #print("filter index to be pruned: " + str(filter_index))
                    new_conv = get_new_conv(conv, filter_index)
                    setattr(modules_prune, attribute_prune, new_conv)

                    for paired_dep_modules in dependant_modules_list:
                        modules_dep, attribute_dep = paired_dep_modules
                        modules_dep_layer = getattr(modules_dep, attribute_dep)
                        if isinstance(modules_dep_layer, torch.nn.Conv2d):
                            modules_dep_layer = get_new_dependant_conv(modules_dep_layer, filter_index)
                        else:
                            modules_dep_layer = get_new_bn(modules_dep_layer, filter_index)
                        setattr(modules_dep, attribute_dep, modules_dep_layer)
                    
                    if conv.out_channels / initial_total_filter <= threshold or conv.out_channels <= min_filter_in_layer:
                        #print(initial_total_filter, conv.out_channels)
                        break
                # neuron_index = pick_neuron_to_remove(model.fc, norm_ord)
                # model.fc = get_new_dependant_nn(model.fc, neuron_index)
            for j in range(model.fc.in_features - model.layer4[0].conv_id.out_channels):
                neuron_index = pick_neuron_to_remove(model.fc, norm_ord)
                model.fc = get_new_dependant_nn(model.fc, neuron_index)
    


def get_pruned_resnet50(norm_ord, ratio):
    prune_model(norm_ord=norm_ord, ratio=ratio)
    print(f'pruned model:{get_model_size(model)}')
    return model


if __name__ == "__main__":
    model = get_pruned_resnet50(norm_ord=2, ratio=0.5)
    dummy_input = torch.randn((1,3,224,224), requires_grad=True)  
    model(dummy_input)
    
