import torch
from collections import OrderedDict
import loralib as lora


def lora_state_dict(state_dict):
    my_state_dict = state_dict
    return OrderedDict((k, my_state_dict[k]) for k in my_state_dict if 'lora_' in k)


def moe_layers_list(spk_list, lora_r):
    state_dict_list = {}
    for spk in spk_list:
        path = f"dir/spk{spk}/lora-r{lora_r}.cpu.pth"

        model = torch.load(path)
        state_dict = lora_state_dict(model)
        state_dict_list[spk] = state_dict

    layers_list = {}
    for spk in spk_list:
        for key, value in state_dict_list[spk].items():
            if key not in layers_list:
                layers_list[key] = {}
            layers_list[key][spk] = value
    
    return layers_list
