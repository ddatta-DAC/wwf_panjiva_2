import numpy as np
import yaml

'''
The Goodall3 measure between 2 two categorical variables:
if x == y : 1 - p(x)^2
if x != y : 0  
Todo : memoize
Input : 
data_1 [ np.array ]
data_2 [ np.array ]
prob_list [ np.array ]    # p(x)
weight_list [ np.array ]  # weight of the dimensions
'''

CONFIG = None
CONFIG_FILE = 'config_v1.yaml'

def set_up():
    global CONFIG
    global _DIR
    global CONFIG_FILE

    config_file = 'config_v1.yaml'
    with open(config_file) as f:
        CONFIG = yaml.safe_load(f)


set_up()
def get_config():
    global CONFIG
    return CONFIG



def Goodall3(
        data_1,
        data_2,
        prob_list,
        weight_list
):
    p_w = 1 - np.power(prob_list, 2)
    xy = np.logical_not(data_1 - data_2) + 0
    xy = np.multiply(xy, p_w)
    xy = np.sum(np.multiply(xy, weight_list))
    return xy


'''
Important to maintain same order as the data for the relative ordering of domains
'''


def get_domain_wights():
    global CONFIG
    global _DIR
    # if nothing special : set all to 1
    _domains = CONFIG[_DIR]['domains']
    num_domains = len(_domains)
    _weights = np.ones(len(_domains)) / num_domains
    if CONFIG[_DIR]['domain_weight_multiplier'] is False:
        return _weights
    else:
        _dict = CONFIG[_DIR]['domain_weight_multiplier']
        _weights = []
        for _d in _domains:
            if _d in _dict.keys():
                _weights.append(_dict[_d] / num_domains)
            else:
                _weights.append(1 / num_domains)

        # --- #
        # Normalize array
        _weights = np.array(_weights) / np.sum(_weights)
        return _weights
