import os
from tqdm import tqdm
import pandas as pd
import argparse
from sklearn.utils import shuffle

def judge_filter(file_name,filter_dict):
    for key in filter_dict:
        if filter_dict[key][0] != 'all' and all(option not in file_name for option in filter_dict[key]): #
            return False
    return True

def get_done_list(src_folder,outDir,filter):
    save_name = 'done_list_'
    for key in filter:
        save_name += '_'.join(filter[key]) + '_'
    save_name = save_name[:-1] + '.txt'
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    f_w = open(outDir + save_name,'w')
    for root, dirs, files in os.walk(src_folder):
        for f in files:
            if judge_filter(f,filter) and f.split('.')[-1] == 'pickle':
                print(os.path.join(root, f.split('.pickle')[0]))
                f_w.write(os.path.join(root, f.split('.pickle')[0]) + '\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='preprocess_BCSA')
    parser.add_argument('--src_folder', type=str, default='xxx', help='sentences folder')
    parser.add_argument('--out_folder', type=str, default='xxx', help='save folder')
    args = parser.parse_args()

    filter = {'bin_name':['all'],
              'version':['all'],
              'compiler':['all'],
              'arch':['all'],
              'opt':['all'],
              'others':['all']}
    not_include = ['lto','noinline','obfus_2loop','pie','sizeopt']
    get_done_list(args.src_folder, args.out_folder, filter)
