# coding:utf-8
import os
import time
import ntpath
import datetime
import shutil
import matplotlib.pyplot as plt
import matplotlib
plt.set_loglevel("info") 
from optparse import OptionParser
from pathlib import Path
from subprocess import run, PIPE
from ida_scripts.utils import do_multiprocess,parse_fname,parse_other_options
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from collections import Counter
import networkx as nx
try:
    import cPickle as pickle
except:
    import pickle

import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)
coloredlogs.install(level=logging.DEBUG)
np.seterr(divide="ignore", invalid="ignore")
Mode = 'Presematic'

def load_func_data(bin_name):
    data_name = bin_name + ".pickle"
    with open(data_name, "rb") as f:
        func_data_list = pickle.load(f)
    return bin_name, func_data_list

def is_done(file_path):
    if Mode == 'Presematic':
        return os.path.exists(file_path + ".pickle")

def check_file(file_path,filter):
    os.chdir(file_path)
    all_file = os.listdir()
    files = []
    for f in all_file:
        if os.path.isdir(f):
            files.extend(check_file(file_path+'/'+f,filter))
            os.chdir(file_path)
        else:
            if judge_filter(f,filter):
                files.append(file_path+'/'+f)
    return files

def judge_filter(file_name,filter_dict):
    for key in filter_dict:
        if filter_dict[key][0] != 'all' and all(option not in file_name for option in filter_dict[key]):
            return False
    return True

def get_done_list(src_folder,filter,outDir):
    save_name = 'done_list.txt'
    os.makedirs(outDir, exist_ok=True)
    f_w = open(os.path.join(outDir,save_name),'w')
    files = check_file(src_folder,filter)
    for f in files:
        if Mode == 'Presematic':
            if f.split('.')[-1] == 'pickle':
                f_w.write(f.split('.pickle')[0] + '\n')

def clear_mid_file(src_folder):
    for root, dirs, files in os.walk(src_folder):
        for f in files:
            if f.split('.')[-1] in ["id0", "id1", "nam", "til", "id2", "cfg", "i64", "idb"]:
                try:
                    os.remove(os.path.join(root, f))
                except:
                    pass
                logger.info("[-] remove {} ...".format(f))

class IDAScript():
    def __init__(self,ida_path,log_path,script_path,out_folder) -> None:
        self.ida_path = ida_path
        self.log_path = log_path
        self.script_path = script_path
        self.out_folder = out_folder

    def main(self,file_folder,filter):
        all_file_list = check_file(file_folder,filter)
        file_list = []
        not_include = ['i64','idb','id0','id1','id2','nam','til','done','utils','pickle']
        for f in all_file_list:
            if os.path.basename(f).split('.')[-1] in not_include:
                pass
            else:
                file_list.append(f)
        
        logger.info("[+] start extracting {0} files ...".format(len(file_list)))
        t0 = time.time()
        res = do_multiprocess(self.run_helper, file_list, chunk_size=1, threshold=1) # 多线程
        logger.info("[-] done in: (%0.3fs)" % (time.time() - t0))
        
    def run_helper(self,file_path):
        if not os.path.exists(file_path):
            logger.debug("multiprocess: {} not exist".format(file_path))
            return file_path, None

        if is_done(os.path.join(self.out_folder,os.path.basename(file_path))):
            logger.debug("multiprocess: {} already done".format(file_path))
            return file_path, True

        if file_path.find("_32") != -1:
            ida = self.ida_path + '/ida.exe'
        else:
            ida = self.ida_path + "/ida64.exe"

        # Setup command line arguments
        path = [ida, "-A", "-S{} {}".format(self.script_path,self.out_folder)]
        path.append("-L{}".format(self.log_path))
        path.append(file_path)

        ret = run(path, env=os.environ.copy(), stdout=PIPE).returncode
        if ret != 0:
            logger.error("multiprocess: IDA returned {} for {}".format(ret, file_path))
            return file_path, False
        else:
            return file_path, True

def Presematic_features_extract(file_folder,out_folder,filter,ida_path,script_path,log_path):
    global Mode
    Mode = 'Presematic'
    idascript = IDAScript(
        ida_path,log_path,script_path,out_folder
    )
    idascript.main(file_folder,filter)

if __name__ == '__main__':
    op = OptionParser()
    op.add_option(
        "--src_folder",
        action="store",
        dest="src_folder",
        help="bin file folder",
        default='xxx'
    )
    op.add_option(
        "--out_folder",
        action="store",
        dest="out_folder",
        help="pickle file folder",
        default='xxx'
    )
    op.add_option(
        "--ida_path",
        action="store",
        dest="ida_path",
        help="ida tool path",
        default='xxx'
    )
    op.add_option(
        "--script_path",
        action="store",
        dest="script_path",
        help="ida script path",
        default='xxx'
    )
    op.add_option(
        "--log_folder",
        action="store",
        dest="log_folder",
        help="log result save folder",
        default='xxx'
    )
    op.add_option(
        "--run_type",
        action="store",
        dest="run_type",
        help="run task type",
        choices=['presematic'],
        default='presematic'
    )

    (opts, args) = op.parse_args()

    filter = {'bin_name':['all'],
              'version':['all'],
              'compiler':['all'],
              'arch':['all'],
              'opt':['all'],
              'others':['all']}

    os.makedirs(opts.log_folder,exist_ok=True)
    os.makedirs(opts.out_folder,exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    run_log_path = os.path.join(opts.log_folder,'run_{}.log'.format(timestamp))
    file_handler = logging.FileHandler(run_log_path)
    logger.addHandler(file_handler)
    log_path = os.path.join(opts.log_folder,'{}_{}.log'.format(opts.run_type,timestamp))
    if opts.run_type == 'presematic':
        logger.info('Start presematic features extract ...')
        Presematic_features_extract(opts.src_folder,opts.out_folder,filter,opts.ida_path,opts.script_path,log_path)
    
    clear_mid_file(opts.src_folder)
    get_done_list(opts.out_folder, filter, opts.log_folder)