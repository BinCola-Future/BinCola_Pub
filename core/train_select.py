# coding:utf-8
# set positive num
import time
import random
import itertools
import os
import sys
import datetime
from matplotlib import get_backend
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
import heapq
sys.path.append('..')
from operator import itemgetter
from optparse import OptionParser
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from nce_utils import do_multiprocess,parse_fname,load_bfunc_data
from nce_model.datasets_select import CreateDataLoader
from nce_model.model_select import SiameseAttentionNet,InfoNCELoss,MSELoss
from nce_model.Similar import SimCal
from nce_model.Optim import ScheduledOptim
from nce_model.DrawPic import DrawROC,DrawRecall_Pre_F1,heapMapPlot
from torch import optim
import torch
import torch.nn.functional as F


import logging
import coloredlogs
logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)
coloredlogs.install(level=logging.DEBUG)
np.seterr(divide="ignore", invalid="ignore")


def get_package(func_key):
    return func_key[0]


def get_binary(func_key):
    return func_key[1]


def get_func(func_key):
    return func_key[2]


def get_opti(option_key):
    return option_key[0]


def get_arch(option_key):
    return option_key[1]


def get_arch_nobits(option_key):
    return option_key[1].split("_")[0]


def get_bits(option_key):
    return option_key[1].split("_")[1]


def get_compiler(option_key):
    return option_key[2]


def get_others(option_key):
    return option_key[3]


def parse_other_options(bin_path):
    other_options = ["lto", "pie", "noinline"]
    for opt in other_options:
        if opt in bin_path:
            return opt
    return "normal"

# opt:idx
def get_optionidx_map(options):
    return {opt: idx for idx, opt in enumerate(sorted(options))}

# idx:opt
def get_optionidx_map_re(options):
    return {idx: opt for idx, opt in enumerate(sorted(options))}


def is_valid(dictionary, s):
    return s in dictionary and dictionary[s]


def calc_ap(X, y):
    return average_precision_score(y, X)


def calc_roc(X, y):
    fpr, tpr, tresholds = roc_curve(y, X, pos_label=1)
    return auc(fpr, tpr)


def calc_tptn_gap(tps, tns):
    return np.mean(np.abs(tps - tns), axis=0)


def relative_difference(a, b):
    max_val = np.maximum(np.absolute(a), np.absolute(b))
    d = np.absolute(a - b) / max_val
    d[np.isnan(d)] = 0  # 0 / 0 = nan -> 0
    d[np.isinf(d)] = 1  # x / 0 = inf -> 1 (when x != 0)
    return d


def relative_distance(X, feature_indices):
    return 1 - (np.sum(X[:, feature_indices], axis=1)) / len(feature_indices)


def calc_metric_helper(func_key):
    global g_funcs, g_func_keys, g_dst_options
    func_data = g_funcs[func_key]
    option_candidates = list(func_data.keys())
    tp_results = []
    tn_results = []
    target_opts = []
    for src_opt, src_func in func_data.items():
        candidates = []
        for opt in func_data:
            if opt == src_opt:
                continue
            if src_opt not in g_dst_options:
                continue
            if opt not in g_dst_options[src_opt]:
                continue
            candidates.append(opt)
        if not candidates:
            continue
        dst_opt = random.choice(candidates)
        tp_func = func_data[dst_opt]

        while True:
            func_tn_key = random.choice(g_func_keys)
            if get_func(func_tn_key) != get_func(func_key):
                if dst_opt in g_funcs[func_tn_key]:
                    tn_func = g_funcs[func_tn_key][dst_opt]
                    break
        assert not np.isnan(src_func).any()
        assert not np.isnan(tp_func).any()
        assert not np.isnan(tn_func).any()
        tp_results.append(relative_difference(src_func, tp_func))
        tn_results.append(relative_difference(src_func, tn_func))
        target_opts.append((src_opt, dst_opt))
    if tp_results:
        tp_results = np.vstack(tp_results)
    if tn_results:
        tn_results = np.vstack(tn_results)
    return func_key, tp_results, tn_results, target_opts

def _init_calc(funcs, dst_options):
    global g_funcs, g_func_keys, g_dst_options
    g_funcs = funcs
    g_func_keys = sorted(funcs.keys())
    g_dst_options = dst_options


def calc_metric(funcs, dst_options):
    metric_results = do_multiprocess(
        calc_metric_helper,
        funcs.keys(),
        chunk_size=1,
        threshold=1,
        initializer=_init_calc,
        initargs=(funcs, dst_options),
    )
    func_keys, tp_results, tn_results, target_opts = zip(*metric_results)
    tp_results = np.vstack([x for x in tp_results if len(x)])
    tn_results = np.vstack([x for x in tn_results if len(x)])
    assert len(tp_results) == len(tn_results)
    return func_keys, tp_results, tn_results, target_opts

# tp_pairs:[[{func_src:feature},{func_dst:feature}]]
# tn_pairs:[[{func_src:feature},tn_dst_bins]]
def create_train_pairs(funcs, dst_options, optionidx_map, positive_num, negative_num):
    g_funcs = funcs
    g_func_keys = sorted(funcs.keys())
    g_dst_options = dst_options
    tp_pairs = []
    tn_pairs = []
    for func_key in funcs.keys():
        func_data = g_funcs[func_key]
        option_candidates = list(func_data.keys())
        for src_opt, src_func in func_data.items():
            candidates = []
            for opt in func_data:
                if opt == src_opt:
                    continue
                if src_opt not in g_dst_options:
                    continue
                if opt not in g_dst_options[src_opt]:
                    continue
                candidates.append(opt)
            if not candidates:
                continue

            if positive_num <= len(candidates):
                dst_opts = random.sample(candidates, positive_num)
            else:
                dst_opts = candidates
                while len(dst_opts) < positive_num:
                    dst_opts.append(random.choice(candidates))
            tp_funcs = []
            for dst_opt in dst_opts:
                tp_funcs.append(func_data[dst_opt])

            # select n tn function
            n = 0
            tn_funcs = []
            func_tn_keys = []
            while n < negative_num:
                while True:
                    func_tn_key = random.choice(g_func_keys)
                    if get_func(func_tn_key) != get_func(func_key):
                        if dst_opt in g_funcs[func_tn_key]:
                            tn_func = g_funcs[func_tn_key][dst_opt]
                            tn_funcs.append(tn_func)
                            func_tn_keys.append(func_tn_key)
                            break
                n += 1
                            
            assert not np.isnan(src_func).any()
            for tp_func in tp_funcs:
                assert not np.isnan(tp_func).any()
            for tn_func in tn_funcs:
                assert not np.isnan(tn_func).any()
            src_bin = "{}#{}".format('_'.join(list(func_key)),'_'.join(list(optionidx_map[src_opt])))
            tp_dst_bins = []
            for idx,dst_opt in enumerate(dst_opts):
                tp_dst_bin = "{}#{}".format('_'.join(list(func_key)), '_'.join(list(optionidx_map[dst_opt])))
                tp_dst_bins.append({tp_dst_bin:tp_funcs[idx]})

            tn_dst_bins = []
            for idx,func_tn_key in enumerate(func_tn_keys):
                tn_dst_bin = "{}#{}".format('_'.join(list(func_tn_key)), '_'.join(list(optionidx_map[dst_opt])))
                tn_dst_bins.append({tn_dst_bin:tn_funcs[idx]})

            tp_pairs.append([{src_bin:src_func},tp_dst_bins])
            tn_pairs.append([{src_bin:src_func},tn_dst_bins])
    return tp_pairs,tn_pairs

def load_model(opts, device):
    checkpoint = torch.load(opts.model_path, map_location=device)
    model_opt = checkpoint['settings']

    model = SiameseAttentionNet(model_opt.feature_dim,
                                model_opt.hidden_dim,
                                model_opt.n_layers,
                                model_opt.n_head,
                                model_opt.d_k,
                                model_opt.d_v,
                                model_opt.att_type,
                                model_opt.dropout,
                                model_opt.out_type).to(device)

    model.load_state_dict(checkpoint['model'])
    logger.info('[Info] Trained model state loaded.')
    return model

def calc_results(pred, label):
    return calc_roc(pred, label), calc_ap(pred, label)

# funcs:{(package, bin_name, func_name):{option_idx:[feature]}}
# feature_indices:[select_feature]
def calc_topK(src_all, dst_all, lable_all, k_list, func_num, sim_method):
    query = []
    data = []
    y_score = []
    choose_num = 0
    pos_true = []
    sim_obj = SimCal()

    if sim_method == 'EculidDisSim':
        sim_func = sim_obj.eculidDisSim
    elif sim_method == 'CosSim':
        sim_func = sim_obj.cosSim
    elif sim_method == 'PearsonrSim':
        sim_func = sim_obj.pearsonrSim
    elif sim_method == 'ManhattanDisSim':
        sim_func = sim_obj.manhattanDisSim
    for i,x in enumerate(lable_all):
        if x == 1.0:
            pos_true.append(i)
            if len(pos_true) >= func_num:
                break
    for i in pos_true:
        query.append(src_all[i])
        data.append(dst_all[i])

    num_list = list(np.zeros(len(k_list)))
    total = float(len(pos_true))
    score_list = []
    rank_list = []
    for i in tqdm(range(len(query))):
        q = query[i]
        pred_list = [sim_func(q,d) for d in data]
        pred_dict = {}
        for idx,item in enumerate(pred_list):
            pred_dict[idx] = item
        pred_dict = dict(sorted(pred_dict.items(), key=lambda d: d[1], reverse=True))
        for rank,pred in enumerate(list(pred_dict.keys())):
            if i == pred:
                rank_list.append(float(rank+1))
                break
        for idx, k in enumerate(k_list):
            pred = list(pred_dict.keys())[:k]
            if i in pred:
                num_list[idx] += 1
    for idx in range(len(k_list)):
        score_list.append(num_list[idx]/total)
    rank_re_array = np.array(list(map(lambda x : 1.0/x, rank_list)))
    MRR = np.sum(rank_re_array)/total
    return score_list,MRR


# preprocess possible target options for src option
def load_options(config):
    options = ["opti", "arch", "compiler", "others"]
    src_options = []
    dst_options = []
    fixed_options = []
    for idx, opt in enumerate(options):
        src_options.append(config["src_options"][opt])
        dst_options.append(config["dst_options"][opt])
        if is_valid(config, "fixed_options") and opt in config["fixed_options"]:
            fixed_options.append(idx)
    src_options = set(itertools.product(*src_options))
    dst_options = set(itertools.product(*dst_options))
    options = sorted(src_options.union(dst_options))
    optionidx_map = get_optionidx_map(options)

    dst_options_filtered = {}
    # Filtering dst options
    for src_option in src_options:

        def _check_option(opt):
            if opt == src_option:
                return False
            for idx in fixed_options:
                if opt[idx] != src_option[idx]:
                    return False
            return True

        candidates = list(filter(_check_option, dst_options))

        if "arch_bits" in config["fname"]:

            def _check_arch_without_bits(opt):
                return get_arch_nobits(opt) == get_arch_nobits(src_option)

            candidates = list(filter(_check_arch_without_bits, candidates))
        # need to have same bits
        elif "arch_endian" in config["fname"]:

            def _check_bits(opt):
                return get_bits(opt) == get_bits(src_option)

            candidates = list(filter(_check_bits, candidates))
        candidates = list(set([optionidx_map[opt] for opt in candidates]))
        dst_options_filtered[optionidx_map[src_option]] = candidates

    logger.info("total %d options.", len(options))
    logger.info("%d src options.", len(src_options))
    logger.info("%d dst options.", len(dst_options))
    logger.info("%d filtered dst options.", len(dst_options_filtered))
    return options, dst_options_filtered


def group_binaries(input_list):
    with open(input_list, "r") as f:
        bin_paths = f.read().splitlines()
    bins = {}
    packages = set()
    for bin_path in bin_paths:
        package, compiler, arch, opti, bin_name = parse_fname(bin_path)
        others = parse_other_options(bin_path)
        key = (package, bin_name)
        if key not in bins:
            bins[key] = []
        bins[key].append(bin_path)
        packages.add(package)
    logger.info(
        "%d packages, %d unique binaries, total %d binaries",
        len(packages),
        len(bins),
        len(bin_paths),
    )
    return bins, packages

def load_func_features_helper(bin_paths):
    global g_options, g_features
    global g_baseline
    func_features = {}
    num_features = len(g_features)
    optionidx_map = get_optionidx_map(g_options)
    for bin_path in bin_paths:
        package, compiler, arch, opti, bin_name = parse_fname(bin_path)
        others = parse_other_options(bin_path)
        _, bfunc_data_dict, bin_cg = load_bfunc_data(bin_path)
        for func_name in bfunc_data_dict.keys():
            bfunc_data = bfunc_data_dict[func_name]
            func_data = bfunc_data.info
            # Use only .text functions for testing
            if func_data["seg_name"] != ".text":
                continue
            if func_data["name"].startswith("sub_"):
                continue
            func_key = (package, bin_name, func_data["name"])
            option_key = (opti, arch, compiler, others)
            if option_key not in optionidx_map:
                continue
            option_idx = optionidx_map[option_key]
            if func_key not in func_features:
                func_features[func_key] = {}
            
            if g_baseline == 'Self':
                if option_idx not in func_features[func_key]:
                    func_features[func_key][option_idx] = np.zeros(
                        num_features, dtype=np.float64
                    )
                for feature_idx, feature in enumerate(g_features):
                    try:
                        if feature not in func_data["feature"]:
                            continue
                        val = func_data["feature"][feature]
                        func_features[func_key][option_idx][feature_idx] = val
                    except Exception as e:
                        logger.info('{}-{}'.format(bin_path,e))
            else:
                pass
    return func_features


# inevitably use globals since it is fast.
def _init_load(options, features, baseline):
    global g_options, g_features
    global g_baseline
    g_options = options
    g_features = features
    g_baseline = baseline


def load_func_features(input_list, options, features, baseline):
    grouped_bins, packages = group_binaries(input_list)
    func_features_list = do_multiprocess(
        load_func_features_helper,
        grouped_bins.values(),
        chunk_size=1,
        threshold=1,
        force=True,
        initializer=_init_load,
        initargs=(options, features, baseline),
    )
    funcs = {}
    for func_features in func_features_list:
        funcs.update(func_features)
    return funcs

def save_funcdatalist_csv(funcs,options,features,outdir):
    logger.info('start save func_data list ...')
    func_list = []
    opts_list = []
    features_list = []
    features_dict = {}
    for func in funcs.keys():
        for opts in funcs[func].keys():
            func_list.append(func)
            opts_list.append(options[opts])
            features_list.append('-'.join(map(str,funcs[func][opts])))
            for idx,feature in enumerate(features):
                if feature not in features_dict:
                    features_dict[feature] = [funcs[func][opts][idx]]
                else:
                    features_dict[feature].append(funcs[func][opts][idx])
    data_dict = {'func_name': func_list, 'options': opts_list}
    data_dict.update(features_dict)
    dataframe = pd.DataFrame(data_dict)
    dataframe.to_csv(os.path.join(outdir,"funcdatalist.csv"), index=False, sep=',')
    logger.info('save func_data list csv in {}'.format(os.path.join(outdir,"funcdatalist.csv")))

def save_origin_attn_feature_csv(src_options, src_origin, src_all, dst_options, dst_origin, dst_all, features, outdir):
    logger.info('start save origin_attn_feature ...')
    func_list = []
    origin_att = []
    features_dict = {}
    for i in range(1000):
        func_list.extend([src_options[i],dst_options[i],src_options[i],dst_options[i]])
        origin_att.extend(['origin', 'origin', 'att', 'att'])

        for idx, feature in enumerate(features):
            if feature not in features_dict:
                features_dict[feature] = [src_origin[i][idx]]
                features_dict[feature].append(dst_origin[i][idx])
                features_dict[feature].append(src_all[i][idx])
                features_dict[feature].append(dst_all[i][idx])
            else:
                features_dict[feature].append(src_origin[i][idx])
                features_dict[feature].append(dst_origin[i][idx])
                features_dict[feature].append(src_all[i][idx])
                features_dict[feature].append(dst_all[i][idx])

    data_dict = {'func_option': func_list, 'origin_att': origin_att}
    data_dict.update(features_dict)
    dataframe = pd.DataFrame(data_dict)
    dataframe.to_csv(os.path.join(outdir,"origin_attn_feature.csv"), index=False, sep=',')
    logger.info('save origin_attn_feature csv in {}'.format(os.path.join(outdir,"origin_attn_feature.csv")))


def save_result_csv(save_dict,outdir):
    os.makedirs(outdir, exist_ok=True)
    dataframe = pd.DataFrame(save_dict)
    dataframe.to_csv(os.path.join(outdir, "result.csv"), index=False, sep=',')
    logger.info('save result data csv')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def att_weight_analysis(att,features,outdir):
    fileObject = open(os.path.join(outdir,"features.txt"), 'w') 
    for fea in features:  
        fileObject.write(str(fea))  
        fileObject.write('\n') 
    fileObject.close()
    multi_att = att[0]
    for i in range(1,len(att)):
        multi_att = multi_att + att[i]
    multi_att = multi_att.cpu().detach().numpy()
    multi_att = np.mean(multi_att,axis=0)
    multi_att = np.sum(multi_att,axis=0)
    np.save(os.path.join(outdir,"multi_att.npy"), multi_att)
    heapMapPlot(multi_att,features,"multi_att.pdf",outdir,'YlGnBu')

    multi_att = np.sum(multi_att, axis=0)
    fea_ids = heapq.nlargest(10, range(len(multi_att)), multi_att.take)
    fileObject = open(os.path.join(outdir,"features_att.txt"), 'w')
    for i in range(len(fea_ids)):
        index = fea_ids[i]
        logger.info("att:{}-{}".format(multi_att[index],features[index]))
        fileObject.write("att:{}-{}".format(multi_att[index],features[index]))  
        fileObject.write('\n')
    fileObject.close()


def do_train(opts):
    binary_list = []
    options_list = []
    train_pairs_list = []
    valid_pairs_list = []
    test_pairs_list = []
    train_keys_list = []
    valid_keys_list = []
    test_keys_list = []
    ROC_list = []
    AP_list = []
    TOP1_list = []
    TOP5_list = []
    MRR_list = []
    Test_Poolsize_list = []
    poolsize_list =[32,1000]
    config_folder = opts.config_folder
    config_fname_list = [
        os.path.join(config_folder,"config_gnu_normal_opti_O0-O3_type.yml"),
    ]
    
    for config_fname in tqdm(config_fname_list):
        with open(config_fname, "r") as f:
            config = yaml.load(f,Loader=yaml.FullLoader)
        config["fname"] = config_fname

        # setup output directory
        if "outdir" in config and config["outdir"]:
            outdir = config["outdir"]
        else:
            base_name = os.path.splitext(os.path.basename(config_fname))[0]
        outdir = os.path.join(opts.log_out, opts.sim_method, opts.out_type, opts.feature_choose, base_name)
        out_curve = os.path.join(outdir, 'curve')
        out_attn = os.path.join(outdir, 'attn')
        date = datetime.datetime.now()
        outdir = os.path.join(outdir, str(date).replace(':','-').replace(' ','-'))

        model_save = os.path.join(opts.log_out, opts.sim_method, opts.out_type, opts.feature_choose, opts.model_save)
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(model_save, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(outdir, "log.txt"))
        logger.addHandler(file_handler)
        logger.info("config file name: %s", config["fname"])
        logger.info("output directory: %s", outdir)

        options, dst_options = load_options(config)
        features = sorted(config["features"])
        logger.info("%d features", len(features))
        optionidx_map = get_optionidx_map_re(options)
        t0 = time.time()
        logger.info("Feature loading ...")

        funcs = load_func_features(opts.input_list, options, features, opts.baseline)
        logger.info(
            "%d functions (%d unique).", sum([len(x) for x in funcs.values()]), len(funcs)
        )
        logger.info("Feature loading done. (%0.3fs)", time.time() - t0)

        k_list = [1,5]
        logger.info("[+] Model Parameter: ")
        logger.info("{}".format(opts))

        device = torch.device(opts.device if torch.cuda.is_available() else "cpu")
        optionidx_map = get_optionidx_map_re(options)

        logger.info("Split Datasets and Create Dataloader...")
        func_keys = sorted(funcs.keys())
        if opts.debug:
            set_seed(1226)
        random.shuffle(func_keys)
        train_num = int(len(func_keys) * opts.train_per)
        test_num = int(len(func_keys) * opts.valid_per)
        train_func_keys = func_keys[:train_num]
        valid_func_keys = func_keys[train_num:train_num + test_num]
        test_func_keys = func_keys[train_num + test_num:]
        
        logger.info(
            "Train: %d unique funcs, Valid: %d unique funcs , Test: %d unique funcs",
            len(train_func_keys),
            len(valid_func_keys),
            len(test_func_keys),
        )

        train_funcs = {key: funcs[key] for key in train_func_keys}
        valid_funcs = {key: funcs[key] for key in valid_func_keys}
        test_funcs = {key: funcs[key] for key in test_func_keys}

        
        train_tp_pairs, train_tn_pairs = create_train_pairs(train_funcs, dst_options, optionidx_map, opts.positive_num, opts.negative_num)
        train_data_loader = CreateDataLoader(train_tp_pairs, train_tn_pairs, opts.batch_size, device)
        
        valid_tp_pairs, valid_tn_pairs = create_train_pairs(valid_funcs, dst_options, optionidx_map, opts.positive_num, opts.negative_num)
        valid_data_loader = CreateDataLoader(valid_tp_pairs, valid_tn_pairs, opts.batch_size, device)

        test_tp_pairs, test_tn_pairs = create_train_pairs(test_funcs, dst_options, optionidx_map, opts.positive_num, opts.negative_num)
        test_data_loader = CreateDataLoader(test_tp_pairs, test_tn_pairs, opts.batch_size, device)

        logger.info("# of Train Pairs: %d", len(train_tp_pairs)*opts.positive_num + len(train_tn_pairs)*opts.negative_num)
        logger.info("# of Valid Pairs: %d", len(valid_tp_pairs)*opts.positive_num + len(valid_tn_pairs)*opts.negative_num)
        logger.info("# of Test Pairs: %d", len(test_tp_pairs)*opts.positive_num + len(test_tn_pairs)*opts.negative_num)
        
        model_name = 'funcs{}_{}_fea{}_hid{}_kv{}_head{}_layer{}_posnum{}_negnum{}_outtype{}_feature_{}_temper_{}.chkpt'.format(
            len(func_keys),
            config_fname.split('config_gnu_normal_')[1].split('_type.yml')[0],
            opts.feature_dim,
            opts.hidden_dim,
            opts.d_k,
            opts.n_head,
            opts.n_layers,
            opts.positive_num,
            opts.negative_num,
            opts.out_type,
            opts.feature_choose,
            opts.temper,
        )
        

        if opts.train:
            opts.model_path = os.path.join(model_save, model_name)
            if os.path.exists(opts.model_path):
                net = load_model(opts,device)
                logger.info("Load model {}...".format(opts.model_path))
            else:
                net = SiameseAttentionNet(opts.feature_dim,
                                        opts.hidden_dim,
                                        opts.n_layers,
                                        opts.n_head,
                                        opts.d_k,
                                        opts.d_v,
                                        opts.att_type,
                                        opts.dropout,
                                        opts.out_type).to(device)
            optimizer = ScheduledOptim(
                optim.Adam(net.parameters(), lr=0.0005, betas=(0.9, 0.98), eps=1e-09),
                opts.lr_mul, opts.feature_dim, opts.warmup_steps)
            logger.info("create train model ...")

            valid_loss = []

            logger.info("train...")
            # ===================== training ======================
            t0 = time.time()
            if opts.use_tb:
                logger.info("Use Tensorboard")
                from torch.utils.tensorboard import SummaryWriter
                tb_writer = SummaryWriter(log_dir=os.path.join(outdir, 'tensorboard'))
            stop_mark = 0
            for epoch in range(opts.epoch):
                pred_all = []
                lable_all = []
                loss_all = []
                for i,data in enumerate(train_data_loader):
                    src_option, src, pos, neg = data
                    optimizer.zero_grad()
                    src_out,pos_out,neg_out,similarity,slf_attn = net(src, pos, neg)
                    if opts.loss_type == "InfoNCE":
                        pos_num = pos_out.shape[1]
                        loss_contrastives = torch.zeros(pos_num)
                        for idx in range(pos_num):
                            loss_contrastives[idx] = InfoNCELoss(src_out, pos_out[:,idx,:], neg_out, opts.temper)
                        loss_contrastive = loss_contrastives.mean()
                    else:
                        loss_contrastive = MSELoss(src_out, pos_out, neg_out)
                    loss_contrastive.backward()
                    pred = similarity.cpu().detach().numpy()
                    optimizer.step_and_update_lr()
                    loss_all.append(loss_contrastive.cpu().detach().numpy())
                    pred_all.extend(pred)
                    lable_all.extend([1.0 for _ in range(int(pred.shape[0]/2))])
                    lable_all.extend([0.0 for _ in range(int(pred.shape[0]/2))])
                lr = optimizer._optimizer.param_groups[0]['lr']
                epoch_train_roc, epoch_train_ap = calc_results(pred_all, lable_all)
                epoch_train_loss = np.mean(loss_all)
                logger.info(" -Train- Epoch number:{} , AUC:{:.6f} , Loss:{:.6f} , Lr:{}".format(epoch, epoch_train_roc, epoch_train_loss,lr))

                # ===================== validing ======================
                pred_all = []
                lable_all = []
                loss_all = []
                for i, data in enumerate(valid_data_loader):
                    src_option, src, pos, neg = data
                    src_out,pos_out,neg_out,similarity,slf_attn = net(src, pos, neg)
                    if opts.loss_type == "InfoNCE":
                        pos_num = pos_out.shape[1]
                        loss_contrastives = torch.zeros(pos_num)
                        for idx in range(pos_num):
                            loss_contrastives[idx] = InfoNCELoss(src_out, pos_out[:,idx,:], neg_out, opts.temper)
                        loss_contrastive = loss_contrastives.mean()
                    else:
                        loss_contrastive = MSELoss(src_out, pos_out, neg_out)
                    pred = similarity.cpu().detach().numpy()
                    pred_all.extend(pred)
                    lable_all.extend([1.0 for _ in range(int(pred.shape[0]/2))])
                    lable_all.extend([0.0 for _ in range(int(pred.shape[0]/2))])
                    loss_all.append(loss_contrastive.cpu().detach().numpy())

                epoch_valid_roc, epoch_valid_ap = calc_results(pred_all, lable_all)
                epoch_valid_loss = np.mean(loss_all)
                logger.info(" -Valid- Epoch number:{} , AUC:{:.6f} , Loss:{:.6f} , Time:{:.3f}".format(epoch, epoch_valid_roc, epoch_valid_loss, time.time()-t0))


                valid_loss += [epoch_valid_loss]
                checkpoint = {'epoch': epoch, 'settings': opts, 'model': net.state_dict()}

                if epoch_valid_loss <= min(valid_loss):
                    torch.save(checkpoint, os.path.join(model_save, model_name))
                    logger.info('-The checkpoint file has been updated.')
                    stop_mark = 0
                else:
                    stop_mark += 1
                    if stop_mark > opts.stop_th:
                        logger.info('-The checkpoint file has not been updated for {} time.'.format(opts.stop_th))
                        break

                if opts.use_tb:
                    tb_writer.add_scalars('roc_auc', {'train': epoch_train_roc*100, 'val': epoch_valid_roc*100}, epoch)
                    tb_writer.add_scalars('avg_ap', {'train': epoch_train_ap*100, 'val': epoch_valid_ap*100}, epoch)
                    tb_writer.add_scalars('loss', {'train': epoch_train_loss, 'val': epoch_valid_loss}, epoch)
                    tb_writer.add_scalar('learning_rate', lr, epoch)

            train_time = time.time() - t0
            logger.info("train down. (%0.3fs)", train_time)


        # ===================== testing ======================
        t0 = time.time()
        logger.info("testing ...")
        pred_all = []
        lable_all = []
        src_options = []
        src_out = []
        pos_out = []
        neg_out = []
        src_origin = []
        pos_origin = []
        neg_origin = []
        if opts.test_type == 'with_dl':
            opts.model_path = os.path.join(model_save, model_name)
            net = load_model(opts,device)
            for i, data in enumerate(test_data_loader):
                src_option, src, pos, neg = data
                src_origin.extend(src.cpu().detach().numpy())
                pos_origin.extend(pos[:,0,:].cpu().detach().numpy())
                neg_origin.extend(neg[:,0,:].cpu().detach().numpy())
                src_options.extend(src_option)
                src,pos,neg,similarity,slf_attn = net(src, pos, neg)
                pred = similarity.cpu().detach().numpy()
                src_out_array = src.cpu().detach().numpy()
                pos_out_array = pos[:,0,:].cpu().detach().numpy()
                neg_out_array = neg[:,0,:].cpu().detach().numpy()
                src_out.extend(src_out_array)
                pos_out.extend(pos_out_array)
                neg_out.extend(neg_out_array)
                
                for j in range(len(src_out_array)):
                    simobj = SimCal(src_out_array[j],pos_out_array[j])
                    pred_all.append(simobj.v_t_dict[opts.sim_method])
                for j in range(len(src_out_array)):
                    simobj = SimCal(src_out_array[j],neg_out_array[j])
                    pred_all.append(simobj.v_t_dict[opts.sim_method])
                    
                lable_all.extend([1.0 for _ in range(int(pred.shape[0]/2))])
                lable_all.extend([0.0 for _ in range(int(pred.shape[0]/2))])
                if i == 0:
                    os.makedirs(out_attn, exist_ok=True)
                    att_weight_analysis(slf_attn,features,out_attn)
                    logger.info('save att_weight...')
        elif opts.test_type == 'no_dl':
            for i, data in enumerate(test_data_loader):
                src_option, src, pos, neg = data
                src_out_array = src.cpu().detach().numpy()
                pos_out_array = pos[:,0,:].cpu().detach().numpy()
                neg_out_array = neg[:,0,:].cpu().detach().numpy()
                src_out.extend(src_out_array)
                pos_out.extend(pos_out_array)
                neg_out.extend(neg_out_array)
                similarity = F.cosine_similarity(torch.cat((src,src),0), torch.cat((pos[:,0,:],neg[:,0,:]),0), dim=1, eps=1e-8)
                pred = similarity.cpu().detach().numpy()
                pred_all.extend(pred)
                lable_all.extend([1.0 for _ in range(int(pred.shape[0]/2))])
                lable_all.extend([0.0 for _ in range(int(pred.shape[0]/2))])

        for poolsize in poolsize_list:
            logger.info(" -Test- Poolsize:{}".format(poolsize))
            binary_list.append(opts.input_list.split('done_list_')[1].split('.txt')[0])
            options_list.append(config_fname.split('config_gnu_normal_')[1].split('_type.yml')[0])
            train_keys_list.append(len(train_func_keys))
            valid_keys_list.append(len(valid_func_keys))
            test_keys_list.append(len(test_func_keys))
            Test_Poolsize_list.append(poolsize)
            topk_list,MRR = calc_topK(src_out+src_out, pos_out+neg_out, lable_all, k_list, poolsize, opts.sim_method)
            test_roc, test_ap = calc_results(pred_all, lable_all)

            os.makedirs(os.path.join(outdir, 'curve'), exist_ok=True)
            DrawROC(lable_all,pred_all,os.path.join(outdir, 'curve'))
            DrawRecall_Pre_F1(lable_all,pred_all,os.path.join(outdir, 'curve'))

            os.makedirs(out_curve, exist_ok=True)
            DrawROC(lable_all, pred_all, out_curve)
            DrawRecall_Pre_F1(lable_all, pred_all, out_curve)


            logger.info(" -Test- AUC:{:.6f} , AP:{:.6f}".format(test_roc, test_ap))
            for idx, k in enumerate(k_list):
                logger.info(" -Test- top{}:{:.6f}".format(k,topk_list[idx]))
            logger.info(" -Test- MRR:{:.6f}".format(MRR))

            test_time = time.time() - t0
            logger.info("testing done. (%0.3fs)", test_time)
            logger.info("# of Train Pairs: %d", len(train_tp_pairs) + len(train_tn_pairs))
            logger.info("# of Valid Pairs: %d", len(valid_tp_pairs) + len(valid_tn_pairs))
            logger.info("# of Test Pairs: %d", len(test_tp_pairs) + len(test_tn_pairs))
            train_pairs_list.append(len(train_tp_pairs) + len(train_tn_pairs))
            valid_pairs_list.append(len(valid_tp_pairs) + len(valid_tn_pairs))
            test_pairs_list.append(len(test_tp_pairs) + len(test_tn_pairs))

            ROC_list.append(test_roc)
            AP_list.append(test_ap)
            TOP1_list.append(topk_list[0])
            TOP5_list.append(topk_list[1])
            MRR_list.append(MRR)
        logger.removeHandler(file_handler)
    result_dict = {'poolsize':Test_Poolsize_list,
                'binary':binary_list,
                'options':options_list,
                'ROC':ROC_list,
                'AP':AP_list,
                'TOP1':TOP1_list,
                'TOP5':TOP5_list,
                'MRR':MRR_list,
                'train_keys':train_keys_list,
                'valid_keys':valid_keys_list,
                'test_keys':test_keys_list,
                'train_pairs':train_pairs_list,
                'valid_pairs':valid_pairs_list,
                'test_pairs':test_pairs_list}
    date = datetime.datetime.now()
    savedir = os.path.join(opts.log_out, opts.sim_method, opts.out_type, opts.feature_choose,'result_csv',str(date).replace(':', '-').replace(' ', '-'))
    save_result_csv(result_dict,savedir)

if __name__ == "__main__":
    op = OptionParser()
    op.add_option(
        "--input_list",
        type="str",
        action="store",
        dest="input_list",
        help="a file containing a list of input binaries",
        default="xxx"
    )
    op.add_option(
        "--config_folder",
        type="str",
        action="store",
        dest="config_folder",
        help="config folder",
        default="xxx"
    )
    op.add_option("--batch_size",type=int,default=64)
    op.add_option("--feature_dim",type=int,default=256)
    op.add_option('--hidden_dim', type=int, default=512)
    op.add_option('--n_layers', type=int, default=6)
    op.add_option('--epoch', type=int, default=300)
    op.add_option('--n_head', type=int, default=4)
    op.add_option('--d_k', type=int, default=64)
    op.add_option('--d_v', type=int, default=64)
    op.add_option('--dropout', type=float, default=0.1)
    op.add_option('--warmup_steps', type=int, default=4000)
    op.add_option('--lr_mul', type=float, default=0.5)
    op.add_option('--num_folds', type=int, default=5)
    op.add_option('--train_ratio', type=float, default=0.8)
    op.add_option('--positive_num', type=int, default=1)
    op.add_option('--negative_num', type=int, default=1)
    op.add_option('--model_save', type=str, default='model_save')
    op.add_option('--log_out', type=str, default="xxx")
    op.add_option('--use_tb', action='store_true')
    op.add_option('--debug', action='store_true')
    op.add_option('--train', action='store_true')
    op.add_option('--train_per', type=float, default=0.8)
    op.add_option('--valid_per', type=float, default=0.1)
    op.add_option('--stop_th', type=int, default=5)
    op.add_option('--device', type=str, default="cuda:0")
    op.add_option('--temper', type=float, default=0.1)
    op.add_option('--test_type', choices=['with_dl','no_dl'], default='with_dl')
    op.add_option('--test_poolsize', type=int, default='32')
    op.add_option('--feature_choose', choices=['fusion','type'], default='fusion')
    op.add_option('--loss_type', choices=['InfoNCE','MSE'], default='InfoNCE')
    op.add_option('--sim_method', choices=[
                    'EculidDisSim',
			        'CosSim',
			        'PearsonrSim',
			        'ManhattanDisSim'], default='CosSim')
    op.add_option('--att_type', choices=[
                    'SelfAttention',
                    'NoAttention'], default='SelfAttention')
    op.add_option('--out_type', choices=[
                    'last',
                    'mean',
                    'sum'], default='mean')
    op.add_option('--baseline', choices=['Self'], default='Self')

    (opts, args) = op.parse_args()
    do_train(opts)