# coding:utf-8
from traceback import print_exc
import idautils
import idc
import idaapi
import ida_pro
import ida_nalt
import ida_bytes
import pickle
import hashlib
import sys
import os
import csv
import json
import time
import binascii
import string
import numpy as np
from tqdm import tqdm
from hashlib import scrypt, sha1
from pyinstrument import Profiler
from capstone import *
from ida_scripts.utils import get_arch,store_func_data,parse_fname
from ida_scripts.fetch_funcdata import get_func_info,get_call_graph,get_bb_graph
from feature.asm import analyze_insts
from feature.asm_const import GRP_NO_MAP
from feature.feature_manager import FeatureManager
from bclass import *

out_folder = idc.ARGV[1]
printset = set(string.printable)
isprintable = lambda x: set(x).issubset(printset)

fm = FeatureManager()

# find strings
def get_strings(start_addr, end_addr):
    strings = []
    for h in idautils.Heads(start_addr, end_addr):
        refs = idautils.DataRefsFrom(h)
        for ref in refs:
            t = idc.get_str_type(ref)
            if isinstance(t, int) and t >= 0:
                s = idc.get_strlit_contents(ref)
                if s and isprintable(s):
                    strings.append([h, s, t, ref])
    return strings

# find consts
def get_consts(start_addr, end_addr):
    consts = []
    for h in idautils.Heads(start_addr, end_addr):
        insn = idautils.DecodeInstruction(h)
        if insn:
            for op in insn.ops:
                if op.type == idaapi.o_imm:
                    # get operand value
                    imm_value = op.value
                    # check if addres is loaded in idb
                    if not ida_bytes.is_loaded(imm_value):
                        consts.append(imm_value)
    return consts

def dump_function_details(  func_name,
                            arch,
                            func_ea,
                            bin_path,
                            bin_hash,
                            caller_map,
                            callee_map,
                            edge_map,
                            bb_callee_map,
                        ):
    func = idaapi.get_func(func_ea)
    bFunc = BFunc(func_ea,func.end_ea)
    bFunc.program = idc.GetInputFile()
    bFunc.funcname = func_name
    bFunc.arch = arch
    bFunc.info = get_func_info(func_ea,arch,bin_path,bin_hash,caller_map,callee_map,edge_map,bb_callee_map) # 非常耗时
    bFunc.info["feature"] = fm.get_all(bFunc.info)

    for bb in idaapi.FlowChart(idaapi.get_func(func_ea), flags=idaapi.FC_PREDS):
        if bb.start_ea == idaapi.BADADDR or bb.end_ea == idaapi.BADADDR:
            continue
        bBasicBlock = BBasicBlock(str(hex(bb.startEA)))
        bb_size = bb.end_ea - bb.start_ea
        block_data = idc.get_bytes(bb.start_ea, bb_size) or b""
        block_data_hash = hashlib.sha1(block_data).hexdigest()
        bb_strings = get_strings(bb.start_ea, bb.end_ea)
        bb_consts = get_consts(bb.start_ea, bb.end_ea)
        bBasicBlock.bb_data = {
            "size": bb_size,
            "block_id": bb.id,
            "startEA": bb.start_ea,
            "endEA": bb.end_ea,
            "type": bb.type,
            "is_ret": idaapi.is_ret_block(bb.type),
            "hash": block_data_hash,
            "strings": bb_strings,
            "consts": bb_consts,
        }

        preds = bb.preds()
        succs = bb.succs()
        
        if preds:
            preds_list = []
            for preds_block in preds:
                preds_list.append(str(hex(preds_block.startEA)))
            bBasicBlock.preds = preds_list
            
        if succs:
            succs_list = []
            for succs_block in succs:
                succs_list.append(str(hex(succs_block.startEA)))
            bBasicBlock.succs = succs_list
        
        bb_ins_features = {}
        res = analyze_insts(block_data, arch, start_addr=0)
        if res:
            group_data, consts = res
            group_cnt = {}
            for inst in group_data:
                for group_no in inst:
                    if group_no in group_cnt:
                        group_cnt[group_no] += 1
                    else:
                        group_cnt[group_no] = 1
            num_inst = len(group_data)
            bb_ins_features["inst_num_total"] = num_inst
            for no, cnt in group_cnt.items():
                cnt = float(cnt)
                name = GRP_NO_MAP[no]
                bb_ins_features["inst_num_{0}".format(name)] = cnt
        bBasicBlock.bb_ins_features = bb_ins_features

        for head in idautils.Heads(bb.startEA,bb.endEA):
            if idc.isCode(idaapi.getFlags(head)):
                bInstr = BInstr(head)
                next = idc.NextHead(head)
                length = 0
                if next < (bb.endEA + 1):
                    length = next - head
                else: 
                    length = bb.endEA - head
                bytes = idc.GetManyBytes(head, length, False)
                if bytes:
                    bytes=binascii.b2a_hex(bytes)
                disasm = idc.GetDisasm(head)
                bInstr.address = str(hex(head))
                bInstr.disasm = disasm
                bInstr.bytes = bytes
                mnem = idc.GetMnem(head)
                if mnem:
                    bInstr.mnem = mnem
                bBasicBlock.add_instr(bInstr)
        bBasicBlock.precess_bb(arch)
        bFunc.add_bb(bBasicBlock)
    return bFunc

def dump_functions(arch,bin_path,bin_hash):
    func_name_list = []
    bfunc_data_dict = {}
    caller_map, callee_map = get_call_graph()
    edge_map, bb_callee_map = get_bb_graph(caller_map, callee_map)
    profiler = Profiler()
    profiler.start()
    for idx, addr in enumerate(list(idautils.Functions())):
        function = idaapi.get_func(addr)
        if (not function) or (function.start_ea == idaapi.BADADDR) or (function.end_ea == idaapi.BADADDR):
            print("BAD Addr : {}".format(addr))
            continue
        func_name = idc.get_func_name(addr).strip()
        # Ignore Library Code
        flags = idc.get_func_attr(addr, idc.FUNCATTR_FLAGS)
        if flags & idc.FUNC_LIB:
            print("{} FUNC_LIB: {}".format(hex(addr),func_name))
            continue
        func_name_list.append(func_name)
        bfunc_data = dump_function_details(func_name,arch,addr,bin_path,bin_hash,caller_map,callee_map,edge_map,bb_callee_map)
        bfunc_data_dict.update({func_name:bfunc_data})
    save_data = {'bin_path':bin_path,
                 'bfunc_data_dict':bfunc_data_dict,
                }
    store_func_data(os.path.join(out_folder,os.path.basename(bin_path)), save_data)
    profiler.stop()
    profiler.print()


# Belows are utilitiy functions for IDAPython
def load_plugins():
    import idaapi
    plugins_dir = idaapi.idadir("plugins")
    files = [f for f in os.listdir(plugins_dir) if re.match(r".*\.py", f)]
    for path in files:
        idaapi.load_plugin(path)


def wait_auto_analysis():
    import ida_auto
    try:
        # >= IDA Pro 7.4
        ida_auto.auto_wait()
    except AttributeError:
        # < IDA Pro 7.4
        ida_auto.autoWait()

def init_idc():
    load_plugins()
    wait_auto_analysis()

if __name__ == '__main__':
    init_idc()
    try:
        bin_path = ida_nalt.get_input_file_path()
        bin_name = os.path.basename(bin_path)
        with open(bin_path, "rb") as f:
            bin_hash = sha1(f.read()).hexdigest()
        info = idaapi.get_inf_structure()
        if info.is_64bit():
            bits = 64
        elif info.is_32bit():
            bits = 32
        else:
            bits = 16
        endian = "little"
        if info.is_be():
            endian = "big"
        arch = "_".join([info.procName, str(bits), endian])
        arch = get_arch(arch)
        print("start analysis {}".format(bin_name))
        dump_functions(arch,bin_path,bin_hash)
        print("{} done".format(bin_name))
    except:
        import traceback
        traceback.print_exc()
        ida_pro.qexit(1)
    else:
        ida_pro.qexit(0)
