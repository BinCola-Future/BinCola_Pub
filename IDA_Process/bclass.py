# coding:utf-8
import hashlib
import pickle
import re
import networkx as nx

class BFunc:
	def __init__(self, start_addr, end_addr):
		self.program = None					
		self.funcname = None				
		self.start_addr = start_addr 		
		self.end_addr = end_addr 			
		self.bbs = [] 											
		self.info = {}						
		self.hash = None					
		self.flag = None
		self.arch = None
	
	def add_bb(self, bb):    				
		self.bbs.append(bb)
	
	def print_func_bb(self):				
		for item in self.bbs:
			item.print_bb()

	def print_func_base(self):				
		print("program name: " + self.program)
		print("function name: " + self.funcname)
		print("arch: " + self.arch)
		print("info: ")
		for key in self.info.keys():
			print(key," : ",self.info[key])
		print("feature: ")
		for key in self.feature.keys():
			print(key," : ",self.feature[key])

class BBasicBlock:
	CMP_REPS = ["short loc_", "loc_", "j_nullsub_", "nullsub_", "j_sub_", "sub_",
				"qword_", "dword_", "byte_", "word_", "off_", "def_", "unk_", "asc_",
				"stru_", "dbl_", "locret_"]
	CMP_REMS = ["dword ptr ", "byte ptr ", "word ptr ", "qword ptr ", "short ptr "]
	CMP_REGS = {
		'x86_32':["rax", "eax", "ax", "al", 
				"rbx", "ebx", "bx", "bl", 
				"rcx", "ecx", "cx", "cl", 
				"rdx", "edx", "dx", "dl", 
				"rsi", "esi", "si", "sil", 
				"rdi", "edi", "di", "dil", 
				"rbp", "ebp", "bp", "bpl", 
				"rsp", "esp", "sp", "spl", 
				"r8d", "r8w", "r8b", "r8",
				"r9d", "r9w", "r9b", "r9",
				"r10d", "r10w", "r10b", "r10", 
				"r11d", "r11w", "r11b", "r11", 
				"r12d", "r12w", "r12b", "r12", 
				"r13d", "r13w", "r13b", "r13", 
				"r14d", "r14w", "r14b", "r14", 
				"r15d", "r15w", "r15b", "r15"],
		'x86_64':["rax", "eax", "ax", "al", 
				"rbx", "ebx", "bx", "bl", 
				"rcx", "ecx", "cx", "cl", 
				"rdx", "edx", "dx", "dl", 
				"rsi", "esi", "si", "sil", 
				"rdi", "edi", "di", "dil", 
				"rbp", "ebp", "bp", "bpl", 
				"rsp", "esp", "sp", "spl", 
				"r8d", "r8w", "r8b", "r8",
				"r9d", "r9w", "r9b", "r9",
				"r10d", "r10w", "r10b", "r10", 
				"r11d", "r11w", "r11b", "r11", 
				"r12d", "r12w", "r12b", "r12", 
				"r13d", "r13w", "r13b", "r13", 
				"r14d", "r14w", "r14b", "r14", 
				"r15d", "r15w", "r15b", "r15"],
		'arm_32':["R0","R1","R2","R3","R4","R5","R6","R7","R8","R9","R10","R11","R12","R13","R14","R15"],
		'arm_64':["X0","X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15",
				"X16","X17","X18","X19","X20","X21","X22","X23","X24","X25","X26","X27","X28","X29","X30",
				"W0","W1","W2","W3","W4","W5","W6","W7","W8","W9","W10","W11","W12","W13","W14","W15",
				"W16","W17","W18","W19","W20","W21","W22","W23","W24","W25","W26","W27","W28","W29","W30"],
		'mips_32':["$zero", "$at", "$v0", "$v1", "$a0", "$a1", "$a2", "$a3", "$t0", "$t1", "$t2",
				"$t3", "$t4", "$t5", "$t6", "$t7", "$s0", "$s1", "$s2", "$s3", "$s4", "$s5", "$s6", "$s7",
				"$t8", "$t9", "$k0", "$k1", "$gp", "$sp", "$fp", "$ra"],
		'mips_64':["$zero", "$at", "$v0", "$v1", "$a0", "$a1", "$a2", "$a3", "$t0", "$t1", "$t2",
				"$t3", "$t4", "$t5", "$t6", "$t7", "$s0", "$s1", "$s2", "$s3", "$s4", "$s5", "$s6", "$s7",
				"$t8", "$t9", "$k0", "$k1", "$gp", "$sp", "$fp", "$ra"],
		'mipseb_32':["$zero", "$at", "$v0", "$v1", "$a0", "$a1", "$a2", "$a3", "$t0", "$t1", "$t2",
				"$t3", "$t4", "$t5", "$t6", "$t7", "$s0", "$s1", "$s2", "$s3", "$s4", "$s5", "$s6", "$s7",
				"$t8", "$t9", "$k0", "$k1", "$gp", "$sp", "$fp", "$ra"],
		'mipseb_64':["$zero", "$at", "$v0", "$v1", "$a0", "$a1", "$a2", "$a3", "$t0", "$t1", "$t2",
				"$t3", "$t4", "$t5", "$t6", "$t7", "$s0", "$s1", "$s2", "$s3", "$s4", "$s5", "$s6", "$s7",
				"$t8", "$t9", "$k0", "$k1", "$gp", "$sp", "$fp", "$ra"]
	}
	
	def __init__(self, start_addr):
		self.function = None
		self.start_addr = start_addr
		self.end_address = None
		self.binstrs = []
		self.preds = []
		self.succs = []
		self.hash_v1 = None					
		self.hash_v2 = None 				
		self.hash_v3 = None
		self.bb_data = {}					
		self.bb_ins_features = {}			
		
		self.neighbour_disasm_list = None
		self.mnen_list = None
		self.disasm_list = None				
		self.flag = None
		self.flag_v2 = None
		
	def add_instr(self, binstr):
		self.binstrs.append(binstr)
	def add_preds(self, pred):
		self.preds.append(pred)
	def add_succs(self, succ):
		self.succs.append(succ)
	
	def normalize_instruction(self, instr_str, arch):
		instr_str = instr_str.split(";")[0]

		for sub in self.CMP_REMS:
			if sub in instr_str:
				instr_str = instr_str.replace(sub, "")
		
		for r in self.CMP_REGS[arch]:
			if r in instr_str and 'call' not in instr_str:
				instr_str = re.sub(r + "[0-9]*", 'reg', instr_str)

		instr_str = re.sub('\[.*\]',  'mem', instr_str)
		instr_str = re.sub('=\(.*\)',  'mem', instr_str)
		
		instr_str = re.sub('0x[0-9a-fA-F]{5}',  'address', instr_str)
		
		instr_str = re.sub("[ \t\n]+$", "", instr_str)
		return instr_str
	
	def set_mnen_list(self):
		mnen_list = []
		for binstr in self.binstrs:
			mnen_list.append(binstr.mnem)
		self.mnen_list = mnen_list

	def set_hash_v1(self):
		m = hashlib.md5()
		mnem_str = ' '.join(self.mnen_list)
		m.update(mnem_str.encode('utf-8'))
		md5 = str(m.hexdigest())
		self.hash_v1 = md5
	
	def set_disasm_list(self,arch):
		disasm_list = []
		for instr in self.binstrs:
			d = re.sub('\t+',' ' ,instr.disasm)
			d = re.sub(' +',' ' ,d)
			n_disasm = self.normalize_instruction(d,arch)
			disasm_list.append(n_disasm)
		self.disasm_list = disasm_list

	def set_hash_v2(self):
		m = hashlib.md5()
		instr_str = ' '.join(self.disasm_list) 
		m.update(instr_str.encode('utf-8'))
		md5 = str(m.hexdigest())
		self.hash_v2 = md5
		
	def precess_bb(self,arch):
		self.set_mnen_list()
		self.set_hash_v1()
		self.set_disasm_list(arch)
		self.set_hash_v2()
		
	def print_bb(self):
		print ("BB name : " + str(self.start_addr))
		print ("BB base_fea : ",self.bb_data)
		print ("BB bb_ins_features :")
		for k in self.bb_ins_features.keys():
			print('{:<30}:{:<0}'.format(k,self.bb_ins_features[k]))
		print ("BB preds : ")
		print (self.preds)
		print ("BB succs :")
		print (self.succs)

		print ("BB disam_list :")
		for item in self.disasm_list:
			print ('{:<38} | {:<0}'.format('',item))
		
		print ("BB instrs: ")
		for item in self.binstrs:
			item.print_instr()
	
class BInstr:
	def __init__(self, start_addr):
		self.basicblock = None
		self.start_addr = start_addr
		self.disasm = None					
		self.mnem = None
		self.bytes = None
		self.flag = None
		self.hash = None

	def print_instr(self):
		instr = '{:<10} | {:<25} | {:<0}'.format(self.address, str(self.bytes), self.disasm)
		print (instr)