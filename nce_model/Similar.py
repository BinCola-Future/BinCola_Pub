# coding:utf-8
import numpy as np
from scipy.stats import pearsonr

class SimCal():
	def __init__(self,vul=np.array([1,1]),tar=np.array([1,1])) -> None:
		self.v_t_eculid = self.eculidDisSim(vul,tar)
		self.v_t_cos = self.cosSim(vul,tar)
		self.v_t_pearsonr = self.pearsonrSim(vul,tar)
		self.v_t_manhattan = self.manhattanDisSim(vul,tar)
		self.v_t_dict = {
			'EculidDisSim':self.v_t_eculid,
			'CosSim':self.v_t_cos,
			'PearsonrSim':self.v_t_pearsonr,
			'ManhattanDisSim':self.v_t_manhattan,
		}
		
	def eculidDisSim(self,x,y):
		return np.round(1.0/(1.0+np.sqrt(sum(pow(a-b,2) for a,b in zip(x,y)))),9)

	def cosSim(self,x,y):
		tmp=np.sum(x*y)
		non=np.linalg.norm(x)*np.linalg.norm(y)
		return np.round(tmp/float(non),9)

	def pearsonrSim(self,x,y):
		return np.round(pearsonr(x,y)[0],9)

	def manhattanDisSim(self,x,y):
		return np.round(1.0/(1.0+sum(abs(a-b) for a,b in zip(x,y))),9)

if __name__ == "__main__":
	a = np.array([1,2,3])
	b = np.array([1,2,3])
	sim = SimCal(a,b)
	print(a,b,sim.v_t_dict)
