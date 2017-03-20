import numpy as np
import matplotlib.pyplot as plt
from computeFitnessFunctionsLF import *
from computeFitnessFunctionsHF import *


def optimizeDecoderCoeffSD(spkrAngles,nIter):

	spkrAngles = spkrAngles * (2 * np.pi / 360)
	#nIter = 1
	obsWindow = 50
	varThreshold = 0.5
	d=1
	paramListLF = np.array([1/np.sqrt(2),1/np.sqrt(2),1/np.sqrt(2),np.cos(spkrAngles[0]),np.cos(spkrAngles[1]),np.cos(spkrAngles[0]),np.sin(spkrAngles[3]),np.sin(spkrAngles[3]),1])
	paramListHF = np.array([1/np.sqrt(2),1/np.sqrt(2),1/np.sqrt(2),np.cos(spkrAngles[0]),np.cos(spkrAngles[1]),np.cos(spkrAngles[0]),np.sin(spkrAngles[3]),np.sin(spkrAngles[3]),1])
	#paramListLF = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0])
	#paramListHF = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0])
	stepSizeLF = 0.05
	stepSizeHF = 0.05
	tempMax = 1
	tempLF = tempMax
	tempHF = tempMax
	tempStep = float(tempMax)/float(nIter)
	print(tempStep)
	
	paramListLF_opt = np.array([0,0,0,0,0,0,0,0,0])
	paramListHF_opt = np.array([0,0,0,0,0,0,0,0,0])
	LFFitness_opt = 0
	HFFitness_opt = 0
	
	LFFitnessCurve = np.zeros(nIter)
	HFFitnessCurve = np.zeros(nIter)
	
	for i in range(0,nIter):
		
		gwLF = np.array([paramListLF[0],paramListLF[1],paramListLF[1],paramListLF[2],paramListLF[2]])
		gxLF = np.array([paramListLF[3],paramListLF[4],paramListLF[4],paramListLF[5],paramListLF[5]])
		gyLF = np.array([0,paramListLF[6],paramListLF[6],paramListLF[7],paramListLF[7]])
		
		gwHF = np.array([paramListHF[0],paramListHF[1],paramListHF[1],paramListHF[2],paramListHF[2]])
		gxHF = np.array([paramListHF[3],paramListHF[4],paramListHF[4],paramListHF[5],paramListHF[5]])
		gyHF = np.array([0,paramListHF[6],paramListHF[6],paramListHF[7],paramListHF[7]])
		
		LFFitness = computeFitnessFunctionsLF(gwLF,gxLF,gyLF,spkrAngles,paramListLF[8],d)
		HFFitness = computeFitnessFunctionsHF(gwHF,gxHF,gyHF,spkrAngles,paramListLF[8],d)
		
		if i==1:
			LFFitness_opt = LFFitness
			HFFitness_opt = HFFitness
		
		gradLF = np.array([0,0,0,0,0,0,0,0,0])
		gradHF = np.array([0,0,0,0,0,0,0,0,0])
		
		for k in range(0,9):
		
			paramList1 = np.copy(paramListLF)
			paramList1[k] = paramList1[k] + stepSizeLF
			
			gw1 = np.array([paramList1[0],paramList1[1],paramList1[1],paramList1[2],paramList1[2]])
			gx1 = np.array([paramList1[3],paramList1[4],paramList1[4],paramList1[5],paramList1[5]])
			gy1 = np.array([0,paramList1[6],paramList1[6],paramList1[7],paramList1[7]])
			LFFitness1 = computeFitnessFunctionsLF(gw1,gx1,gy1,spkrAngles,paramList1[8],d)
			
			paramList1 = np.copy(paramListHF)
			paramList1[k] = paramList1[k] + stepSizeHF
			
			gw1 = np.array([paramList1[0],paramList1[1],paramList1[1],paramList1[2],paramList1[2]])
			gx1 = np.array([paramList1[3],paramList1[4],paramList1[4],paramList1[5],paramList1[5]])
			gy1 = np.array([0,paramList1[6],paramList1[6],paramList1[7],paramList1[7]])
			HFFitness1 = computeFitnessFunctionsHF(gw1,gx1,gy1,spkrAngles,paramListLF[8],d)
			
			gradLF[k] = (-1 * (LFFitness1 - LFFitness)/stepSizeLF)
			gradHF[k] = (-1 * (HFFitness1 - HFFitness)/stepSizeHF)
			
		paramListLF = paramListLF + np.sign(gradLF) * stepSizeLF
		paramListHF = paramListHF + np.sign(gradHF) * stepSizeHF
		
		LFFitnessCurve[i] = LFFitness
		HFFitnessCurve[i] = HFFitness
		
		print("Iteration " + str(i))
		print("--------------")
		print("LF Fitness = " + str(LFFitness))
		print("HF Fitness = " + str(HFFitness))
		
		if LFFitness_opt>LFFitness:
			LFFitness_opt = LFFitness
			paramListLF_opt = paramListLF
			print("LF Parameters updated")
						
		if HFFitness_opt>HFFitness:
			HFFitness_opt = HFFitness
			paramListHF_opt = paramListHF
			print("HF Parameters updated")
			
		nUniqueLF = np.size(np.unique(LFFitnessCurve[(i-50+1):i]))
		nUniqueHF = np.size(np.unique(HFFitnessCurve[(i-50+1):i]))
			
		if i>=50 and ((np.var(LFFitnessCurve[(i-50+1):i])/np.nanmax(LFFitnessCurve[(i-50+1):i]))<varThreshold or nUniqueLF==2):
		
			print("LF Fitness Variance = " + str(np.var(LFFitnessCurve[(i-50+1):i])/np.nanmax(LFFitnessCurve[(i-50+1):i])))
			paramList1 = paramListLF + 2*(np.random.random(9)/2 - 1) * tempLF/tempMax * 0.25
			print("LF Temp = " + str(tempLF))
			gw1 = np.array([paramList1[0],paramList1[1],paramList1[1],paramList1[2],paramList1[2]])
			gx1 = np.array([paramList1[3],paramList1[4],paramList1[4],paramList1[5],paramList1[5]])
			gy1 = np.array([0,paramList1[6],paramList1[6],paramList1[7],paramList1[7]])
			LFFitness1 = computeFitnessFunctionsLF(gw1,gx1,gy1,spkrAngles,paramList1[8],d)
			
			if LFFitness1<=LFFitness:
				paramListLF = paramList1
				tempLF = tempLF - tempStep
			else:
				probNewState = np.exp(-1 * (LFFitness1 - LFFitness)/tempLF)#LFFitness1/LFFitness
				if np.random.random(1)<probNewState:
					paramListLF = paramList1
					tempLF = tempLF - tempStep
		
		if i>=50 and ((np.var(HFFitnessCurve[(i-50+1):i])/np.nanmax(HFFitnessCurve[(i-50+1):i]))<varThreshold or nUniqueHF==2):
		
			print("HF Fitness Variance = " + str(np.var(HFFitnessCurve[(i-50+1):i])/np.nanmax(HFFitnessCurve[(i-50+1):i])))
			paramList1 = paramListHF + 2*(np.random.random(9)/2 - 1) * tempHF/tempMax * 0.25
			print("HF Temp = " + str(tempHF))
			gw1 = np.array([paramList1[0],paramList1[1],paramList1[1],paramList1[2],paramList1[2]])
			gx1 = np.array([paramList1[3],paramList1[4],paramList1[4],paramList1[5],paramList1[5]])
			gy1 = np.array([0,paramList1[6],paramList1[6],paramList1[7],paramList1[7]])
			HFFitness1 = computeFitnessFunctionsHF(gw1,gx1,gy1,spkrAngles,paramList1[8],d)
			
			if HFFitness1<=HFFitness:
				paramListHF = paramList1
				tempHF = tempHF - tempStep
			else:
				probNewState = np.exp(-1 * (HFFitness1 - HFFitness)/tempHF)#HFFitness1/HFFitness
				if np.random.random(1)<probNewState:
					paramListHF = paramList1
					tempHF = tempHF - tempStep
				
			
		print(" ")
			
	print("Final LF Parameter List:")
	print(paramListLF)
	print(" ")
	print("Final HF Parameter List:")
	print(paramListHF)
	print(" ")
	print("Final LF Fitness = " + str(LFFitness_opt))
   	print(" ")
	print("Final HF Fitness = " + str(HFFitness_opt))
	plt.plot(np.arange(0,nIter),LFFitnessCurve,'r-',np.arange(0,nIter),HFFitnessCurve,'b-')
	plt.show()
	return([paramListLF_opt,paramListHF_opt,LFFitness_opt,HFFitness_opt])
	
if __name__ == "__main__":

	optimizeDecoderCoeffSD()
