import numpy as np
import scipy as sp
import scipy.signal as sig
import scipy.io.wavfile as wv
import sys
import pickle
import os
from optimizeDecoderCoeffSD import *
from HW4 import downmixFivePointOnetoStereo

def ambisonicsDecoder(useExistingCoeff,soundFileName,spkrAngles=[0.0,30.0,-30.0,110.0,-110.0],nIter=1000):

# Input Arguments:
# ---------------
#
# useExistingCoeff - If 1, coefficients computed from a previous run
#		     (stored in a file "ambisonicsDecoderCoeff.dat")
#		     are used for decoding the B-format file. Otherwise,
#		     new coefficients are computed
#
# soundFileName    - The file name for the set of Ambisonic B-format
#		     channels, stored in separate files. For example,
#	 	     if soundFileName is 'temp', then the function will
#		     read the files 'temp_W.wav', 'temp_X.wav', 
#		     'temp_Y.wav' and 'temp_Z.wav', containing the W,X,
#		     Y and Z channels respectively. (These files are
#		     located in the path /AmbisonicFiles/AmbisonicsB/)
#
# spkrAngles 	   - A tuple containing the set of angles for each of 
#		     the 5 full channels in the 5.1 layout, in the order
#		     of front center, fron left, front right, rear left,
#		     and rear right
#
# nIter		   - The number of iterations through which the 
#		     optimization proceeds

	d=1 #The directivity parameter d, not considered in this 
	    #optimization, so set to unity
	
	if useExistingCoeff!=1:
		
		#Computing a fresh set of decoder coefficients
		
		#The optimization is carried out (by a modified Steepest
		#Descent) in the function optimizeDecoderCoeffSD()
		
		#The decoder coefficients for the low frequency and high
		#frequency bands are returned, along with the fitness
		#values
		(paramListLF, paramListHF,LFFitness_opt,HFFitness_opt) = optimizeDecoderCoeffSD(spkrAngles,nIter)
		
		#The computed decoder coefficients are stored in a 
		#(pickled) dat file 'ambisonicsDecoderCoeff.dat',
		#loacted in /AmbisonicFiles
		filePath = os.getcwd()
		filePath = filePath + "/AmbisonicFiles/ambisonicsDecoderCoeff.dat"
		fileHandle = open(filePath,'wb')
		pickle.dump([paramListLF, paramListHF,LFFitness_opt,HFFitness_opt],fileHandle)
		fileHandle.close()
		
	else:
		
		#Using the decoder coefficients ccomputed in a previous
		#run of ambisonicsDecoder
		
		#Load the coefficients from the pickled dat file
		#ambisonicsDecoderCoeff.dat, located in
		#/AmbisonicsFiles
		filePath = os.getcwd()
		filePath = filePath + "AmbisonicFiles/ambisonicsDecoderCoeff.dat"
		fileHandle = open(filePath,'rb')
		paramListLF, paramListHF, LFFitness_opt, HFFitness_opt = pickle.load(fileHandle)
		print("Using parameters with LF Fitness " + str(LFFitness_opt) + " and HF Fitness " + str(HFFitness_opt))
		fileHandle.close()
	
	#Read B-format signals (4 separate files, one for each channel)
	WFile = os.getcwd() + "/AmbisonicFiles/AmbisonicsB/" + soundFileName + "_W.wav"
	XFile = os.getcwd() + "/AmbisonicFiles/AmbisonicsB/" + soundFileName + "_X.wav"
	YFile = os.getcwd() + "/AmbisonicFiles/AmbisonicsB/" + soundFileName + "_Y.wav"
	ZFile = os.getcwd() + "/AmbisonicFiles/AmbisonicsB/" + soundFileName + "_Z.wav"
	(fs,Wsig) = wv.read(WFile)
	print('Finished reading W.wav')
	(fs,Xsig) = wv.read(XFile)
	print('Finished reading X.wav')
	(fs,Ysig) = wv.read(YFile)
	print('Finished reading Y.wav')
	(fs,Zsig) = wv.read(ZFile)
	print('Finished reading Z.wav')
	
	#Create FIR filters to separate each channel into low (<700Hz)
	#and high (>700Hz) frequency bands
	LPF = sig.firwin(1023, 700, width=None, window='hamming', pass_zero=True, nyq=fs/2)
	print('Finished creating LPF')
	HPF = sig.firwin(1023, 700, width=None, window='hamming', pass_zero=False, nyq=fs/2)
	print('Finished creating HPF')
	
	#Split each channel (W,X,Y,Z) into low (WL,XL,YL,ZL) and high
	#(WH,XH,YH,ZH) frequency components
	WL = sp.signal.filtfilt(LPF,[1],Wsig)
	print('Finished LP Filtering W.wav')
	XL = sp.signal.filtfilt(LPF,[1],Xsig)
	print('Finished LP Filtering X.wav')
	YL = sp.signal.filtfilt(LPF,[1],Ysig)
	print('Finished LP Filtering Y.wav')
	ZL = sp.signal.filtfilt(LPF,[1],Zsig)
	print('Finished LP Filtering Z.wav')
	
	WH = sp.signal.filtfilt(HPF,[1],Wsig)
	print('Finished HP Filtering W.wav')
	XH = sp.signal.filtfilt(HPF,[1],Xsig)
	print('Finished HP Filtering X.wav')
	YH = sp.signal.filtfilt(HPF,[1],Ysig)
	print('Finished HP Filtering Y.wav')
	ZH = sp.signal.filtfilt(HPF,[1],Zsig)
	print('Finished HP Filtering Z.wav')
	
	#Apply decoding coefficients to each of the components, to
	#generate the outputs for each of the 5 (ITU 5.1) channels
	CF = 0.5 * ((2-d) * paramListLF[0] * WL + d * paramListLF[3] * XL) + 0.5 * ((2-d) * paramListHF[0] * WH + d * paramListHF[3] * XH)
	LF = 0.5 * ((2-d) * paramListLF[1] * WL + d * paramListLF[4] * XL + d * paramListLF[6] * YL) + 0.5 * ((2-d) * paramListHF[1] * WH + d * paramListHF[4] * XH +  d * paramListHF[6] * YH)
	RF = 0.5 * ((2-d) * paramListLF[1] * WL + d * paramListLF[4] * XL - d * paramListLF[6] * YL) + 0.5 * ((2-d) * paramListHF[1] * WH + d * paramListHF[4] * XH -  d * paramListHF[6] * YH)
	LB = 0.5 * ((2-d) * paramListLF[2] * WL + d * paramListLF[5] * XL + d * paramListLF[7] * YL) + 0.5 * ((2-d) * paramListHF[2] * WH + d * paramListHF[5] * XH +  d * paramListHF[7] * YH)
	RB = 0.5 * ((2-d) * paramListLF[2] * WL + d * paramListLF[5] * XL - d * paramListLF[7] * YL) + 0.5 * ((2-d) * paramListHF[2] * WH + d * paramListHF[5] * XH -  d * paramListHF[7] * YH)
	print("Applied LF and HF Decoding matrices to each channel")
	
	
	#The individual channels (front center, front left, front right,
	#rear left, rear right) are written to files (stored in the 
	#directory '/AmbisonicFiles/FiveChannels/')
	wv.write(os.getcwd() + "/AmbisonicFiles/FiveChannels/" + soundFileName + '_CF.wav',fs,CF.astype('float32'))
	wv.write(os.getcwd() + "/AmbisonicFiles/FiveChannels/" + soundFileName + '_LF.wav',fs,LF.astype('float32'))
	wv.write(os.getcwd() + "/AmbisonicFiles/FiveChannels/" + soundFileName + '_RF.wav',fs,RF.astype('float32'))
	wv.write(os.getcwd() + "/AmbisonicFiles/FiveChannels/" + soundFileName + '_LB.wav',fs,LB.astype('float32'))
	wv.write(os.getcwd() + "/AmbisonicFiles/FiveChannels/" + soundFileName + '_RB.wav',fs,RB.astype('float32'))
	print("Finished writing individual channels to file")
	
	#The 5.1 channel signal is then dowmixed into stereo, using 
	#a HRTF, as in HW 4
	downmixFivePointOnetoStereo(CF.astype('float32'),LF.astype('float32'),RF.astype('float32'),LB.astype('float32'),RB.astype('float32'),soundFileName,spkrAngles);
	print("Finished writing downmixed stereo output to file")
	
if __name__ == "__main__":

	#Take provided arguments (speaker angles in degrees in the order of CF, LF, RF, LB, RB) and pack into a numpy array
	useExistingCoeff = sys.argv[1] #1 for using old coeffs from a pickled dat, 0 to compute new coefficients
	if len(sys.argv)>2:
		spkrAngles = np.array([sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6]])
		nIter = int(sys.argv[7])
		soundFileName  = sys.argv[8]
		
	else:
		spkrAngles = np.array([0.0,30.0,-30.0,110.0,-110.0])
		nIter = 1000
		soundFileName = "spokenDirections"
		
	ambisonicsDecoder(int(useExistingCoeff),soundFileName,spkrAngles.astype(float),nIter)
