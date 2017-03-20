import numpy as np

def computeFitnessFunctionsLF(gw,gx,gy,spkrAngles,l,d):

	P0 = 0.5 * ((2-d) * gw / np.sqrt(2) + d * gx)
	P0 = np.sum(P0)
	VFit = 0
	MFit = 0
	AFit = 0
	
	gw1 = np.absolute(gw)
	gx1 = np.absolute(gx)
	gy1 = np.absolute(gy)
	
	for thetai in range(0,360,4):
	
		thetai1 = thetai * 2 * np.pi / 360;
	
		W = 1/np.sqrt(2)
		X = np.cos(thetai1)
		Y = np.sin(thetai1)
		
		W1 = 0.5 * (l + 1/l) * W + 1/np.sqrt(8) * (l - 1/l) * X;
		X1 = 0.5 * (l + 1/l) * X + 1/np.sqrt(2) * (l - 1/l) * W;
		Y1 = Y;
	
		Pi = 0.5 * ((2-d) * gw1 * W1 + d * (gx1 * X1 + gy1 * Y1))
		#Pi = 0.5 * ((2-d) * gw / np.sqrt(2) + d * (gx * np.cos(thetai1) + gy * np.sin(thetai1)))
		Pi = np.sum(Pi)
		#print([thetai, (1-P0/Pi)*(1-P0/Pi)])
		VFit = VFit + (1 - P0/Pi)*(1 - P0/Pi)/360
		
		Vx = Pi * np.cos(spkrAngles)
		Vx = np.sum(Vx)
		Vy = Pi * np.sin(spkrAngles)
		Vy = np.sum(Vy)
		Ri = np.sqrt(np.square(Vx) + np.square(Vy));
		MFit = MFit + (1 - Ri)*(1 - Ri)/360
		
		theta = np.arctan2(Vx,Vy)
		AFit = AFit + (thetai1 - theta)*(thetai1 - theta)/360
		
	VFit = np.sqrt(VFit)
	MFit = np.sqrt(MFit)
	AFit = np.sqrt(AFit)
	
	return VFit+MFit+AFit
