import cv2
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

''' FOR HELP WIITH THE OPTIMISATION CODE FOLLOW THE LINKS=
https://docs.scipy.org/doc/scipy-0.14.0/reference/tutorial/optimize.html#unconstrained-minimization-method-brent'''

# getting the initial estimate

def Rx(th):
	th = np.radians(th)
	c, s = np.cos(th) , np.sin(th)
	R = np.array(((0,0, 0, 0), (0, c, s, 0), (0, -s, c, 0), (0, 0 ,0, 1) ))
	return R

def Ry(th):
	th = np.radians(th)
	c, s = np.cos(th) , np.sin(th)
	R = np.array(((c,0, -s, 0), (0, 1, 0, 0), (s, 0, c, 0), (0, 0 ,0, 1) ))
	return R

def Rz(th):
	th = np.radians(th)
	c, s = np.cos(th) , np.sin(th)
	R = np.array(((c,s, 0, 0), (-s, c, 0, 0), (0, 0, 1, 0), (0, 0 ,0, 1) ))
	return R

def col(a):
	a = a.reshape(len(a),1)
	return a

#defining the empty list containing transformation from a particular ID to central coordinate system
#refer to the CAD model 
TransfMat = []


ID6toCentTranslation = np.array(((1, 0, 0, 0),(0, 1, 0, 19.92),(0, 0, 1,9.96),(0,0,0,1)))
ID6toCent = np.matmul(ID6toCentTranslation,Rx(64.43))

ID3toCentTranslation = np.array(((1, 0, 0, 0),(0, 1, 0, -19.92),(0, 0, 1,-9.96),(0,0,0,1)))
ID3toCent = np.matmul(ID3toCentTranslation,Rx(180-64.43))

TransfMat.append(np.identity(4))
 
for i in range (1,6):
	TransfMat.append((np.matmul(Rz(72*(3-i)),ID3toCent)))
	print i

for i in range (6,11):
	TransfMat.append((np.matmul(Rz(72*(i-6)),ID6toCent)))
	print i


ID11toCentTranslation = np.array(((1, 0, 0, 0),(0, 1, 0, -0),(0, 0, 1, 22.27),(0,0,0,1)))
ID11toCent = np.matmul(ID11toCentTranslation,Rz(-180))
TransfMat.append(ID11toCent)


ID12toCentTranslation = np.array(((1, 0, 0, 0),(0, 1, 0, 0),(0, 0, 1, -22.27),(0,0,0,1)))
ID12toCent = np.matmul(Ry(180),Rz(36))
ID12toCent = np.matmul(ID12toCentTranslation,ID12toCent)
TransfMat.append(ID12toCent)


print TransfMat[4]
print np.matmul(TransfMat[4],col(np.array([0,2,0,1])))


#Getting the data from the images



