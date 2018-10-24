#Used this code to confirm that the tvec and rvec given by the 
 # estimatePoseSingleMarkers is of the marker frame wrt the camera frame


import numpy as np
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
from mpl_toolkits.mplot3d import Axes3D
import transforms3d as tf3d
import time

def RodriguesToTransf(x):
    #input (6,)
    x = np.array(x)
    # print x[0:3]
    
    rot,_ = cv2.Rodrigues(x[0:3])
    
    
    trans =  np.reshape(x[3:6],(3,1))

    Trransf = np.concatenate((rot,trans),axis = 1)
    Trransf = np.concatenate((Trransf,np.array([[0,0,0,1]])),axis = 0)

    return Trransf



### Switches: 
sub_pix_refinement_switch =1
detect_tip_switch = 0



iterations_for_while = 100
marker_size_in_mm = 19.16
tip_coord  = np.array([     2.89509534, -111.83311787  , -2.33105497,1]) 



with np.load('B4Aruco.npz') as X:
    # mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
    mtx, dist = [X[i] for i in ('mtx','dist')] 


cap = cv2.VideoCapture(0)



xs,ys,zs,color = [0]*iterations_for_while,[0]*iterations_for_while,[0]*iterations_for_while,[0]*iterations_for_while
rxs,rys,rzs = [0]*iterations_for_while,[0]*iterations_for_while,[0]*iterations_for_while

pose = np.zeros((iterations_for_while,6))

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters =  aruco.DetectorParameters_create()
parameters.cornerRefinementMethod = sub_pix_refinement_switch
parameters.cornerRefinementMinAccuracy = 0.00005


emptyList = list()
emptyList.append(np.zeros((4,4)))


j = 0

t0 = time.time() 
time_vect = [0]*iterations_for_while

while(j<iterations_for_while):
    # Capture frame-by-frame
    ret, frame = cap.read()


 
    #print(parameters)
 
    '''    detectMarkers(...)
        detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedI
        mgPoints]]]]) -> corners, ids, rejectedImgPoints
        '''
        #lists of ids and the corners beloning to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)


    if ids is not  None :
        for i in range(0,len(ids)):
            if ids[i] ==7:  
                emptyList[0] = corners[i]
                # print corners[i]
                frame = aruco.drawDetectedMarkers(frame, emptyList)
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers( emptyList, marker_size_in_mm, mtx,dist)
                transf_mat_for_frame = RodriguesToTransf(np.append(rvecs,tvecs))
                tip_loc_cam_frame = transf_mat_for_frame.dot(tip_coord.T)
    # print tvecs[0,0,0]
    # print 
        # print tip_loc_cam_frame
        xs[j] = tvecs[0,0,0]
        ys[j] = tvecs[0,0,1]
        zs[j] = tvecs[0,0,2]
        rxs[j] = rvecs[0,0,0]*180/np.pi
        rys[j] = rvecs[0,0,1]*180/np.pi
        rzs[j] = rvecs[0,0,2]*180/np.pi
        pose[j,:] = np.append(tvecs,rvecs)
        print rxs[j]
        print rys[j]
        print rzs[j]
        print ""
    
        if detect_tip_switch ==1:
            xs[j] = tip_loc_cam_frame[0]
            ys[j] = tip_loc_cam_frame[1]
            zs[j] = tip_loc_cam_frame[2]
            
        else:        
            xs[j] = tvecs[0,0,0]
            ys[j] = tvecs[0,0,1]
            zs[j] = tvecs[0,0,2]
            

        color[j] = j 
        time_vect[j] = time.time() - t0
        rot,_ = cv2.Rodrigues(rvecs)


        


        j = j+1
        print j
    # plt.scatter (time_vect,xs,color= 'red')
    # plt.scatter (time_vect,ys,color= 'green')
    # plt.scatter (time_vect,zs,color= 'blue')
    # # plt.show()
    # plt.grid(linewidth=1)
    #                 # ax.set_yticks(np.arange(-120,41,10))
    # plt.pause(0.05)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or j >= iterations_for_while:
            break

cap.release()
cv2.destroyAllWindows()

## removing list elements whicj didn't update
del(xs[j:])  
del(ys[j:])  
del(zs[j:])
del(rxs[j:])  
del(rys[j:])  
del(rzs[j:])    
del(color[j:])
del(time_vect[j:])



print len(rxs),"len(rxs)"
print len(xs),"len(xs)"
print rxs, "rxs"
# When everything done, release the capture
print np.std(xs), "std_x"
print np.std(ys), "std_y"
print np.std(zs), "std_z"

###Saving the data
# np.savetxt("x_coordinate_along_straight_line",xs,delimiter=',')
# np.savetxt("y_coordinate_along_straight_line",ys,delimiter=',')
# np.savetxt("z_coordinate_along_straight_line",zs,delimiter=',')
# np.savetxt("time_vect",time_vect,delimiter=',')
# np.savetxt("all_coordinate_along_straight_line",np.array([xs,ys,zs]).T,delimiter=',')


### distance travelled 
x_change = xs[0]-xs[-1]
y_change = ys[0]-ys[-1]
z_change = zs[0]-zs[-1]


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.scatter(xs,ys,zs,c=color)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.scatter(rxs,rys,rzs,c=color)


# ax.xlim = (-35 ,-45)
# ax.ylim = (-30 ,20)
# ax.zlim = (300 ,400)
# ax.axis('equal')

# plt.show()
print sub_pix_refinement_switch, "sub_pix_refinement_switch"  
print detect_tip_switch, "detect_tip_switch" 


# plt.hist(xs,20,facecolor='red',density=True)
# plt.hist(ys,20,facecolor='green',density=True)
# plt.hist(zs,20,facecolor='blue',density=True)

# fig = plt.figure()
# plt.hist(xs,20,facecolor='red',density=True)
# fig = plt.figure()
# plt.hist(ys,20,facecolor='green',density=True)
# fig = plt.figure()
# plt.hist(zs,20,facecolor='blue',density=True)


plt.show()
