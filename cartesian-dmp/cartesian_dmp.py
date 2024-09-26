import numpy as np
from scipy import interpolate
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.spatial.distance import directed_hausdorff
import copy

import quaternion_dmp
import trajectory_gen
import discrete_frechet

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



class CartesianDMP():
    
    def __init__(self,N_bf=20,alphaz=4.0,betaz=1.0,orientation=True):
        
        self.alphax = 1.0
        self.alphaz = alphaz
        self.betaz = betaz
        self.N_bf = N_bf # number of basis functions
        self.tau = 1.0 # temporal scaling

        self.phase = 1.0 # initialize phase variable

        # Orientation dmp
        self.orientation = orientation 
        if orientation:
            self.dmp_ori = quaternion_dmp.QuaternionDMP(self.N_bf,self.alphax,self.alphaz,self.betaz,self.tau)

    def imitate(self, pose_demo, sampling_rate=100, oversampling=True):
        
        self.T = pose_demo.shape[0] / sampling_rate
        
        if not oversampling:
            self.N = pose_demo.shape[0]
            self.dt = self.T / self.N
            self.x = pose_demo[:,:3]
            
        else:
            self.N = 10 * pose_demo.shape[0] # 10-fold oversample
            self.dt = self.T / self.N

            t = np.linspace(0.0,self.T,pose_demo[:,0].shape[0])
            self.x_des = np.zeros([self.N,3])
            for d in range(3):
                x_interp = interpolate.interp1d(t,pose_demo[:,d])
                for n in range(self.N):
                    self.x_des[n,d] = x_interp(n * self.dt)
                
        # Centers of basis functions 
        self.c = np.ones(self.N_bf) 
        c_ = np.linspace(0,self.T,self.N_bf)
        for i in range(self.N_bf):
            self.c[i] = np.exp(-self.alphax *c_[i])

        # Widths of basis functions 
        # (as in https://github.com/studywolf/pydmps/blob/80b0a4518edf756773582cc5c40fdeee7e332169/pydmps/dmp_discrete.py#L37)
        self.h = np.ones(self.N_bf) * self.N_bf**1.5 / self.c / self.alphax

        self.dx_des = np.gradient(self.x_des,axis=0)/self.dt
        self.ddx_des = np.gradient(self.dx_des,axis=0)/self.dt

        # Initial and final orientation
        self.x0 = self.x_des[0,:]
        self.dx0 = self.dx_des[0,:] 
        self.ddx0 = self.ddx_des[0,:]
        self.xT = self.x_des[-1,:]

        # Initialize the DMP
        self.x = copy.deepcopy(self.x0)
        self.dx = copy.deepcopy(self.dx0)
        self.ddx = copy.deepcopy(self.ddx0)

        # Evaluate the phase variable
        # self.phase = np.exp(-self.alphax*np.linspace(0.0,self.T,self.N))

        # Evaluate the forcing term
        forcing_target_pos = self.tau*self.ddx_des - self.alphaz*(self.betaz*(self.xT-self.x_des) - self.dx_des)

        self.fit_dmp(forcing_target_pos)
        
        # Imitate orientation
        if self.orientation:
            q_des = self.dmp_ori.imitate(pose_demo[:,3:], sampling_rate, oversampling)
            return self.x_des, q_des
        else:
            return self.x_des
    
    def RBF(self, phase):

        if type(phase) is np.ndarray:
            return np.exp(-self.h*(phase[:,np.newaxis]-self.c)**2)
        else:
            return np.exp(-self.h*(phase-self.c)**2)

    def forcing_function_approx(self,weights,phase,xT=1,x0=0):

        BF = self.RBF(phase)
        if type(phase) is np.ndarray:
            return np.dot(BF,weights)*phase/np.sum(BF,axis=1)
        else:
            return np.dot(BF,weights)*phase/np.sum(BF)
    
    def fit_dmp(self,forcing_target):

        phase = np.exp(-self.alphax*np.linspace(0.0,self.T,self.N))
        BF = self.RBF(phase)
        X = BF*phase[:,np.newaxis]/np.sum(BF,axis=1)[:,np.newaxis]

        self.weights_pos = np.zeros([self.N_bf,3])

        # for d in range(3):
        #     self.weights_pos[:,d] = np.dot(np.linalg.pinv(X),forcing_target[:,d])

        regcoef = 0.01
        for d in range(3):        
            self.weights_pos[:,d] = np.dot(np.dot(np.linalg.pinv(np.dot(X.T,(X)) + \
                                    regcoef * np.eye(X.shape[1])),X.T),forcing_target[:,d].T) 

    def reset(self):
        
        self.phase = 1.0
        self.x = copy.deepcopy(self.x0)
        self.dx = copy.deepcopy(self.dx0)
        self.ddx = copy.deepcopy(self.ddx0)

        if self.orientation:
            self.dmp_ori.reset()

    def step(self, disturbance=None):
        
        disturbance_pos = np.zeros(3)
        disturbance_ori = np.zeros(3)

        if disturbance is None:
            disturbance = np.zeros(6)
        else:
            disturbance_pos = disturbance[:3]
            disturbance_ori = disturbance[3:]
        
        self.phase += (-self.alphax * self.tau * self.phase) * (self.T/self.N)
        forcing_term_pos = self.forcing_function_approx(self.weights_pos,self.phase)

        self.ddx = self.alphaz * (self.betaz * (self.xT - self.x) - self.dx) + forcing_term_pos + disturbance_pos
        self.dx += self.ddx * self.dt * self.tau
        self.x += self.dx * self.dt * self.tau

        if self.orientation:
            q, dq, ddq = self.dmp_ori.step(disturbance=disturbance_ori)
            return copy.deepcopy(self.x), copy.deepcopy(self.dx), copy.deepcopy(self.ddx), q, dq, ddq
        else:
            return copy.deepcopy(self.x), copy.deepcopy(self.dx), copy.deepcopy(self.ddx)

    def rollout(self,tau=1.0,xT=None):

        x_rollout = np.zeros([self.N,3])
        dx_rollout = np.zeros([self.N,3])
        ddx_rollout = np.zeros([self.N,3])
        x_rollout[0,:] = self.x0
        dx_rollout[0,:] = self.dx0
        ddx_rollout[0,:] = self.ddx0
        
        if xT is None:
            xT = self.xT
        
        phase = np.exp(-self.alphax*tau*np.linspace(0.0,self.T,self.N))

        # Position forcing term
        forcing_term_pos = np.zeros([self.N,3])
        for d in range(3):
            forcing_term_pos[:,d] = self.forcing_function_approx(
                self.weights_pos[:,d],phase,xT[d],self.x0[d])

        for d in range(3):
            for n in range(1,self.N):
                ddx_rollout[n,d] = self.alphaz*(self.betaz*(xT[d]-x_rollout[n-1,d]) - \
                                               dx_rollout[n-1,d]) + forcing_term_pos[n,d]
                dx_rollout[n,d] = dx_rollout[n-1,d] + tau*ddx_rollout[n-1,d]*self.dt
                x_rollout[n,d] = x_rollout[n-1,d] + tau*dx_rollout[n-1,d]*self.dt
        
        # Get orientation rollout
        if self.orientation:
            q_rollout,dq_log_rollout,ddq_log_rollout = self.dmp_ori.rollout(tau=tau)
            return x_rollout,dx_rollout,ddx_rollout, q_rollout,dq_log_rollout,ddq_log_rollout
        else:
            return x_rollout,dx_rollout,ddx_rollout

    def euclidean_distance(self,x1,x2):
        square_distance = np.sum((x1-x2)**2,axis=0)
        distance = np.sqrt(square_distance)
        sum_distance = np.sum(distance)
        return sum_distance

        return np.linalg.norm(x1-x2)

    def downsampling(self, oversample):
        # down sample because of 10-fold oversample
        downsample = np.zeros((int(oversample.shape[0]/10), 3))
        for i in range(downsample.shape[0]):
            downsample[i,:] = oversample[i*10,:]

        return downsample




# Test

if __name__ == "__main__":

    # traj without orientation
    traj = trajectory_gen.Trajectory3D()

    helix_demo = traj.gen_circular_helix()
    sinusoid_demo = traj.gen_sinusoidal_curve()
    spiral_demo = traj.gen_spiral_on_a_plane()
    parabola_demo = traj.gen_parabolic_curve()
    curve_1_demo = traj.gen_curve_1()
    curve_daxbench = traj.load_traj_from_file()
    curve_random_demo = traj.gen_dynamic_sinusoidal_curve_start_goal()

    dynamic_parabola_demo = traj.gen_dynamic_parabola()
    dynamic_sinu_demo = traj.gen_dynamic_sinusoidal_curve_start_goal()



    # CartesianDMP
    dmp = CartesianDMP(alphaz=25.0,betaz=25.0/4.0,orientation=False)

    helix_dmp = dmp.downsampling(dmp.imitate(helix_demo))
    sinusoid_dmp = dmp.downsampling(dmp.imitate(sinusoid_demo))
    spiral_dmp = dmp.downsampling(dmp.imitate(spiral_demo))
    parabola_dmp = dmp.downsampling(dmp.imitate(parabola_demo))
    curve_1_dmp = dmp.downsampling(dmp.imitate(curve_1_demo))
    curve_daxbench_dmp = dmp.downsampling(dmp.imitate(curve_daxbench))
    curve_random_dmp = dmp.downsampling(dmp.imitate(curve_random_demo))

    dynamic_parabola_dmp = dmp.downsampling(dmp.imitate(dynamic_parabola_demo))
    dynamic_sinu_dmp = dmp.downsampling(dmp.imitate(dynamic_sinu_demo))





    # helix_demo = curve_daxbench
    # helix_dmp = curve_daxbench_dmp



    # calculate Frechet distance
    discretFrechet = discrete_frechet.DiscreteFrechet(discrete_frechet.euclidean)

    p = helix_demo
    q = helix_dmp

    # Frechet and Hausdorff
    frechet_distance = discretFrechet.distance(p,q)
    print("Frechet distance: \n",frechet_distance)

    symmetric_hausdorff_distance = max(directed_hausdorff(p,q)[0],directed_hausdorff(q,p)[0])
    print("symmetric hausdorff distance: \n", symmetric_hausdorff_distance)
    print("the indices of the points that generate the Hausdorff distance (the Hausdorff pari): \n", directed_hausdorff(p,q)[1:])






    # # plots
    # Create a figure with 1 row and 3 columns
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={'projection': '3d'})

    # First trajectory (helix_demo)
    x, y, z = helix_demo[:, 0], helix_demo[:, 1], helix_demo[:, 2]
    axs[0].scatter(x, y, z, c='r', s=100)
    axs[0].plot(x, y, z, c='r')
    axs[0].set_title('Random Curve Demo Trajectory')
    axs[0].set_xlabel('X Axis')
    axs[0].set_ylabel('Y Axis')
    axs[0].set_zlabel('Z Axis')

    # Second trajectory (helix_dmp)
    x1, y1, z1 = helix_dmp[:, 0], helix_dmp[:, 1], helix_dmp[:, 2]
    axs[1].scatter(x1, y1, z1, c='b', s=100)
    axs[1].plot(x1, y1, z1, c='b')
    axs[1].set_title('DMP Imitation Trajectory')
    axs[1].set_xlabel('X Axis')
    axs[1].set_ylabel('Y Axis')
    axs[1].set_zlabel('Z Axis')

    # Combined plot (both helix_demo and helix_dmp)
    axs[2].scatter(x, y, z, c='r', s=100)
    axs[2].plot(x, y, z, c='r', label='First Trajectory')
    axs[2].scatter(x1, y1, z1, c='b', s=100)
    axs[2].plot(x1, y1, z1, c='b', label='Second Trajectory')
    axs[2].set_title('Combined Trajectories')
    axs[2].set_xlabel('X Axis')
    axs[2].set_ylabel('Y Axis')
    axs[2].set_zlabel('Z Axis')

    # Adjust layout
    plt.tight_layout()
    plt.show()
















