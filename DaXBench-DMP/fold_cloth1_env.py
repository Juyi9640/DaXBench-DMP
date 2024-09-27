import os
import time
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from array import array

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from daxbench.core.engine.cloth_simulator import ClothState
from daxbench.core.envs.basic.cloth_env import ClothEnv
from daxbench.core.utils.util import get_expert_start_end_cloth

from daxbench.core.cartesian_dmp.cartesian_dmp import CartesianDMP
from daxbench.core.cartesian_dmp.quaternion_dmp import QuaternionDMP
from daxbench.core.cartesian_dmp.trajectory_gen import Trajectory3D

my_path = os.path.dirname(os.path.abspath(__file__))


@dataclass
class DefaultConf:
    N = 80
    cell_size = 1.0 / N
    gravity = 0.5
    stiffness = 900
    damping = 2
    dt = 2e-3
    max_v = 2.
    small_num = 1e-8
    mu = 0.5  # friction
    seed = 1
    size = int(N / 5.0)
    mem_saving_level = 2
    # 1:lesser memory, but faster
    # 2:much lesser memory but much slower
    task = "fold_cloth1"
    goal_path = f"{my_path}/goals/{task}/goal.npy"
    use_substep_obs = True


FoldCloth1Conf = DefaultConf


class FoldCloth1Env(ClothEnv):

    def __init__(self, batch_size, conf=None, aux_reward=False, seed=1):
        conf = DefaultConf() if conf is None else conf
        max_steps = 3
        super().__init__(conf, batch_size, max_steps, aux_reward)
        self.observation_size = 1544

    def create_cloth_mask(self, conf):
        N, size = conf.N, conf.size
        cloth_mask = jnp.zeros((N, N))
        cloth_mask = cloth_mask.at[size * 2:size * 3, size * 2:size * 4].set(1)

        return cloth_mask
    

def plot_trajectories(trajectories):
    """
    Plot the trajectories of the cloth particles during the pick and place actions.
    :param trajectories: List of numpy arrays containing particle positions at each step.
    """
    first_cloth_particle = []
    for step, positions in enumerate(trajectories):
        # print("shape of step: \n", np.shape(step))
        # print("shape of positions: \n", np.shape(positions))
        first_cloth_particle.append(positions[0][31])



    # convert the list to array
    first_particle_array = np.array(first_cloth_particle)
    #print("particle array: \n",first_particle_array)
    # exchange [Y,Z,X] to [X,Y,Z]
    first_particle_array_swapped = first_particle_array[:,[2,0,1]]
    np.save('outfile_swapped_31', first_particle_array_swapped)
    # print("particle 3d trajectories: \n", first_particle_array_swapped)


    # # plot trajectories
    first_particle_array = first_particle_array.T
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot(first_particle_array[0], first_particle_array[1], first_particle_array[2],marker='x')
    ax.scatter(*first_particle_array.T[0], color = 'red')
    plt.savefig('trajectories.png')  # Save the plot as an image file
    plt.close()

    return first_particle_array_swapped

def plot_distance(point_xyz_array):
    # print("point_xyz_array", point_xyz_array)
    # print("point_xyz_array shape:", point_xyz_array.shape)
    start_point_xyz = point_xyz_array[0]

    # calculate Euclidean distance
    distance = np.sqrt(np.sum((point_xyz_array - start_point_xyz) ** 2, axis=1))
    #print("distance: ",distance)
    length = len(distance)
    # generate a array counting from 1 to length
    count_array = np.arange(1, length + 1)
    # print("counting the step: \n",count_array)

    # plot
    plt.plot(count_array, distance)
    # naming the x axis
    plt.xlabel("Step")
    # naming the y axis
    plt.ylabel("Distance between current point and start point")
    # title of graph
    plt.title("Step - P graph")
    plt.savefig('Step-P.png')


if __name__ == "__main__":
    env = FoldCloth1Env(batch_size=1)
    env.seed(1)
    # mute env.seed(1) and run next:
    # print(env.simulator.seed): seed is 1, proves that the conf is defined
    # in child but shared across all parents
    
    # env.collect_goal()
    # env.collect_expert_demo(10)

    obs, state = env.reset(env.simulator.key_global)

    # actions = np.zeros((env.batch_size, 6))
    # env.step_diff(actions, state)  # to compile the jax module

    # interactive test
    print("time start")
    start_time = time.time()
    for _ in range(100):
        actions, my_actions_numpy = get_expert_start_end_cloth(env.get_x_grid(state), env.cloth_mask)

        print("start and goal: \n", my_actions_numpy)
        traj_obj = Trajectory3D()
        dmp_obj = CartesianDMP(alphaz=25.0,betaz=25.0/4.0,orientation=False)

        parabola_demo = traj_obj.gen_dynamic_sinusoidal_curve_start_goal(my_actions_numpy)
        parabola_dmp = dmp_obj.downsampling(dmp_obj.imitate(parabola_demo))
        path_for_sim = dmp_obj.gen_incremental_vector(parabola_dmp)

        # actions = env.get_random_fold_action(state)
        # obs, reward, done, info = env.step_diff(actions, state)
        obs, reward, done, info = env.step_with_render(actions, state, path_for_sim)
        point_xyz = plot_trajectories(info["trajectory"])
        plot_distance(point_xyz)
        state = info['state']
    print(time.time() - start_time)
    
    
    
    
    

    
    
    
