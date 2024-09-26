import numpy as np



class Trajectory3D:

    def __init__(self):

        self.path_for_simulator = self.gen_dynamic_sinusoidal_curve_start_goal()


        self.path_for_simulator = np.delete(self.path_for_simulator, (2, 19), axis=0)
        # np.save('path_for_simulator', self.path_for_simulator)

        self.path_for_DMP = self.gen_dynamic_sinusoidal_curve_start_goal()
        # np.save('sinusoidal_start_goal',self.path_for_DMP)


    def load_traj_from_file(self):
        # load the path generated from daxbench
        return np.load(
            'outfile_swapped_31.npy')


    def gen_circular_helix(self):
        # Circular helix parameters
        num_points = 40
        radius = 5
        pitch = 2 * np.pi  # How tight the helix is (distance between turns)
        z_increment = 1  # How quickly the helix rises along the z-axis

        # Circular helix trajectory
        path_helix = np.zeros((num_points, 3))
        theta = np.linspace(0, 4 * np.pi, num_points)  # 2 full turns

        for i in range(num_points):
            path_helix[i, 0] = radius * np.cos(theta[i])  # x = r*cos(θ)
            path_helix[i, 1] = radius * np.sin(theta[i])  # y = r*sin(θ)
            path_helix[i, 2] = z_increment * theta[i] / pitch  # z increases linearly

        return path_helix

    def gen_sinusoidal_curve(self):
        # Sinusoidal curve parameters
        num_points = 40
        amplitude = 5
        frequency = 0.2

        # Sinusoidal trajectory
        path_sinusoid = np.zeros((num_points, 3))
        x = np.linspace(0, 10, num_points)

        for i in range(num_points):
            path_sinusoid[i, 0] = x[i]  # Linear x-coordinate
            path_sinusoid[i, 1] = amplitude * np.sin(2 * np.pi * frequency * x[i])  # y = A*sin(ωx)
            path_sinusoid[i, 2] = 0  # z-coordinate (optional, can keep it zero if only 2D sinusoid is needed)

        return path_sinusoid

    def gen_dynamic_sinusoidal_curve_start_goal(self):
        # Sinusoidal curve parameters
        num_points = 40
        amplitude = 0.05  # Adjusted amplitude for a smaller sinusoid
        frequency = 8  # Frequency adjusted to fit within the distance

        # Define the start and goal points
        start_point = np.array([0.388, 0.285, 0])
        goal_point = np.array([0.570, 0.660, 0])

        # Generate linearly spaced points between start and goal in the x direction
        x_values = np.linspace(start_point[0], goal_point[0], num_points)

        # Generate y and z values that linearly interpolate between start and goal points
        y_linear = np.linspace(start_point[1], goal_point[1], num_points)

        # z_linear = np.linspace(start_point[2], goal_point[2], num_points)
        z_1 = np.linspace(start_point[2], 0.3, 20)
        z_2 = np.linspace(0.3, goal_point[2], 20)
        z_linear = np.concatenate((z_1, z_2))

        # Create the sinusoidal perturbation along the y-direction
        y_values = y_linear + amplitude * np.sin(2 * np.pi * frequency * np.linspace(0, 1, num_points))

        # Adding Gaussian noise to x, y, and z values
        noise_level = 0.005  # Adjust noise level as needed
        x_values += np.random.normal(0, noise_level, num_points)
        y_values += np.random.normal(0, noise_level, num_points)
        z_linear += np.random.normal(0, noise_level, num_points)

        # Combine x, y, and z values into the trajectory array
        path_sinusoid = np.vstack((x_values, y_values, z_linear)).T

        return path_sinusoid



    def gen_spiral_on_a_plane(self):
        # Spiral parameters
        num_points = 40
        a = 0.5  # Initial radius
        b = 0.2  # Spiral growth rate

        # Spiral trajectory
        path_spiral = np.zeros((num_points, 3))
        theta = np.linspace(0, 4 * np.pi, num_points)

        for i in range(num_points):
            r = a + b * theta[i]  # r = a + bθ
            path_spiral[i, 0] = r * np.cos(theta[i])  # x = r*cos(θ)
            path_spiral[i, 1] = r * np.sin(theta[i])  # y = r*sin(θ)
            path_spiral[i, 2] = 0  # z-coordinate (spiral is in the xy-plane)

        return path_spiral

    def gen_parabolic_curve(self):
        # Parabolic curve parameters
        num_points = 40
        a = 0.1  # Parabolic coefficient

        # Parabolic trajectory
        path_parabola = np.zeros((num_points, 3))
        x = np.linspace(-10, 10, num_points)

        for i in range(num_points):
            path_parabola[i, 0] = x[i]  # Linear x-coordinate
            path_parabola[i, 2] = a * x[i] ** 2  # y = ax^2 (parabola)
            path_parabola[i, 1] = 0  # z-coordinate (optional)

        return path_parabola

    def gen_parabola_start_end(self):
        # Define the start and goal points
        start_point = np.array([0.388, 0.285, 0])
        goal_point = np.array([0.570, 0.660, 0])

        # Number of points in the trajectory
        num_points = 40

        # Generate x and y values
        x_values = np.linspace(start_point[0], goal_point[0], num_points)
        y_values = np.linspace(start_point[1], goal_point[1], num_points)

        # Generate z values using a parabolic equation z = a*t^2 + b*t + c
        # where t is a normalized parameter from 0 to 1
        t_values = np.linspace(0, 1, num_points)

        # Let's assume the maximum height of the "bridge" at t=0.5 is h_max
        h_max = 0.2  # You can adjust this value to change the arch height

        # The equation for a parabola passing through (0, 0), (0.5, h_max), and (1, 0)
        # can be written as z(t) = 4 * h_max * t * (1 - t)

        z_values = 4 * h_max * t_values * (1 - t_values)

        # Combine x, y, z values into the trajectory array
        path_1 = np.vstack((x_values, y_values, z_values)).T

        return path_1

    def gen_curve_1(self):
        # Test system without orientation
        demo = np.zeros((40,3))
        demo[:,0] = np.sin(np.arange(0, 0.4, 0.01) * 5)
        demo[:,1] = np.arange(0, 0.4, 0.01) * 5
        demo[:,2] = np.cos(np.arange(0, 0.4, 0.01) * 5)
        return demo

    def gen_dynamic_parabola(self):
        # Define the start and goal points
        start_point = np.array([0.388, 0.285, 0])
        goal_point = np.array([0.570, 0.660, 0])

        # Number of points in the trajectory
        num_points = 40

        # Generate x and y values
        x_values = np.linspace(start_point[0], goal_point[0], num_points)
        y_values = np.linspace(start_point[1], goal_point[1], num_points)

        # Generate z values using a parabolic equation z = a*t^2 + b*t + c
        # where t is a normalized parameter from 0 to 1
        t_values = np.linspace(0, 1, num_points)

        # Let's assume the maximum height of the "bridge" at t=0.5 is h_max
        h_max = 0.2  # Adjust this value to change the arch height

        # The equation for a parabola passing through (0, 0), (0.5, h_max), and (1, 0)
        # z(t) = 4 * h_max * t * (1 - t)
        z_values = 4 * h_max * t_values * (1 - t_values)

        # Adding Gaussian noise to x, y, and z values
        noise_level = 0.005  # Adjust noise level as needed
        x_values += np.random.normal(0, noise_level, num_points)
        y_values += np.random.normal(0, noise_level, num_points)
        z_values += np.random.normal(0, noise_level, num_points)

        # Adding sinusoidal perturbation to z values for dynamic behavior
        sine_amplitude = 0.05  # Adjust amplitude as needed
        sine_frequency = 3  # Adjust frequency as needed
        z_values += sine_amplitude * np.sin(2 * np.pi * sine_frequency * t_values)

        # Combine x, y, z values into the trajectory array
        path_1 = np.vstack((x_values, y_values, z_values)).T

        return path_1

True

if __name__ == "__main__":


    traj = Trajectory3D()












