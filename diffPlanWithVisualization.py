import pybullet as p
import pybullet_data
import numpy as np
import time
import math
from scipy.interpolate import BSpline, splev
import torch
import torch.nn as nn
import torch.nn.functional as F
import colorsys

def safe_tensor(x, min_value=-10, max_value=10):
    """
    Replace NaNs with zero and clamp values to a safe range.
    Args:
        x: Input tensor
        min_value: Minimum allowed value
        max_value: Maximum allowed value
    Returns:
        Sanitized tensor with NaNs replaced and values clamped
    """
    x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
    x = torch.clamp(x, min_value, max_value)
    return x

class BSplineTrajectory:
    """
    Represents a trajectory using B-spline curves for smooth interpolation.
    Based on the B-spline parameterization described in the MPD paper.
    """
    def __init__(self, control_points, degree=3, num_points=100):
        """
        Initialize a B-spline trajectory.
        Args:
            control_points: Array of control points (n_points x dim)
            degree: Degree of the B-spline (default: 3 for cubic)
            num_points: Number of points to evaluate along the curve
        """
        self.control_points = control_points
        self.degree = degree
        self.num_points = num_points
        n = len(control_points)
        k = degree
        
        # Create clamped knot vector (ensures curve passes through endpoints)
        self.knots = np.concatenate((
            np.zeros(k),
            np.linspace(0, 1, n - k + 1),
            np.ones(k)
        ))
        
        # Parameter values for evaluation
        self.u = np.linspace(0, 1, num_points)
        
    def evaluate(self):
        """
        Evaluate the B-spline at the parameter values.
        Returns:
            Array of points along the curve (num_points x dim)
        """
        points = np.zeros((self.num_points, self.control_points.shape[1]))
        for i in range(self.control_points.shape[1]):
            points[:, i] = splev(self.u, (self.knots, self.control_points[:, i], self.degree))
        return points

class DiffusionNetwork(nn.Module):
    """
    Neural network for the diffusion model that predicts noise.
    This network takes a state vector and time step as input and predicts
    the noise component to be removed in the reverse diffusion process.
    """
    def __init__(self, state_dim, hidden_dim=256):
        """
        Initialize the diffusion network.
        Args:
            state_dim: Dimension of the state vector (n_control_points * 2 for 2D)
            hidden_dim: Size of hidden layers
        """
        super(DiffusionNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + 1, hidden_dim)  # +1 for time step
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, state_dim)
        
    def forward(self, x):
        """
        Forward pass of the network.
        Args:
            x: Input tensor containing flattened state and time step
               Shape: (batch_size, state_dim + 1)
        Returns:
            Predicted noise with shape (batch_size, state_dim)
        """
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        eps = self.fc4(h)
        return eps

class MotionPlanningDiffusion:
    """
    Implements Motion Planning Diffusion (MPD) for trajectory generation.
    Based on the paper: "Motion Planning Diffusion: Learning and Planning of Robot Motions
    with Diffusion Models" (https://arxiv.org/pdf/2412.19948)
    """
    def __init__(self, start_pos, goal_pos, obstacles, n_control_points=10, num_diffusion_steps=100, device='cpu'):
        """
        Initialize the MPD planner.
        Args:
            start_pos: Starting position [x, y, z]
            goal_pos: Goal position [x, y, z]
            obstacles: List of obstacle IDs in PyBullet
            n_control_points: Number of control points for B-spline
            num_diffusion_steps: Number of steps in the diffusion process
            device: Device to run computations on ('cpu' or 'cuda')
        """
        # Store only x,y for 2D planning
        self.start_pos = np.array(start_pos)[:2]
        self.goal_pos = np.array(goal_pos)[:2]
        self.obstacles = obstacles
        self.n_control_points = n_control_points
        self.num_diffusion_steps = num_diffusion_steps
        self.device = device
        self.debug_line_ids = []  # Store line IDs for visualization

        # Set up diffusion process parameters
        # These define the noise schedule for the forward and reverse processes
        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.betas = torch.linspace(self.beta_start, self.beta_end, num_diffusion_steps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-8)  # Avoid log(0)

        # Define state dimension and create model
        self.state_dim = self.n_control_points * 2  # 2D control points
        self.model = DiffusionNetwork(state_dim=self.state_dim).to(device)
        self._initialize_model_weights()

    def _initialize_model_weights(self):
        """
        Initialize model weights using Xavier initialization.
        In a real implementation, you would load weights from a trained model.
        """
        for param in self.model.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)

    def _get_obstacle_positions(self):
        """
        Get the positions of obstacles from PyBullet.
        Returns:
            Array of obstacle positions (n_obstacles x 2)
        """
        positions = []
        for obs in self.obstacles:
            pos, _ = p.getBasePositionAndOrientation(obs)
            positions.append(pos[:2])  # Only use x,y for 2D planning
        return np.array(positions)

    def _collision_cost(self, trajectory):
        """
        Calculate collision cost for a trajectory.
        Higher cost for points closer to obstacles.
        Args:
            trajectory: Array of points along the trajectory
        Returns:
            Collision cost (scalar)
        """
        obstacle_positions = self._get_obstacle_positions()
        if isinstance(trajectory, torch.Tensor):
            trajectory = trajectory.detach().cpu().numpy()
        
        cost = 0
        for point in trajectory:
            for obs_pos in obstacle_positions:
                dist = np.linalg.norm(point - obs_pos)
                # Apply cost for points within 1.0 unit of an obstacle
                if dist < 1.0:
                    cost += (1.0 - dist) ** 2
        return cost

    def _smoothness_cost(self, trajectory):
        """
        Calculate smoothness cost based on acceleration.
        Args:
            trajectory: Array of points along the trajectory
        Returns:
            Smoothness cost (scalar)
        """
        if isinstance(trajectory, torch.Tensor):
            trajectory = trajectory.detach().cpu().numpy()
            
        # Calculate velocities (first derivatives)
        velocities = np.diff(trajectory, axis=0)
        # Calculate accelerations (second derivatives)
        accelerations = np.diff(velocities, axis=0)
        
        # Return sum of squared accelerations
        return np.sum(accelerations**2)

    def _boundary_cost(self, trajectory):
        """
        Cost to ensure trajectory starts and ends at specified positions.
        Args:
            trajectory: Array of points along the trajectory
        Returns:
            Boundary cost (scalar)
        """
        if isinstance(trajectory, torch.Tensor):
            trajectory = trajectory.detach().cpu().numpy()
            
        start_error = np.linalg.norm(trajectory[0] - self.start_pos)
        goal_error = np.linalg.norm(trajectory[-1] - self.goal_pos)
        
        return 10.0 * (start_error**2 + goal_error**2)

    def _total_cost(self, trajectory):
        """
        Calculate total cost for a trajectory.
        Combines collision, smoothness, and boundary costs.
        Args:
            trajectory: Array of points along the trajectory
        Returns:
            Total cost (scalar)
        """
        collision = self._collision_cost(trajectory)
        smoothness = self._smoothness_cost(trajectory)
        boundary = self._boundary_cost(trajectory)
        
        return collision + 0.1 * smoothness + boundary

    def _cost_gradient(self, x, eps=1e-4):
        """
        Calculate gradient of cost function using finite differences.
        Args:
            x: Control points
            eps: Small perturbation for finite differences
        Returns:
            Gradient of cost with respect to control points
        """
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
            
        grad = np.zeros_like(x_np)
        cost_0 = self._total_cost(self._control_points_to_trajectory(x_np))
        
        # Calculate gradient for each dimension using finite differences
        for i in range(x_np.shape[0]):
            for j in range(x_np.shape[1]):
                x_perturb = x_np.copy()
                x_perturb[i, j] += eps
                cost_perturb = self._total_cost(self._control_points_to_trajectory(x_perturb))
                grad[i, j] = (cost_perturb - cost_0) / eps
        
        # Handle NaNs and infinities in the gradient
        grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        
        return torch.tensor(grad, device=self.device, dtype=torch.float32)

    def _control_points_to_trajectory(self, control_points):
        """
        Convert B-spline control points to a trajectory.
        Args:
            control_points: Array of control points
        Returns:
            Array of points along the trajectory
        """
        if len(control_points.shape) == 1:
            control_points = control_points.reshape(-1, 2)
            
        bspline = BSplineTrajectory(control_points)
        trajectory = bspline.evaluate()
        
        # Check for NaNs and reset to linear trajectory if needed
        if np.isnan(trajectory).any():
            print("NaN detected in trajectory! Resetting to linear trajectory.")
            trajectory = BSplineTrajectory(self._create_linear_trajectory()).evaluate()
            
        return trajectory

    def _create_linear_trajectory(self):
        """
        Create a linear trajectory from start to goal.
        Used as a fallback and initialization.
        Returns:
            Array of control points for a linear trajectory
        """
        control_points = np.zeros((self.n_control_points, 2))
        for i in range(self.n_control_points):
            t = i / (self.n_control_points - 1)
            control_points[i] = (1 - t) * self.start_pos + t * self.goal_pos
        return control_points

    def _p_sample(self, x_t, t):
        """
        Sample from p(x_{t-1} | x_t) - reverse diffusion process with guidance.
        Implements the reverse diffusion step with cost function guidance.
        Args:
            x_t: Current state at time t
            t: Current time step
        Returns:
            Sampled state at time t-1
        """
        # Flatten control points and concatenate time
        x_t_flat = x_t.view(-1).unsqueeze(0)  # (1, n_control_points*2)
        t_tensor = torch.tensor([[t / self.num_diffusion_steps]], device=self.device)  # (1, 1)
        input_tensor = torch.cat([x_t_flat, t_tensor], dim=1)  # (1, n_control_points*2 + 1)
        
        # Predict noise using the model
        with torch.no_grad():
            predicted_noise = self.model(input_tensor)  # (1, n_control_points*2)
        predicted_noise = predicted_noise.view(self.n_control_points, 2)

        # Get alpha values for timestep t
        alpha = self.alphas[t]
        alpha_cumprod = self.alphas_cumprod[t]
        alpha_cumprod_prev = self.alphas_cumprod_prev[t]
        beta = self.betas[t]

        # Posterior mean calculation (using the diffusion model equations)
        # These equations come from the DDPM paper and are adapted for motion planning
        coef1 = torch.sqrt(alpha_cumprod_prev) / (1.0 - alpha_cumprod + 1e-8)
        coef2 = (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod + 1e-8)
        sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha + 1e-8)
        posterior_mean = coef1 * (x_t - ((1.0 - alpha_cumprod) / sqrt_one_minus_alpha) * predicted_noise)

        # Cost guidance - this is a key contribution of the MPD paper
        # It guides the diffusion process using the cost function gradient
        if t > 0:
            cost_grad = self._cost_gradient(x_t)
            # Guidance strength decreases over time
            guidance_strength = 0.1 * (1.0 - t / self.num_diffusion_steps)
            posterior_mean = posterior_mean - guidance_strength * cost_grad

        # Calculate posterior variance and log variance
        posterior_variance = torch.clamp(self.posterior_variance[t], min=1e-8)
        posterior_log_variance = torch.log(posterior_variance)
        
        # Sample x_{t-1} using the posterior mean and variance
        noise = torch.randn_like(x_t)
        x_t_minus_1 = posterior_mean + torch.exp(0.5 * posterior_log_variance) * noise

        # Clamp and replace NaNs for numerical stability
        x_t_minus_1 = safe_tensor(x_t_minus_1)
        return x_t_minus_1
    
    def _visualize_trajectory(self, control_points, t, lifetime=0.0):
        """
        Visualize the current trajectory during diffusion.
        This is a key addition for visualizing the diffusion process.
        
        Args:
            control_points: B-spline control points
            t: Current diffusion time step
            lifetime: Lifetime of visualization lines (0.0 for persistent)
        """
        # Clear previous visualization lines if needed
        if lifetime == 0.0 and self.debug_line_ids:
            for line_id in self.debug_line_ids:
                p.removeUserDebugItem(line_id)
            self.debug_line_ids = []
            
        # Convert control points to trajectory
        if isinstance(control_points, torch.Tensor):
            control_points = control_points.detach().cpu().numpy()
        
        # Generate trajectory from control points
        trajectory = BSplineTrajectory(control_points).evaluate()
        
        # Calculate color based on diffusion step (from red at t=max to green at t=0)
        # Using HSV color space for smoother color transitions
        hue = 0.33 * (1.0 - t / self.num_diffusion_steps)  # 0 (red) to 0.33 (green)
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        color = [r, g, b]
        
        # Draw the trajectory
        line_ids = []
        for i in range(len(trajectory) - 1):
            # Add small z-offset for visualization
            start = [trajectory[i, 0], trajectory[i, 1], 0.12]
            end = [trajectory[i+1, 0], trajectory[i+1, 1], 0.12]
            line_id = p.addUserDebugLine(start, end, color, 2, lifetime)
            line_ids.append(line_id)
        
        # Store line IDs if we want to remove them later
        if lifetime == 0.0:
            self.debug_line_ids = line_ids
        
        # Also visualize control points
        for i in range(len(control_points)):
            point = [control_points[i, 0], control_points[i, 1], 0.15]
            p.addUserDebugPoints([point], [[1, 0, 1]], 5, lifetime)  # Magenta points

    def plan(self, visualize_steps=True, visualization_interval=5):
        """
        Generate a trajectory using the diffusion model.
        Implements the full reverse diffusion process from noise to trajectory.
        
        Args:
            visualize_steps: Whether to visualize intermediate steps
            visualization_interval: How often to visualize (every N steps)
        
        Returns:
            3D trajectory for the robot
        """
        print("Planning with Motion Planning Diffusion...")
        
        # Start with random noise for control points
        x_T = torch.randn(self.n_control_points, 2, device=self.device)
        
        # Ensure start and end points are fixed
        x_T[0] = torch.tensor(self.start_pos, device=self.device)
        x_T[-1] = torch.tensor(self.goal_pos, device=self.device)
        
        # Visualize initial noisy trajectory
        if visualize_steps:
            self._visualize_trajectory(x_T, self.num_diffusion_steps, lifetime=0.0)
            time.sleep(0.5)  # Pause to see initial state
        
        # Reverse diffusion process
        x_t = x_T
        for t in reversed(range(self.num_diffusion_steps)):
            if t % 10 == 0:
                print(f"Diffusion step {t}/{self.num_diffusion_steps}")
            
            # Sample from p(x_{t-1} | x_t)
            x_t = self._p_sample(x_t, t)
            x_t = safe_tensor(x_t)
            
            # Fix start and end points
            x_t[0] = torch.tensor(self.start_pos, device=self.device)
            x_t[-1] = torch.tensor(self.goal_pos, device=self.device)
            
            # Reset to linear trajectory if NaNs detected
            if torch.isnan(x_t).any():
                print(f"NaN detected at diffusion step {t}, resetting to linear trajectory.")
                linear = self._create_linear_trajectory()
                x_t = torch.tensor(linear, device=self.device, dtype=torch.float32)
            
            # Visualize intermediate trajectories
            if visualize_steps and t % visualization_interval == 0:
                # Use a short lifetime for intermediate steps to create a fading effect
                lifetime = 0.0 if t % (visualization_interval * 5) == 0 else 0.5
                self._visualize_trajectory(x_t, t, lifetime=lifetime)
                time.sleep(0.1)  # Slow down for visualization
        
        # Get final control points
        control_points = x_t.detach().cpu().numpy()
        if np.isnan(control_points).any():
            print("NaN detected in final control points! Resetting to linear trajectory.")
            control_points = self._create_linear_trajectory()
        
        # Generate final trajectory
        bspline = BSplineTrajectory(control_points)
        trajectory = bspline.evaluate()
        if np.isnan(trajectory).any():
            print("NaN detected in final trajectory! Resetting to linear trajectory.")
            trajectory = BSplineTrajectory(self._create_linear_trajectory()).evaluate()
        
        # Visualize final trajectory with a different color
        if visualize_steps:
            # Clear previous visualizations
            for line_id in self.debug_line_ids:
                p.removeUserDebugItem(line_id)
            self.debug_line_ids = []
            
            # Draw final trajectory
            for i in range(len(trajectory) - 1):
                start = [trajectory[i, 0], trajectory[i, 1], 0.12]
                end = [trajectory[i+1, 0], trajectory[i+1, 1], 0.12]
                p.addUserDebugLine(start, end, [0, 1, 0], 3, 0)  # Green, thicker line
        
        # Add z-coordinate (constant height)
        trajectory_3d = np.zeros((len(trajectory), 3))
        trajectory_3d[:, :2] = trajectory
        trajectory_3d[:, 2] = 0.1  # Constant height
        
        return trajectory_3d

def execute_trajectory(robot_id, trajectory, duration=10.0):
    """
    Execute a trajectory on the robot in PyBullet.
    Args:
        robot_id: PyBullet ID of the robot
        trajectory: Array of points along the trajectory
        duration: Duration of the trajectory execution in seconds
    """
    start_time = time.time()
    trajectory_time = np.linspace(0, duration, len(trajectory))
    
    while time.time() - start_time < duration:
        # Calculate current time in trajectory
        current_time = time.time() - start_time
        if current_time >= duration:
            break
        
        # Find the closest trajectory point
        idx = np.argmin(np.abs(trajectory_time - current_time))
        position = trajectory[idx]
        
        # Check for NaNs
        if np.isnan(position).any():
            print("NaN detected in trajectory execution! Skipping this step.")
            continue
        
        # Calculate orientation (facing the direction of motion)
        if idx < len(trajectory) - 1:
            direction = trajectory[idx+1][:2] - trajectory[idx][:2]
            yaw = math.atan2(direction[1], direction[0])
            orientation = p.getQuaternionFromEuler([0, 0, yaw])
        else:
            orientation = p.getQuaternionFromEuler([0, 0, 0])
        
        # Set robot position and orientation
        p.resetBasePositionAndOrientation(robot_id, position, orientation)
        p.stepSimulation()
        time.sleep(0.01)

def main():
    """
    Main function to set up the environment and run the planner.
    """
    # Initialize PyBullet
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    
    # Load plane
    planeId = p.loadURDF("plane.urdf")
    
    # Load robot
    startPos = [0, 0, 0.1]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    robotId = p.loadURDF("r2d2.urdf", startPos, startOrientation)
    
    # Create obstacles
    obstacles = []
    obstacle_positions = [
        [2, 0, 0.5],
        [1, 1.5, 0.5],
        [3, -1, 0.5],
        [4, 1, 0.5],
        [3, 2, 0.5]
    ]
    
    for pos in obstacle_positions:
        obstacle = p.loadURDF("cube.urdf", pos, globalScaling=0.5)
        obstacles.append(obstacle)
    
    # Set up camera
    p.resetDebugVisualizerCamera(cameraDistance=7, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[2, 0, 0])
    
    # Define start and goal positions
    start_position = [0, 0, 0.1]
    goal_position = [5, 3, 0.1]
    
    # Create the planner
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    planner = MotionPlanningDiffusion(
        start_position,
        goal_position,
        obstacles,
        n_control_points=10,
        num_diffusion_steps=50,  # decrease for faster demo
        device=device
    )
    
    # Generate the trajectory with visualization
    print("Planning trajectory...")
    trajectory = planner.plan(visualize_steps=True, visualization_interval=2)
    print("Trajectory planning complete!")
    
    # Draw final trajectory in a different color
    for i in range(len(trajectory) - 1):
        p.addUserDebugLine(trajectory[i], trajectory[i+1], [0, 1, 0], 3, 0)
    
    # Execute the trajectory
    print("Executing trajectory...")
    execute_trajectory(robotId, trajectory)
    print("Execution complete!")
    
    # Keep the simulation running for a while
    for _ in range(1000):
        p.stepSimulation()
        time.sleep(0.01)
    
    # Disconnect from the physics server
    p.disconnect()

if __name__ == "__main__":
    main()
