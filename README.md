install dependencies from requirements.txt using
`pip install -r requirements.txt` 

RUN python3 fileName.


Detailed Explanation of Motion Planning Diffusion
1. Overview of Motion Planning Diffusion (MPD)
Motion Planning Diffusion (MPD) is a novel approach to robot motion planning that leverages diffusion models, a class of generative models that have shown remarkable success in image and audio generation. The key idea is to formulate motion planning as a generative modeling problem, where the goal is to generate smooth, collision-free trajectories from a noisy initial trajectory.

2. Key Components
2.1 B-spline Trajectory Representation
What: B-splines are used to represent trajectories with a small set of control points.

Why: This provides a compact, smooth representation that reduces the dimensionality of the planning problem.

How: The trajectory is defined by a set of control points and a B-spline basis function. The control points are what the diffusion process operates on.

2.2 Diffusion Process
Forward Process: Gradually adds noise to a trajectory until it becomes pure Gaussian noise.

Reverse Process: Gradually removes noise from a random trajectory to recover a valid motion plan.

Noise Schedule: Defined by β₁, β₂, ..., βₙ, which controls how quickly noise is added/removed.

2.3 Cost Function Guidance
Collision Avoidance: Penalizes trajectories that come close to obstacles.

Smoothness: Encourages smooth trajectories by penalizing high accelerations.

Boundary Constraints: Ensures the trajectory starts and ends at the specified positions.

Gradient-Based Guidance: The cost function gradient is used to guide the diffusion process toward valid trajectories.

3. Algorithm Steps
Initialization: Start with random Gaussian noise for the control points, fixing the start and end points.

Reverse Diffusion: Iteratively denoise the trajectory:

Predict and remove noise using the diffusion model.

Apply cost function guidance to steer toward valid trajectories.

Ensure start and end points remain fixed.

Trajectory Generation: Convert the final control points to a smooth trajectory using B-spline interpolation.

Execution: Execute the trajectory on the robot.

4. Visualization Enhancements
The visualization version of the code adds:

Color-Coded Trajectories: Shows the evolution of the trajectory with a color gradient from red (early steps) to green (final steps).

Control Point Visualization: Displays the control points as magenta dots to show how they evolve.

Step-by-Step Visualization: Shows intermediate trajectories at specified intervals.

Fading Effect: Creates a visual history of the planning process.

5. Mathematical Foundation
The diffusion process is based on the following equations:

Forward Process: q(x_t | x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)

Reverse Process: p(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))

Where:

μ_θ is the predicted mean (implemented in _p_sample)

Σ_θ is the predicted variance

Cost guidance modifies μ_θ by subtracting the cost gradient

6. Key Innovations in the Implementation
Cost-Guided Sampling: The diffusion process is guided by a cost function gradient, which helps steer the trajectory away from obstacles and toward smoother paths.

B-spline Parameterization: Using B-splines allows for a compact representation and ensures smooth trajectories.

Numerical Stability: The implementation includes safeguards against NaNs and numerical instability.

Visualization: The enhanced version provides a clear visualization of how the diffusion process gradually transforms noise into a valid trajectory.

7. References and Sources
The original MPD paper: "Motion Planning Diffusion: Learning and Planning of Robot Motions with Diffusion Models" (https://arxiv.org/pdf/2412.19948)

Denoising Diffusion Probabilistic Models (DDPM): https://arxiv.org/abs/2006.11239

PyBullet documentation: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA

B-spline theory: De Boor, C. (1978). A practical guide to splines. Springer-Verlag.

This implementation demonstrates how diffusion models can be adapted for motion planning, providing a powerful new approach that combines the strengths of generative modeling with traditional cost-based planning.