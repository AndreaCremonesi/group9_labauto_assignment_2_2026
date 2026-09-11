import yaml
import mujoco
import numpy as np
import plotly.graph_objects as go

from scipy.signal import find_peaks
from labauto import MuJoCoMechanicalSystem, TrapezoidalMotionLaw
from labauto import loadController, loadInstructions


model_name = "crane"
program_name = "identification_trj1"
omega_model = 2.92944


# Controller parameters
with open(f"{model_name}/control_config.yaml", "r") as file:
    params = yaml.safe_load(file)

controller_params = params["controller"]
dynamic_params = np.array(params["model_parameters"])


# Simulator
robot = MuJoCoMechanicalSystem(xml_path=f"{model_name}/model.xml")
robot.show()
robot.initialize()

dof = robot.get_input_number()
Tc = robot.get_sampling_period()


# Pendulum joint
joint_id = mujoco.mj_name2id(
    robot.model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "joint_2"
)

qpos_idx = int(robot.model.jnt_qposadr[joint_id])


# Controller
controller = loadController(
    Tc,
    controller_params,
    dynamic_params,
    model_name
)

controller.initialize()
controller.set_umax(robot.get_umax())


# Initial condition
measured_output = robot.read_sensor_value()

q0 = measured_output[:dof]
Dq0 = measured_output[dof:]
DDq0 = np.zeros(dof)

initial_reference = np.concatenate((q0, Dq0, DDq0))


# Motion law
ml = TrapezoidalMotionLaw({
    "max_velocity": np.array([5.5] * dof),
    "max_acceleration": np.array([5.0] * dof)
}, Tc)

ml.set_initial_condition(q0)
ml.add_instructions(
    loadInstructions(f"{model_name}/{program_name}.txt")
)


# Controller initialization
joint_torque = robot.read_actuator_value()
feedforward_action = np.zeros(dof)

controller.starting(
    initial_reference,
    measured_output,
    joint_torque,
    feedforward_action
)


# Simulation
t = []
pendulum_angle = []
actual_time = 0.0

while ml.depending_instructions():
    target_q, target_Dq, target_DDq = ml.compute_motion_law()

    reference = np.array([
        target_q[0],
        target_Dq[0],
        target_DDq[0]
    ])

    measured_output = robot.read_sensor_value()

    joint_torque = controller.compute_control_action(
        reference,
        measured_output,
        feedforward_action
    )

    robot.write_actuator_value(joint_torque)

    t.append(actual_time)
    pendulum_angle.append(
        float(robot.data.qpos[qpos_idx])
    )

    actual_time += Tc
    robot.simulate()

robot.close()


# Convert to arrays
t = np.array(t)
pendulum_angle = np.unwrap(
    np.array(pendulum_angle)
)


# Free-response analysis
start_time = 20.0
start_idx = np.searchsorted(t, start_time)

t_free = t[start_idx:]
theta_free = pendulum_angle[start_idx:]
theta_free -= np.mean(theta_free)


# Peak detection
min_distance = int(1.0 / Tc)

peaks, _ = find_peaks(
    theta_free,
    distance=min_distance,
    prominence=0.001
)

if len(peaks) < 2:
    raise RuntimeError(
        "Non sono stati trovati abbastanza picchi."
    )

peak_times = t_free[peaks]
peak_values = theta_free[peaks]


# Oscillation period
periods = np.diff(peak_times)
T_measured = np.mean(periods)

omega_d = 2 * np.pi / T_measured


# Damping ratio from logarithmic decrement
n = len(peaks) - 1

if (
    peak_values[0] > 0
    and peak_values[-1] > 0
    and peak_values[0] > peak_values[-1]
):
    delta = (
        np.log(
            peak_values[0] / peak_values[-1]
        ) / n
    )

    zeta = delta / np.sqrt(
        (2 * np.pi)**2 + delta**2
    )
else:
    zeta = 0.0


# Natural frequency
omega_n = omega_d / np.sqrt(
    1 - zeta**2
)

relative_error = (
    100
    * abs(omega_n - omega_model)
    / omega_model
)


# Results
print("\n--- FREQUENCY IDENTIFICATION ---")
print(f"Detected peaks: {len(peaks)}")
print(f"Measured periods: {periods}")
print(f"Mean period T: {T_measured:.6f} s")
print(f"Damped frequency omega_d: {omega_d:.6f} rad/s")
print(f"Estimated damping ratio zeta: {zeta:.6f}")
print(f"Identified natural frequency omega_n: {omega_n:.6f} rad/s")
print(f"Model natural frequency omega_n: {omega_model:.6f} rad/s")
print(f"Relative error: {relative_error:.3f} %")


# Plot
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=t_free,
    y=np.degrees(theta_free),
    name="Pendulum angle"
))

fig.add_trace(go.Scatter(
    x=peak_times,
    y=np.degrees(peak_values),
    mode="markers",
    name="Detected peaks"
))

fig.update_layout(
    title="Free pendulum oscillation - frequency identification",
    xaxis_title="Time [s]",
    yaxis_title="Pendulum angle [deg]"
)

fig.show()