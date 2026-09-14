import time
import yaml
import mujoco

from scipy.io import savemat
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from labauto import MuJoCoMechanicalSystem
from labauto import TrapezoidalMotionLaw
from labauto import loadController
from labauto import loadInstructions

from input_shaper import InputShaper


model_name = "crane"
program_name = "test_trj1"


# Load controller parameters and dynamic parameters

with open(f'{model_name}/control_config.yaml', 'r') as file:
    params_yaml = yaml.safe_load(file)

controller_params = params_yaml['controller']
dynamic_params = np.array(params_yaml['model_parameters'])


# Create simulator

xml_path = f"{model_name}/model.xml"

robot = MuJoCoMechanicalSystem(
    xml_path=xml_path,
    motor_actuators=["motor_1"],
    motor_joints=["joint_1"],
    spring_joints=[],
    ee_site="payload"
)

robot = MuJoCoMechanicalSystem(xml_path=xml_path)
robot.show()
robot.initialize()

dof = robot.get_input_number()


# Pendulum joint indices

pendulum_joint_id = mujoco.mj_name2id(
    robot.model, mujoco.mjtObj.mjOBJ_JOINT, "joint_2"
)

pendulum_qpos_index = int(robot.model.jnt_qposadr[pendulum_joint_id])
pendulum_qvel_index = int(robot.model.jnt_dofadr[pendulum_joint_id])


# Sampling time

Tc = robot.get_sampling_period()


# Input shaper configuration

use_input_shaper = True
shaper_type = "ZVDD"

natural_frequency = 2.92944 
damping_ratio = 0.0
residual_vibration = 0.05

input_shaper = InputShaper( 
    sampling_time=Tc,
    natural_frequency=natural_frequency,
    damping_ratio=damping_ratio,
    shaper_type=shaper_type,
    residual_vibration=residual_vibration
)

print(f"Control sampling time Tc = {Tc} s")
print(f"MuJoCo internal timestep = {robot.model.opt.timestep} s")
print(f"Input shaper enabled: {use_input_shaper}")
print(f"Input shaper type: {shaper_type}")
print(f"Shaper weights: {input_shaper.weights}")
print(f"Shaper delays: {input_shaper.delay_samples} samples")

if shaper_type == "EI":
    print(f"EI residual-vibration tolerance: {100 * residual_vibration:.1f}%")

print(
    f"Pendulum joint: id={pendulum_joint_id}, "
    f"qpos index={pendulum_qpos_index}, "
    f"qvel index={pendulum_qvel_index}"
)


# Load controller

decentralized_ctrl = loadController(
    Tc, controller_params, dynamic_params, model_name
)

decentralized_ctrl.initialize()
decentralized_ctrl.set_umax(robot.get_umax())


# Initial reference

measured_output = robot.read_sensor_value()

q0 = measured_output[:dof]
Dq0 = measured_output[dof:]
DDq0 = np.zeros(dof)

initial_reference = np.concatenate((q0, Dq0, DDq0))


# Motion law

max_Dq = np.array([5.5] * dof)
max_DDq = np.array([5.0] * dof)

motion_law_params = {
    'max_velocity': max_Dq,
    'max_acceleration': max_DDq
}

ml = TrapezoidalMotionLaw(motion_law_params, Tc)
ml.set_initial_condition(q0)

instructions = loadInstructions(f'{model_name}/{program_name}.txt')
ml.add_instructions(instructions)


# Initial actuator force

joint_torque = robot.read_actuator_value()
feedforward_action = np.array([0.0] * dof)

print(
    f"joint_torque={joint_torque}, "
    f"initial_reference={initial_reference}, "
    f"measured_output={measured_output}"
)

decentralized_ctrl.starting(
    initial_reference,
    measured_output,
    joint_torque,
    feedforward_action
)


# Simulation loop

t = []
measured_signal = []
control_action = []
reference_signal = []
raw_reference_signal = []
link_position = []
pendulum_angle = []
pendulum_angular_velocity = []

actual_time = 0.0

while ml.depending_instructions():
    loop_t0 = time.perf_counter()

    target_q, target_Dq, target_DDq = ml.compute_motion_law()

    raw_reference = np.array([
        target_q[0],
        target_Dq[0],
        target_DDq[0]
    ])

    if use_input_shaper:
        reference = input_shaper.shape(raw_reference)
    else:
        reference = raw_reference

    measured_output = robot.read_sensor_value()

    joint_torque = decentralized_ctrl.compute_control_action(
        reference,
        measured_output,
        feedforward_action
    )

    robot.write_actuator_value(joint_torque)

    # Store data before stepping
    t.append(actual_time)
    measured_signal.append(measured_output)
    control_action.append(joint_torque)
    reference_signal.append(reference.copy())
    raw_reference_signal.append(raw_reference.copy())
    link_position.append(robot.link_position())

    pendulum_angle.append(float(robot.data.qpos[pendulum_qpos_index]))
    pendulum_angular_velocity.append(float(robot.data.qvel[pendulum_qvel_index]))

    actual_time += Tc

    robot.simulate()

    computation_time = time.perf_counter() - loop_t0
    time.sleep(max(0.0, Tc - computation_time))


# Convert to arrays

t = np.array(t)
measured_signal = np.array(measured_signal)
control_action = np.array(control_action)
reference_signal = np.array(reference_signal)
raw_reference_signal = np.array(raw_reference_signal)
link_position = np.array(link_position)
pendulum_angle = np.array(pendulum_angle)
pendulum_angular_velocity = np.array(pendulum_angular_velocity)


# Post-processing

joint_position = measured_signal[:, :dof]
joint_velocity = measured_signal[:, dof:]

reference_position = reference_signal[:, :dof]
reference_velocity = reference_signal[:, dof:2*dof]
reference_acceleration = reference_signal[:, 2*dof:]

raw_reference_position = raw_reference_signal[:, :dof]
raw_reference_velocity = raw_reference_signal[:, dof:2*dof]
raw_reference_acceleration = raw_reference_signal[:, 2*dof:]

position_error = reference_position - joint_position
velocity_error = reference_velocity - joint_velocity


# Global tracking metrics

position_tracking_mae = np.mean(np.abs(position_error[:, 0]))
maximum_position_tracking_error = np.max(np.abs(position_error[:, 0]))
maximum_control_action = np.max(np.abs(control_action[:, 0]))


# Pendulum metrics

initial_pendulum_angle = pendulum_angle[0]
pendulum_displacement = pendulum_angle - initial_pendulum_angle

maximum_accumulated_turns = (
    np.max(np.abs(pendulum_displacement)) / (2 * np.pi)
)

complete_rotation_detected = np.any(
    np.abs(pendulum_displacement) >= 2 * np.pi
)

wrapped_pendulum_angle = np.arctan2(
    np.sin(pendulum_angle),
    np.cos(pendulum_angle)
)

maximum_wrapped_angle = np.max(np.abs(wrapped_pendulum_angle))
maximum_angular_velocity = np.max(np.abs(pendulum_angular_velocity))


print(f"Position tracking MAE: {position_tracking_mae:.6f} m")
print(f"Maximum position tracking error: {maximum_position_tracking_error:.6f} m")
print(f"Maximum control action: {maximum_control_action:.6f} N")

print(f"Maximum accumulated pendulum displacement: {maximum_accumulated_turns:.3f} turns")
print(f"Complete rotation detected: {complete_rotation_detected}")

print(
    f"Maximum wrapped pendulum angle: "
    f"{maximum_wrapped_angle:.6f} rad "
    f"({np.degrees(maximum_wrapped_angle):.3f} deg)"
)

print(
    f"Maximum pendulum angular velocity: "
    f"{maximum_angular_velocity:.6f} rad/s "
    f"({np.degrees(maximum_angular_velocity):.3f} deg/s)"
)


# Residual oscillation during pauses
# Final 0.4 s are used because the longest ZVDD delay is close to 4 s.

pause_analysis_duration = 0.4

stationary = np.abs(raw_reference_velocity[:, 0]) < 1e-6

changes = np.diff(
    np.concatenate(([False], stationary, [False])).astype(int)
)

pause_starts = np.where(changes == 1)[0]
pause_ends = np.where(changes == -1)[0]

print("\nResidual oscillation during the final 0.4 s of each pause:")

pause_number = 0
analysis_samples = int(pause_analysis_duration / Tc)

for start, end in zip(pause_starts, pause_ends):
    pause_duration = (end - start) * Tc

    # Ignore initial short pause
    if pause_duration < 2.0:
        continue

    pause_number += 1
    analysis_start = max(start, end - analysis_samples)

    target_position = np.mean(
        raw_reference_position[analysis_start:end, 0]
    )

    pause_position_error = (
        raw_reference_position[analysis_start:end, 0]
        - joint_position[analysis_start:end, 0]
    )

    pause_angles = wrapped_pendulum_angle[analysis_start:end]
    pause_velocity = pendulum_angular_velocity[analysis_start:end]

    pause_position_mae = np.mean(np.abs(pause_position_error))
    pause_angle_max = np.max(np.abs(pause_angles))
    pause_angle_rms = np.sqrt(np.mean(pause_angles**2))
    pause_velocity_max = np.max(np.abs(pause_velocity))

    print(
        f"Pause {pause_number}: "
        f"target={target_position:.1f} m, "
        f"position MAE={pause_position_mae:.6f} m, "
        f"angle max={np.degrees(pause_angle_max):.3f} deg, "
        f"angle RMS={np.degrees(pause_angle_rms):.3f} deg, "
        f"angular velocity max={np.degrees(pause_velocity_max):.3f} deg/s"
    )


# Save test data

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

test_data = {
    "reference_position": reference_position,
    "reference_velocity": reference_velocity,
    "reference_acceleration": reference_acceleration,
    "raw_reference_position": raw_reference_position,
    "raw_reference_velocity": raw_reference_velocity,
    "raw_reference_acceleration": raw_reference_acceleration,
    "joint_position": joint_position,
    "joint_velocity": joint_velocity,
    "joint_torque": control_action,
    "link_position": link_position,
    "pendulum_angle": pendulum_angle,
    "pendulum_angular_velocity": pendulum_angular_velocity,
    "time": t,
    "name": "test"
}

savemat(
    f"{model_name}/tests/{program_name}_{timestamp}.mat",
    test_data
)


LINE_WIDTH = 4        
REF_LINE_WIDTH = 3   
FONT_SIZE = 18
TITLE_FONT_SIZE = 22

labels = ["x"]

fig1 = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    subplot_titles=(
        [f"Posizione {a}" for a in labels]
        + [f"Velocità {a}" for a in labels]
        + [f"Forza attuatore {a} (lato motore)" for a in labels]
    ),
    vertical_spacing=0.12
)

for i, a in enumerate(labels):
    col = i + 1

  
    fig1.add_trace(
        go.Scatter(
            x=t, y=joint_position[:, i],
            name="Valore attuale",
            legendgroup=f"pos_{a}",
            line=dict(color="blue", width=LINE_WIDTH)
        ),
        row=1, col=col
    )
    fig1.add_trace(
        go.Scatter(
            x=t, y=reference_position[:, i],
            name="Riferimento",
            legendgroup=f"pos_{a}",
            line=dict(color="red", width=REF_LINE_WIDTH, dash="dash")
        ),
        row=1, col=col
    )

    
    fig1.add_trace(
        go.Scatter(
            x=t, y=joint_velocity[:, i],
            name="Valore attuale",
            legendgroup=f"pos_{a}",
            showlegend=False,
            line=dict(color="blue", width=LINE_WIDTH)
        ),
        row=2, col=col
    )
    fig1.add_trace(
        go.Scatter(
            x=t, y=reference_velocity[:, i],
            name="Riferimento",
            legendgroup=f"pos_{a}",
            showlegend=False,
            line=dict(color="red", width=REF_LINE_WIDTH, dash="dash")
        ),
        row=2, col=col
    )


    fig1.add_trace(
        go.Scatter(
            x=t, y=control_action[:, i],
            name="Forza attuatore",
            legendgroup=f"u_{a}",
            line=dict(color="orange", width=LINE_WIDTH)
        ),
        row=3, col=col
    )

fig1.update_xaxes(title_text="Tempo (s)", row=3, col=1, title_font=dict(size=FONT_SIZE))
fig1.update_xaxes(showgrid=True, tickfont=dict(size=FONT_SIZE - 2))
fig1.update_yaxes(showgrid=True, tickfont=dict(size=FONT_SIZE - 2))

fig1.update_layout(
    title=dict(text="Inseguimento: posizione / velocità / forza", font=dict(size=TITLE_FONT_SIZE)),
    height=1200,
    width=1600,
    margin=dict(t=160),
    font=dict(size=FONT_SIZE),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
        font=dict(size=FONT_SIZE)
    )
)

fig1.update_annotations(font_size=TITLE_FONT_SIZE - 2)


mae_pos = np.mean(np.abs(position_error), axis=0)
mae_vel = np.mean(np.abs(velocity_error), axis=0)

subplot_titles = (
    [f"Errore di posizione {labels[i]} (MAE={mae_pos[i]:4.3f})" for i in range(dof)]
    + [f"Errore di velocità {labels[i]} (MAE={mae_vel[i]:4.3f})" for i in range(dof)]
)

fig2 = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    subplot_titles=subplot_titles,
    vertical_spacing=0.18
)

for i, a in enumerate(labels):
    col = i + 1

    fig2.add_trace(
        go.Scatter(
            x=t, y=position_error[:, i],
            name="Errore di posizione",
            legendgroup=f"ep_{a}",
            line=dict(color="blue", width=LINE_WIDTH)
        ),
        row=1, col=col
    )

    fig2.add_trace(
        go.Scatter(
            x=t, y=velocity_error[:, i],
            name="Errore di velocità",
            legendgroup=f"ev_{a}",
            line=dict(color="red", width=LINE_WIDTH)
        ),
        row=2, col=col
    )

fig2.update_xaxes(title_text="Tempo (s)", row=2, col=1, title_font=dict(size=FONT_SIZE))
fig2.update_xaxes(showgrid=True, tickfont=dict(size=FONT_SIZE - 2))
fig2.update_yaxes(showgrid=True, tickfont=dict(size=FONT_SIZE - 2))

fig2.update_layout(
    title=dict(text="Errori: posizione e velocità", font=dict(size=TITLE_FONT_SIZE)),
    height=1200,
    width=1600,
    margin=dict(t=140),
    font=dict(size=FONT_SIZE),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
        font=dict(size=FONT_SIZE)
    )
)

fig2.update_annotations(font_size=TITLE_FONT_SIZE - 2)

fig1.show()
fig2.show()

robot.close()