from horizon.problem import Problem
from horizon.rhc.model_description import FullModelInverseDynamics
from horizon.rhc.taskInterface import TaskInterface
from horizon.ros import replay_trajectory
from horizon.utils import utils

import phase_manager.pymanager as pymanager
import phase_manager.pyphase as pyphase

import std_msgs.msg
from xbot_interface import config_options as co
from xbot_interface import xbot_interface as xbot

import casadi_kin_dyn.py3casadi_kin_dyn as casadi_kin_dyn
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

import casadi as cs
import numpy as np
import rospy
import subprocess
import os
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
# from std_msgs.msg import Float64
import std_msgs
from std_srvs.srv import Empty, EmptyResponse
rospy.init_node('horizon_wbc_node')



# load urdf and srdf to create model
urdf = rospy.get_param('robot_description', default='')
if urdf == '':
    raise print('urdf not set')

srdf = rospy.get_param('robot_description_semantic', default='')
if srdf == '':
    raise print('urdf semantic not set')


ns = 0
with open('/home/wang/horizon_wbc/centauro_door.txt', 'r') as file:
    lines = file.readlines()
matrix = []
for line in lines:
    values = [float(x) for x in line.strip().split()]
    matrix.append(values)
    ns = ns + 1
ns = ns - 1
T = 5.
dt = T / ns

# each buff is 100 nodes
buff = 100
prb = Problem(buff, receding=True, casadi_type=cs.SX)
prb.setDt(dt)
# try construct RobotInterface
cfg = co.ConfigOptions()
cfg.set_urdf(urdf)
cfg.set_srdf(srdf)
cfg.generate_jidmap()
cfg.set_string_parameter('model_type', 'RBDL')
cfg.set_string_parameter('framework', 'ROS')
cfg.set_bool_parameter('is_model_floating_base', True)
# robot = xbot.RobotInterface(cfg)
# robot.sense()
# q_init = robot.getJointPositionMap()
q_init = {
    "ankle_pitch_1": -0.301666,
    "ankle_pitch_2": 0.301666,
    "ankle_pitch_3": 0.301667,
    "ankle_pitch_4": -0.30166,
    "ankle_yaw_1": 0.7070,
    "ankle_yaw_2": -0.7070,
    "ankle_yaw_3": -0.7070,
    "ankle_yaw_4": 0.7070,
    "d435_head_joint": 0,
    "hip_pitch_1": -1.25409,
    "hip_pitch_2": 1.25409,
    "hip_pitch_3": 1.25409,
    "hip_pitch_4": -1.25409,
    "hip_yaw_1": -0.746874,
    "hip_yaw_2": 0.746874,
    "hip_yaw_3": 0.746874,
    "hip_yaw_4": -0.746874,
    "j_arm1_1": 0.520149,
    "j_arm1_2": 0.320865,
    "j_arm1_3": 0.274669,
    "j_arm1_4": -2.23604,
    "j_arm1_5": 0.0500815,
    "j_arm1_6": -0.781461,
    "j_arm2_1": 0.520149,
    "j_arm2_2": -0.320865,
    "j_arm2_3": -0.274669,
    "j_arm2_4": -2.23604,
    "j_arm2_5": -0.0500815,
    "j_arm2_6": -0.781461,
    "knee_pitch_1": -1.55576,
    "knee_pitch_2": 1.55576,
    "knee_pitch_3": 1.55576,
    "knee_pitch_4": -1.55576,
    "torso_yaw": 3.56617e-13,
    "velodyne_joint": 0,
    "dagana_2_claw_joint": 0.
}

base_init = np.array([0, 0, 0.8, 0, 0, 0, 1])
urdf = urdf.replace('continuous', 'revolute')
fixed_joints_map = dict()
# fixed_joints_map.update({'j_wheel_1': 0., 'j_wheel_2': 0., 'j_wheel_3': 0., 'j_wheel_4': 0.})
kin_dyn  = casadi_kin_dyn.CasadiKinDyn(urdf)
model = FullModelInverseDynamics(problem=prb,
                                 kd=kin_dyn,
                                 q_init=q_init,
                                 base_init=base_init)
bashCommand = 'rosrun robot_state_publisher robot_state_publisher'
process = subprocess.Popen(bashCommand.split(), start_new_session=True)
ti = TaskInterface(prb=prb, model=model)
ti.setTaskFromYaml(os.getcwd() + '/centauro_wbc_config.yaml')
pm = pymanager.PhaseManager(buff+1)
c_phases = dict()
for c in model.getContactMap():
    c_phases[c] = pm.addTimeline(f'{c}_timeline')
    print("c = ", c)
stance_duration = 10
for c in model.getContactMap():
    stance_phase = pyphase.Phase(stance_duration, f'stance_{c}')
    if ti.getTask(f'{c}') is not None:
        stance_phase.addItem(ti.getTask(f'{c}'))
    else:
        raise Exception(f'Task {c} not found')
    
    c_phases[c].registerPhase(stance_phase)
for c in model.getContactMap():
    stance = c_phases[c].getRegisteredPhase(f'stance_{c}')
    while c_phases[c].getEmptyNodes() > 0:
        c_phases[c].addPhase(stance)
# abstract phase
matrix_np = np.array(matrix)
matrix_np_ = matrix_np
print("matrix_np_.shape = ", matrix_np_.shape) # matrix_np.shape =  (418, 23)
rows = matrix_np_.shape[0]
cols = matrix_np_.shape[1]
print("cols = ", cols)
num = rows // buff
# quotient, remainder = divmod(rows, buff)
# print("remainder", remainder)
matrix_np_arr = np.zeros((num, buff, cols)) # 4 x 100 x 23
print("matrix_np_arr.shape = ", matrix_np_arr.shape)
print("shape = ", matrix_np_arr[0,:,:].shape)
jjjj = np.zeros((buff, cols))
################################Programe#################################
reference = prb.createParameter('upper_body_reference', 23, nodes=range(buff+1))
# for i in range(21):
#     reference[i] = matrix[i][0]
#    x y z;4 quan; yaw_joint , 6 left arm, 6 right arm, 1 grippers + 2 headjoints = 7 + 15

prb.createResidual('upper_body_trajectory', 5 * (cs.vertcat(model.q[:7], model.q[-16:]) - reference))
reference.assign(matrix_np_arr[0,:,:].T)
exit()

#
# reference.assign(matrix 21 x 100)
model.q.setBounds(model.q0, model.q0, nodes=0)
# model.q[0].setBounds(model.q0[0] + 1, model.q0[0] + 1, nodes=buff)
model.v.setBounds(np.zeros(model.nv), np.zeros(model.nv), nodes=0)
model.v.setBounds(np.zeros(model.nv), np.zeros(model.nv), nodes=buff)
q_min = kin_dyn.q_min()
q_max = kin_dyn.q_max()
print(kin_dyn.joint_names())
print(q_min)
print(q_max)
prb.createResidual('lower_limits', 30 * utils.barrier(model.q[-3] - q_min[-3]))
prb.createResidual('upper_limits', 30 * utils.barrier1(model.q[-3] - q_max[-3]))
# prb.createResidual('support_polygon', wheel1 - whheel2 = fixed_disanace)
f0 = [0, 0, kin_dyn.mass() / 4 * 9.81]
for cname, cforces in model.getContactMap().items():
    for c in cforces:
        c.setInitialGuess(f0)
ti.finalize()
# ti.load_initial_guess()
for i in range(num):
    reference.assign(matrix_np_.T)
    ti.bootstrap()
    solution = ti.solution
    arr[i] = solution['q']

print("solution['q'].shape = ", solution['q'].shape)
arr = np.zeros((solution['q'].shape[0], solution['q'].shape[1]))
arr = solution['q']


rate = rospy.Rate(10) # 10hz
while not rospy.is_shutdown():
    repl = replay_trajectory.replay_trajectory(prb.getDt(), kin_dyn.joint_names(), arr, kindyn=kin_dyn, trajectory_markers=model.getContactMap().keys())
    repl.replay(is_floating_base=True)
exit()
time = 0
i = 0
rate = rospy.Rate(1./dt)
# robot = xbot.RobotInterface(cfg)

while time <= T:
    robot.setPositionReference(solution['q'][7:,i])
    robot.move() 
    time += dt
    i += 1
    rate.sleep()












