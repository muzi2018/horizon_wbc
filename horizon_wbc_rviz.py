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
rospy.init_node('horizon_wbc_node_rviz')



# load urdf and srdf to create model
urdf = rospy.get_param('robot_description', default='')
if urdf == '':
    raise print('urdf not set')

srdf = rospy.get_param('robot_description_semantic', default='')
if srdf == '':
    raise print('urdf semantic not set')


ns = 0
with open('/home/wang/horizon_wbc/output.txt', 'r') as file:
    lines = file.readlines()
matrix = []
for line in lines:
    values = [float(x) for x in line.strip().split()]
    matrix.append(values)
    ns = ns + 1
ns = ns - 1
T = 5.
dt = T / ns


prb = Problem(ns, receding=True, casadi_type=cs.SX)
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
    "ankle_yaw_1": 0.,
    "ankle_yaw_2": -0.,
    "ankle_yaw_3": -0.,
    "ankle_yaw_4": 0.,
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

kin_dyn  = casadi_kin_dyn.CasadiKinDyn(urdf, fixed_joints=fixed_joints_map)

model = FullModelInverseDynamics(problem=prb,
                                 kd=kin_dyn,
                                 q_init=q_init,
                                 base_init=base_init,
                                 fixed_joint_map=fixed_joints_map)

bashCommand = 'rosrun robot_state_publisher robot_state_publisher'
process = subprocess.Popen(bashCommand.split(), start_new_session=True)

ti = TaskInterface(prb=prb, model=model)
ti.setTaskFromYaml(os.getcwd() + '/centauro_wbc_config.yaml')

pm = pymanager.PhaseManager(ns+1)

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




matrix_np = np.array(matrix)
matrix_np_ = matrix_np
# matrix_np_ = np.zeros((matrix_np.shape[0], matrix_np.shape[1] + 1))
# matrix_np_[:, 0:3] = matrix_np[:, 0:3]


# for i in range(matrix_np.shape[0]):
#     ori_vector = matrix_np[i, 3:6].flatten()
#     r = R.from_rotvec(ori_vector)
#     quat = r.as_quat()
#     matrix_np_[i, 3:7] = quat
# matrix_np_[:, 8:] = matrix_np[:, 7:]


print("matrix_np.shape = ", matrix_np.shape)
print("matrix_np_.shape = ", matrix_np_.shape)

# exit()


reference = prb.createParameter('upper_body_reference', 23, nodes=range(ns+1))
# for i in range(21):
#     reference[i] = matrix[i][0]
#    x y z;4 quan; yaw_joint , 6 left arm, 6 right arm, 1 grippers + 2 headjoints = 7 + 15

prb.createResidual('upper_body_trajectory', 1 * (cs.vertcat(model.q[:7], model.q[-16:]) - reference))
# exit()

reference.assign(matrix_np_.T)
print (reference.shape)

#
# reference.assign(matrix 21 x 100)

model.q.setBounds(model.q0, model.q0, nodes=0)
# model.q.setBounds(model.q0, model.q0, nodes=ns)
model.v.setBounds(np.zeros(model.nv), np.zeros(model.nv), nodes=0)
model.v.setBounds(np.zeros(model.nv), np.zeros(model.nv), nodes=ns)

q_min = kin_dyn.q_min()
q_max = kin_dyn.q_max()
print(kin_dyn.joint_names())
print(q_min)
print(q_max)
prb.createResidual('lower_limits', 30 * utils.barrier(model.q - q_min))
prb.createResidual('upper_limits', 30 * utils.barrier1(model.q - q_max))

# prb.createResidual('support_polygon', wheel1 - whheel2 = fixed_disanace)
f0 = [0, 0, kin_dyn.mass() / 4 * 9.81]
for cname, cforces in model.getContactMap().items():
    for c in cforces:
        c.setInitialGuess(f0)

ti.finalize()
# ti.load_initial_guess()
ti.bootstrap()




solution = ti.solution
print("solution['q'].shape[1] = ", solution['q'].shape)
## publish plot data


# publish solution server

publish_bool = False
def start_plot(req):
    global publish_bool
    publish_bool = not publish_bool
    return EmptyResponse()
service = rospy.Service('plot', Empty, start_plot)

# rospy.Service('service_name', ServiceType, callback_function).


# x y z;4 quan; yaw_joint , 6 left arm, 6 right arm, 1 grippers + 2 headjoints = 7 + 15
## publish reference
pub_solx = rospy.Publisher('pub_solx', std_msgs.msg.Float64, queue_size=1)
pub_soly = rospy.Publisher('pub_soly', std_msgs.msg.Float64, queue_size=1)
pub_solz = rospy.Publisher('pub_solz', std_msgs.msg.Float64, queue_size=1)
pub_solqan0 = rospy.Publisher('pub_solqan0', std_msgs.msg.Float64, queue_size=1)
pub_solqan1 = rospy.Publisher('pub_solqan1', std_msgs.msg.Float64, queue_size=1)
pub_solqan2 = rospy.Publisher('pub_solqan2', std_msgs.msg.Float64, queue_size=1)
pub_solqan3 = rospy.Publisher('pub_solqan3', std_msgs.msg.Float64, queue_size=1)
pub_yaw_joint = rospy.Publisher('pub_yaw_joint', std_msgs.msg.Float64, queue_size=1)
pub_solleftarm1 = rospy.Publisher('pub_solleftarm1', std_msgs.msg.Float64, queue_size=1)
pub_solleftarm2 = rospy.Publisher('pub_solleftarm2', std_msgs.msg.Float64, queue_size=1)
pub_solleftarm3 = rospy.Publisher('pub_solleftarm3', std_msgs.msg.Float64, queue_size=1)
pub_solleftarm4 = rospy.Publisher('pub_solleftarm4', std_msgs.msg.Float64, queue_size=1)
pub_solleftarm5 = rospy.Publisher('pub_solleftarm5', std_msgs.msg.Float64, queue_size=1)
pub_solleftarm6 = rospy.Publisher('pub_solleftarm6', std_msgs.msg.Float64, queue_size=1)


pub_solrightarm1 = rospy.Publisher('pub_solrightarm1', std_msgs.msg.Float64, queue_size=1)
pub_solrightarm2 = rospy.Publisher('pub_solrightarm2', std_msgs.msg.Float64, queue_size=1)
pub_solrightarm3 = rospy.Publisher('pub_solrightarm3', std_msgs.msg.Float64, queue_size=1)
pub_solrightarm4 = rospy.Publisher('pub_solrightarm4', std_msgs.msg.Float64, queue_size=1)
pub_solrightarm5 = rospy.Publisher('pub_solrightarm5', std_msgs.msg.Float64, queue_size=1)
pub_solrightarm6 = rospy.Publisher('pub_solrightarm6', std_msgs.msg.Float64, queue_size=1)
pub_solgripper = rospy.Publisher('pub_solgripper', std_msgs.msg.Float64, queue_size=1)

## publish ref
pub_x_ref = rospy.Publisher('pub_x_ref', std_msgs.msg.Float64, queue_size=1)
pub_y_ref = rospy.Publisher('pub_y_ref', std_msgs.msg.Float64, queue_size=1)
pub_z_ref = rospy.Publisher('pub_z_ref', std_msgs.msg.Float64, queue_size=1)
pub_qan0_ref = rospy.Publisher('pub_qan0_ref', std_msgs.msg.Float64, queue_size=1)
pub_qan1_ref = rospy.Publisher('pub_qan1_ref', std_msgs.msg.Float64, queue_size=1)
pub_qan2_ref = rospy.Publisher('pub_qan2_ref', std_msgs.msg.Float64, queue_size=1)
pub_qan3_ref = rospy.Publisher('pub_qan3_ref', std_msgs.msg.Float64, queue_size=1)
pub_yaw_joint_ref = rospy.Publisher('pub_yaw_joint_ref', std_msgs.msg.Float64, queue_size=1)
pub_leftarm1_ref = rospy.Publisher('pub_leftarm1_ref', std_msgs.msg.Float64, queue_size=1)
pub_leftarm2_ref = rospy.Publisher('pub_leftarm2_ref', std_msgs.msg.Float64, queue_size=1)
pub_leftarm3_ref = rospy.Publisher('pub_leftarm3_ref', std_msgs.msg.Float64, queue_size=1)
pub_leftarm4_ref = rospy.Publisher('pub_leftarm4_ref', std_msgs.msg.Float64, queue_size=1)
pub_leftarm5_ref = rospy.Publisher('pub_leftarm5_ref', std_msgs.msg.Float64, queue_size=1)
pub_leftarm6_ref = rospy.Publisher('pub_leftarm6_ref', std_msgs.msg.Float64, queue_size=1)


pub_rightarm1_ref = rospy.Publisher('pub_rightarm1_ref', std_msgs.msg.Float64, queue_size=1)
pub_rightarm2_ref = rospy.Publisher('pub_rightarm2_ref', std_msgs.msg.Float64, queue_size=1)
pub_rightarm3_ref = rospy.Publisher('pub_rightarm3_ref', std_msgs.msg.Float64, queue_size=1)
pub_rightarm4_ref = rospy.Publisher('pub_rightarm4_ref', std_msgs.msg.Float64, queue_size=1)
pub_rightarm5_ref = rospy.Publisher('pub_rightarm5_ref', std_msgs.msg.Float64, queue_size=1)
pub_rightarm6_ref = rospy.Publisher('pub_rightarm6_ref', std_msgs.msg.Float64, queue_size=1)
pub_gripper_ref = rospy.Publisher('pub_gripper_ref', std_msgs.msg.Float64, queue_size=1)



rate = rospy.Rate(10) # 10hz
while not rospy.is_shutdown():
    # repl = replay_trajectory.replay_trajectory(prb.getDt(), kin_dyn.joint_names(), solution['q'], kindyn=kin_dyn, trajectory_markers=model.getContactMap().keys())
    # repl.replay(is_floating_base=True)
    # print("publish_bool = ", publish_bool)
    while not publish_bool:
        rate.sleep()    
    msg = std_msgs.msg.Float64()
    for i in range(solution['q'].shape[1]):
        # print("solution['q'][1,i] = ", solution['q'][1,i])
        msg.data = solution['q'][0,i]
        pub_solx.publish(msg)
        msg.data = solution['q'][1,i]
        pub_soly.publish(msg)
        msg.data = solution['q'][2,i]
        pub_solz.publish(msg)
        msg.data = solution['q'][3,i]
        pub_solqan0.publish(msg)
        msg.data = solution['q'][4,i]
        pub_solqan1.publish(msg)
        msg.data = solution['q'][5,i]
        pub_solqan2.publish(msg)
        msg.data = solution['q'][6,i]
        pub_solqan3.publish(msg)
        msg.data = solution['q'][7,i]
        pub_yaw_joint.publish(msg)
        msg.data = solution['q'][8,i]
        pub_solleftarm1.publish(msg)
        msg.data = solution['q'][9,i]
        pub_solleftarm2.publish(msg)
        msg.data = solution['q'][10,i]
        pub_solleftarm3.publish(msg)
        msg.data = solution['q'][11,i]
        pub_solleftarm4.publish(msg)
        msg.data = solution['q'][12,i]
        pub_solleftarm5.publish(msg)
        msg.data = solution['q'][13,i]
        pub_solleftarm6.publish(msg)
        msg.data = solution['q'][14,i]
        pub_solrightarm1.publish(msg)
        msg.data = solution['q'][15,i] 
        pub_solrightarm2.publish(msg)
        msg.data = solution['q'][16,i] 
        pub_solrightarm3.publish(msg)
        msg.data = solution['q'][17,i] 
        pub_solrightarm4.publish(msg)
        msg.data = solution['q'][18,i] 
        pub_solrightarm5.publish(msg)
        msg.data = solution['q'][19,i] 
        pub_solrightarm6.publish(msg)
        msg.data = solution['q'][20,i] 
        pub_solgripper.publish(msg)


        # print("matrix_np_.shape = ", matrix_np_.shape)
        # exit()
        # print("i: ", i)
        msg.data = matrix_np_[i,0]
        pub_x_ref.publish(msg)
        msg.data = matrix_np_[i,1]
        pub_y_ref.publish(msg)
        msg.data = matrix_np_[i,2]
        pub_z_ref.publish(msg)
        msg.data = matrix_np_[i,3]
        pub_qan0_ref.publish(msg)
        msg.data = matrix_np_[i,4]
        pub_qan1_ref.publish(msg)
        msg.data = matrix_np_[i,5]
        pub_qan2_ref.publish(msg)
        msg.data = matrix_np_[i,6]
        pub_qan3_ref.publish(msg)
        msg.data = matrix_np_[i,7]
        pub_yaw_joint_ref.publish(msg)
        msg.data = matrix_np_[i,8]
        pub_leftarm1_ref.publish(msg)
        msg.data = matrix_np_[i,9]
        pub_leftarm2_ref.publish(msg)
        msg.data = matrix_np_[i,10]
        pub_leftarm3_ref.publish(msg)
        msg.data = matrix_np_[i,11]
        pub_leftarm4_ref.publish(msg)
        msg.data = matrix_np_[i,12]
        pub_leftarm5_ref.publish(msg)
        msg.data = matrix_np_[i,13]
        pub_leftarm6_ref.publish(msg)
        msg.data = matrix_np_[i,14]
        pub_rightarm1_ref.publish(msg)
        msg.data = matrix_np_[i,15] 
        pub_rightarm2_ref.publish(msg)
        msg.data = matrix_np_[i,16] 
        pub_rightarm3_ref.publish(msg)
        msg.data = matrix_np_[i,17] 
        pub_rightarm4_ref.publish(msg)
        msg.data = matrix_np_[i,18] 
        pub_rightarm5_ref.publish(msg)
        msg.data = matrix_np_[i,19] 
        pub_rightarm6_ref.publish(msg)
        msg.data = matrix_np_[i,20] 
        pub_gripper_ref.publish(msg)



        rate.sleep()
exit()



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












