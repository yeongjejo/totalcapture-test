import time

import pandas as pd
import socket, struct, time
import json
import numpy as np
import math
from scipy.spatial.transform import Rotation as R


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)

def quat_inverse(q):
    w, x, y, z = q
    norm2 = np.dot(q,q)
    return np.array([w, -x, -y, -z], dtype=float) / norm2

def load_joint_data(file_path, data_form):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 첫 줄: 관절 이름
    headers = lines[0].strip().split('\t')

    # 나머지 줄: 각 줄은 값들이 \t 구분되어 있음
    data = []
    for line in lines[1:]:
        values = line.strip().split('\t')
        # 각 관절마다 4개의 값 (x,y,z,w)
        frame = []
        for v in values:
            nums = v.split()
            frame.append([float(n) for n in nums])
        data.append(frame)

    # DataFrame으로 변환
    # 멀티컬럼 구조: (Joint, Component)
    # cols = pd.MultiIndex.from_product([headers, ['x', 'y', 'z', 'w']], names=['Joint', 'Component'])
    cols = pd.MultiIndex.from_product([headers, data_form], names=['Joint', 'Component'])
    df = pd.DataFrame([sum(frame, []) for frame in data], columns=cols)

    return df


# 사용 예시
pose = load_joint_data("gt_skel_gbl_pos.txt", ['x', 'y', 'z'])
ori = load_joint_data("gt_skel_gbl_ori.txt", ['x', 'y', 'z', 'w'])
# ori = load_joint_data("gt_skel_gbl_ori.txt", ['w', 'x', 'y', 'z'])
# ori = load_joint_data("gt_skel_gbl_ori.txt", ['y', 'z', 'x', 'w'])
# print(df)

TARGET_IP = "127.0.0.1"
TARGET_PORT = 5005

while True:

    first_q = []
    f_root_postion = []
    for (idx, pose_row), (_, ori_row) in zip(pose.iterrows(), ori.iterrows()):
        time.sleep(0.016)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        bone_seq = [
            [pose_row.Hips, ori_row.Hips],
            [pose_row.Spine3, ori_row.Spine3],
            [pose_row.Head, ori_row.Head],
            [pose_row.LeftArm, ori_row.LeftArm],
            [pose_row.LeftForeArm, ori_row.LeftForeArm],
            [pose_row.LeftHand, ori_row.LeftHand],
            [pose_row.RightArm, ori_row.RightArm],
            [pose_row.RightForeArm, ori_row.RightForeArm],
            [pose_row.RightHand, ori_row.RightHand],
            [pose_row.LeftUpLeg, ori_row.LeftUpLeg],
            [pose_row.LeftLeg, ori_row.LeftLeg],
            [pose_row.LeftFoot, ori_row.LeftFoot],
            [pose_row.RightUpLeg, ori_row.RightUpLeg],
            [pose_row.RightLeg, ori_row.RightLeg],
            [pose_row.RightFoot, ori_row.RightFoot]
        ]

        sned_data = []
        for i, bone in enumerate(bone_seq):
            frame_pose = []
            q = np.array([bone[1].w, -bone[1].x, bone[1].y, -bone[1].z])
            if idx == 0:
                first_q.append(quat_inverse(q))
                if i == 0:
                    f_root_postion = [-bone[0].x / 3.0, bone[0].y / 3.0, -bone[0].z / 3.0]

            frame_pose = [(-bone[0].x / 3.0) - f_root_postion[0], (bone[0].y / 3.0) - f_root_postion[1] + 11.0, (-bone[0].z / 3.0) - f_root_postion[2]]

            frame_bone_data = {
                "time": "1",
                "name": "test",
                # "position": [0.0, 0.0, 0.0],
                "position": frame_pose,
                "rotation": quat_mul(q, first_q[i]).tolist(),
                # "rotation": [bone[1].w, -bone[1].x, -bone[1].z, bone[1].y],
                "acc": [0.0, 0.0, 0.0]
            }
            sned_data.append(frame_bone_data)


        data = json.dumps(sned_data).encode("utf-8")
        sock.sendto(data, (TARGET_IP, TARGET_PORT))
        # break


        # print(idx, row.keys())