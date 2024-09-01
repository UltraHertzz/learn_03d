import torch
import open3d as o3d
import numpy as np


class Camera:

    def __init__(self, frame, config) -> None:
        rgb_image_file = frame[0]
        depth_image_file = frame[1]
        self.gt_pose = frame[2]
        self.rgb_image = o3d.io.read_image(rgb_image_file)
        self.depth_image = o3d.io.read_image(depth_image_file)
        if config["Dataset"]["Calibration"]["distorted"] == False:
            self.intrinsics = o3d.camera.PinholeCameraIntrinsic(
                width=config["Dataset"]["Calibration"]["width"],
                height=config["Dataset"]["Calibration"]["height"],
                fx=config["Dataset"]["Calibration"]["fx"],
                fy=config["Dataset"]["Calibration"]["fy"],
                cx=config["Dataset"]["Calibration"]["cx"],
                cy=config["Dataset"]["Calibration"]["cy"]
            )
        self.extrinsic = self.build_extrinsic_matrix()

    @staticmethod        
    def create_camera_intrinsics(fx, fy, cx, cy):
        # 创建相机内参矩阵
        K = np.array([
            [fx, 0,  cx],
            [0,  fy, cy],
            [0,  0,  1]
        ])
        return K

    def getWorld2View2(R, t, translate=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
        translate = translate.to(R.device)
        Rt = torch.zeros((4, 4), device=R.device)
        # Rt[:3, :3] = R.transpose()
        Rt[:3, :3] = R
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0

        # 将inv操作用物理含义的方法替代，加快速度
        # C2W = torch.linalg.inv(Rt) # test with simple matrix on RTX2060 with 0.2824 s execution time
        C2W = torch.zeros((4,4), device=R.device) # test on RTX2060 with 0.00166 s execution time
        C2W[:3, :3] = R.T
        C2W[:3, 3] = -1*R.T@t
        C2W[3, 3] = 1.0   

        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        C2W[:3, 3] = cam_center
        # Rt = torch.linalg.inv(C2W) # 替换为下列
        Rt[:3, :3] = C2W[:3, :3].T
        Rt[:3, 3] = -1*C2W[:3, :3].T@C2W[:3, 3]

        return Rt
    
    def quaternion_to_rotation_matrix(self):
        """
        Convert a quaternion into a rotation matrix.

        Parameters:
        q (np.ndarray): A quaternion (w, x, y, z).

        Returns:
        np.ndarray: A 3x3 rotation matrix.
        """
        w, x, y, z = self.gt_pose[3:]
        R = np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
        ])
        return R
    
    def build_extrinsic_matrix(self):
        R = self.quaternion_to_rotation_matrix()
        t = self.gt_pose[:3]
        extrinsic_matrix = np.zeros((4, 4))
        extrinsic_matrix[:3, :3] = R
        extrinsic_matrix[:3, 3] = t
        extrinsic_matrix[3, 3] = 1.0
        return extrinsic_matrix
    
    def form_intrinsics_matrix(self, config):
        fx = config["Dataset"]["Calibration"]['fx']
        fy = config["Dataset"]["Calibration"]['fy']
        cx = config["Dataset"]["Calibration"]['cx']
        cy = config["Dataset"]["Calibration"]['cy']
        if config["Dataset"]["Calibration"]["distorted"] == True:
            k1 = config["Dataset"]["Calibration"]['k1']
            k2 = config["Dataset"]["Calibration"]['k2']
            p1 = config["Dataset"]["Calibration"]['p1']
            p2 = config["Dataset"]["Calibration"]['p2']
            intrinsics_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            distortion_coeffs = np.array([k1, k2, p1, p2])
            
            return intrinsics_matrix, distortion_coeffs
        intrinsics_matrix = np.array([[fx, 0, cx], 
                                      [0, fy, cy], 
                                      [0, 0, 1]])
        return intrinsics_matrix


def rotation_matrix(angles):
    """
    Returns the rotation matrix for the given angles.

    """
    base_rotation = torch.eye(3).to('cuda')
    for axis, angle in angles.items():
        
        angle_rad = torch.deg2rad(angle)

        if axis == 'x':
            rotation = torch.tensor([
                [1, 0, 0],
                [0, torch.cos(angle_rad), -torch.sin(angle_rad)],
                [0, torch.sin(angle_rad), torch.cos(angle_rad)]
            ], dtype=torch.float32, device='cuda')
            

        elif axis == 'y':
            rotation = torch.tensor([
                [torch.cos(angle_rad), 0, torch.sin(angle_rad)],
                [0, 1, 0],
                [-torch.sin(angle_rad), 0, torch.cos(angle_rad)]
            ], dtype=torch.float32, device='cuda')

        elif axis == 'z':
            rotation = torch.tensor([
                [torch.cos(angle_rad), -torch.sin(angle_rad), 0],
                [torch.sin(angle_rad), torch.cos(angle_rad), 0],
                [0, 0, 1]
            ], dtype=torch.float32, device='cuda')

        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'.")
        base_rotation = base_rotation @ rotation
    return base_rotation


def transformation_matrix(R, t):
    """
    Returns the transformation matrix combining rotation and translation.

    Parameters:
    R (torch.ndtensor): The rotation matrix.
    t (torch.ndtensor): The translation vector.

    Returns:
    torch.ndtensor: The transformation matrix.
    """
    T = torch.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T



def getWorld2View2(R, t, translate=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
    translate = translate.to(R.device)
    Rt = torch.zeros((4, 4), device=R.device)
    # Rt[:3, :3] = R.transpose()
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    # 将inv操作用物理含义的方法替代，加快速度
    # C2W = torch.linalg.inv(Rt) # test with simple matrix on RTX2060 with 0.2824 s execution time
    C2W = torch.zeros((4,4), device=R.device) # test on RTX2060 with 0.00166 s execution time
    C2W[:3, :3] = R.T
    C2W[:3, 3] = -1*R.T@t
    C2W[3, 3] = 1.0   
    
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    # Rt = torch.linalg.inv(C2W) # 替换为下列
    Rt[:3, :3] = C2W[:3, :3].T
    Rt[:3, 3] = -1*C2W[:3, :3].T@C2W[:3, 3]
    
    return Rt



def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion into a rotation matrix.

    Parameters:
    q (np.ndarray): A quaternion (w, x, y, z).

    Returns:
    np.ndarray: A 3x3 rotation matrix.
    """
    w, x, y, z = q
    R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])
    return R