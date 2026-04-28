import os
import glob
import numpy as np
import cv2
import viser
import viser.transforms as tf
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="WinT3R 推理结果离线可视化")
    parser.add_argument("--save_dir", "-d", type=str, required=True, help="recon.py 的 --save_dir 路径")
    parser.add_argument("--image_dir", "-i", type=str, required=True, help="原始图像路径")
    parser.add_argument("--max_points", type=int, default=5000, help="每帧显示的最大点数")
    parser.add_argument("--conf_thresh", type=float, default=0.01, help="置信度阈值")
    args = parser.parse_args()

    # 路径检查
    local_frame_dir = os.path.join(args.save_dir, "local_frames")
    pose_path = os.path.join(args.save_dir, "global_poses.npz")
    
    if not os.path.exists(pose_path):
        print(f"错误: 找不到全局位姿文件 {pose_path}")
        return
    if not os.path.exists(local_frame_dir):
        print(f"错误: 找不到局部帧目录 {local_frame_dir}")
        return

    # 加载全局位姿 (N, 4, 4) W2C 矩阵
    poses = np.load(pose_path)['poses']
    print(f"已加载 {len(poses)} 个相机的位姿。")

    # 获取所有局部帧文件
    npz_files = sorted(glob.glob(os.path.join(local_frame_dir, "*.npz")))
    if not npz_files:
        print("错误: local_frames 目录下没有 .npz 文件")
        return

    # 启动 Viser
    server = viser.ViserServer()
    server.scene.set_up_direction("-y") # WinT3R 坐标系通常 y 向下

    # GUI
    point_size_slider = server.gui.add_slider("Point Size", min=0.001, max=0.05, step=0.001, initial_value=0.005)

    def load_image(full_name):
        for ext in ['.jpg', '.png', '.jpeg', '.webp']:
            p = os.path.join(args.image_dir, full_name + ext)
            if os.path.exists(p):
                img = cv2.imread(p)
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None

    print("正在处理并加载点云...")
    for p in tqdm(npz_files):
        try:
            f_name = os.path.basename(p)
            f_idx = int(f_name.split('_')[1].split('.')[0])
            
            data = np.load(p)
            pts_local = data['pts_local'] # (H, W, 3)
            conf = data['conf']          # (H, W)
            cam_idx = int(data['cam_idx'])
            
            if cam_idx >= len(poses):
                continue

            # 计算 Camera-to-World (C2W)
            w2c = poses[cam_idx]
            if w2c.shape == (3, 4):
                # 补齐为 4x4 矩阵
                w2c_4x4 = np.eye(4)
                w2c_4x4[:3, :4] = w2c
                w2c = w2c_4x4
                
            c2w = np.linalg.inv(w2c)

            # 1. 过滤低置信度点
            valid_mask = conf > args.conf_thresh
            sel_pts = pts_local[valid_mask]
            
            if len(sel_pts) == 0:
                continue

            # 2. 加载颜色
            img_name = f_name.replace(".npz", "")
            img = load_image(img_name)
            if img is not None:
                img_resized = cv2.resize(img, (pts_local.shape[1], pts_local.shape[0]))
                sel_cols = img_resized[valid_mask]
            else:
                sel_cols = np.ones_like(sel_pts) * 255

            # 3. 随机下采样以提升渲染性能
            if len(sel_pts) > args.max_points:
                idx = np.random.choice(len(sel_pts), args.max_points, replace=False)
                sel_pts, sel_cols = sel_pts[idx], sel_cols[idx]

            # 4. 坐标转换: pts_world = R * pts_local + t
            pts_world = (c2w[:3, :3] @ sel_pts.T).T + c2w[:3, 3]

            # 5. 添加到场景
            # 添加相机
            server.scene.add_camera_frustum(
                f"/cameras/frame_{f_idx}",
                fov=2 * np.arctan(320 / (2 * 500)),
                aspect=1.0, scale=0.05,
                wxyz=tf.SO3.from_matrix(c2w[:3, :3]).wxyz,
                position=c2w[:3, 3]
            )
            
            # 添加点云
            server.scene.add_point_cloud(
                f"/points/frame_{f_idx}",
                points=pts_world,
                colors=sel_cols,
                point_size=point_size_slider.value
            )

        except Exception as e:
            print(f"处理帧 {p} 时出错: {e}")

    print(f"\n可视化完成！请访问: http://localhost:8080")
    
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("退出...")

if __name__ == "__main__":
    main()
