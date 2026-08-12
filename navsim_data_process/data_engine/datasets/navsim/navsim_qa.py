import os
import json
import math
import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any
from pathlib import Path
import sys
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent.parent.parent
sys.path.append(f"{project_root}/data_qa_generate/")
from data_engine.datasets.navsim.dataset_navsim import VLMNavsim
qa_root = os.environ.get(
    "NAVSIM_QA_ROOT",
    str(project_root / "navsim_dataset" / "data_qa_results_ours"),
)
os.makedirs(qa_root, exist_ok=True)
from data_engine.datasets.navsim.dataset_navsim import VLMNavsim
import pdb


def get_ground_np(pts: np.ndarray) -> np.ndarray:
    """
    Ground segmentation via iterative plane fitting (LPR seeds + PCA plane).
    Returns: ground_label (N,) boolean, True = ground.
    """
    assert pts.ndim == 2 and pts.shape[1] >= 3, f"pts should be (N,>=3), got {pts.shape}"

    th_seeds_ = 1.2
    n_iter = 10
    th_dist_ = 0.3

    # Make num_lpr_ larger / adaptive (NavSim often needs more than 20)
    N = pts.shape[0]
    num_lpr_ = int(min(1000, max(50, N // 50)))  # you can tune

    # 1) seed selection by lowest z
    pts_sort = pts[pts[:, 2].argsort()]
    lpr = np.mean(pts_sort[:num_lpr_, 2])
    pts_g = pts_sort[pts_sort[:, 2] < lpr + th_seeds_]

    # Fallback if seeds are empty (shouldn't usually happen, but be safe)
    if pts_g.shape[0] < 3:
        return np.zeros((N,), dtype=bool)

    normal_ = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    d_ = 0.0

    for _ in range(n_iter):
        if pts_g.shape[0] < 3:
            break

        mean = np.mean(pts_g[:, :3], axis=0)

        # covariance of xyz
        X = pts_g[:, :3] - mean[None, :]
        cov = (X.T @ X) / max(X.shape[0], 1)

        # normal: eigenvector of smallest eigenvalue (use SVD on cov)
        U, S, Vt = np.linalg.svd(cov.astype(np.float32), full_matrices=True)
        normal_ = U[:, 2]

        # make normal point "up" to avoid sign flips
        if normal_[2] < 0:
            normal_ = -normal_

        d_ = -normal_.dot(mean)

        # signed distance to plane, then use ABS distance for ground decision
        dist = pts[:, :3] @ normal_ + d_

        # update ground set
        pts_g = pts[np.abs(dist) < th_dist_]

    # final classification
    dist = pts[:, :3] @ normal_ + d_
    ground_label = np.abs(dist) < th_dist_
    return ground_label


import math
import numpy as np
import json
import os
import torch
import matplotlib
matplotlib.use('Agg')           # 置于 import pyplot 前
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
def gt_2_ego(gt_xy, heading=None, k_ahead=1, min_step=1):
    # theta = torch.tensor(-yaw, dtype=torch.float32)
    gt      = torch.from_numpy(gt_xy)            # shape [T, 2]

    # ---- 平移到原点 --------------------------------------------------------
    origin  = gt[0]
    rel_gt  = gt - origin

    # 用多帧位移稳健估计速度朝向
    k = min(k_ahead, len(gt)-1)
    v = gt[k] - gt[0]
    if torch.linalg.norm(v) < min_step:
        # 找到第一个位移够大的帧
        for j in range(1, len(gt)):
            v = gt[j] - gt[0]
            if torch.linalg.norm(v) >= min_step:
                break

    # ---- 旋转：将 heading 对齐到 +Z ---------------------------------------
    if heading is None:
        # heading = rel_gt[1]                 # (Δx, Δy)
        heading = v
        theta   = torch.atan2(heading[1], heading[0])  # 车头相对 +x 的角度
    else:
        theta = heading
    R = torch.tensor([[ torch.cos(theta), -torch.sin(theta)],   # 逆时针旋转
                    [ torch.sin(theta),  torch.cos(theta)]])  # shape [2,2]

    gt_local = torch.matmul(rel_gt, R)
    gt_local[:, [0,1]] = gt_local[:, [1, 0]]
    gt_local[:, 0] = -gt_local[:, 0]
    gt = gt_local.numpy()

    return gt

def gt_2_ego_xy(gt_xy, heading=None, k_ahead=1, min_step=1):
    # theta = torch.tensor(-yaw, dtype=torch.float32)
    gt      = torch.from_numpy(gt_xy)            # shape [T, 2]

    # ---- 平移到原点 --------------------------------------------------------
    origin  = gt[0]
    rel_gt  = gt - origin

    # 用多帧位移稳健估计速度朝向
    k = min(k_ahead, len(gt)-1)
    v = gt[k] - gt[0]
    if torch.linalg.norm(v) < min_step:
        # 找到第一个位移够大的帧
        for j in range(1, len(gt)):
            v = gt[j] - gt[0]
            if torch.linalg.norm(v) >= min_step:
                break

    # ---- 旋转：将 heading 对齐到 +Z ---------------------------------------
    if heading is None:
        # heading = rel_gt[1]                 # (Δx, Δy)
        heading = v
        theta   = torch.atan2(heading[1], heading[0])  # 车头相对 +x 的角度
    else:
        theta = heading
    R = torch.tensor([[ torch.cos(theta), -torch.sin(theta)],   # 逆时针旋转
                    [ torch.sin(theta),  torch.cos(theta)]])  # shape [2,2]

    gt_local = torch.matmul(rel_gt, R)
    # gt_local[:, [0,1]] = gt_local[:, [1, 0]]
    # gt_local[:, 0] = -gt_local[:, 0]
    gt = gt_local.numpy()

    return gt

def visualize_trajectory(trajectory, outdir, trajectory2=None, map_scale=1, f_path=None, plan_2s=None, q3=None, q4=None, img_col_ratio=2):
    """
    trajectory: (N, 2) array-like, [:,0]=X, [:,1]=Z
    outdir:     保存文件路径（含文件名，如 'out/vis.png'）
    f_path:     左侧要拼接的图片路径（可为 None）
    img_col_ratio: 左图列宽相对比（右轨迹列固定为 2）
    """
    locX = np.asarray(trajectory)[:, 0]
    locZ = np.asarray(trajectory)[:, 1]

    if trajectory2 is not None:
        locX2 = np.asarray(trajectory)[:, 0]
        locZ2 = np.asarray(trajectory)[:, 1]

    # 样式
    mpl.rc("figure", facecolor="white")
    plt.style.use("seaborn-v0_8-whitegrid")

    # 是否两列布局
    # two_cols = f_path is not None and os.path.exists(f_path)
    two_cols = True
    if two_cols:
        fig, axes = plt.subplots(
            nrows=1, ncols=2, figsize=(16, 8), dpi=100,
            gridspec_kw={"width_ratios": [img_col_ratio, 2]}
        )
        ax_img, traj_main_plt = axes
    else:
        fig, traj_main_plt = plt.subplots(nrows=1, ncols=1, figsize=(10, 8), dpi=100)
        ax_img = None

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # 左侧图片
    if ax_img is not None:
        try:
            img = Image.open(f_path)
            ax_img.imshow(img)
            ax_img.set_title("Reference image", y=1.02)
            ax_img.axis("off")
        except Exception as e:
            # 读取失败则退回单列绘制
            fig.delaxes(ax_img)
            fig.set_size_inches(10, 8)
            traj_main_plt = fig.add_subplot(111)

    # 右侧轨迹
    traj_main_plt.set_title("Trajectory (Z, X)", y=1.02)
    traj_main_plt.plot(
        locX, locZ, ".-", label="Ego",
        zorder=6, linewidth=1, markersize=4, color=colors[0]
    )
    if trajectory2 is not None:
        traj_main_plt.plot(
            locX2, locZ2, ".-", label="Ego-smooth",
            zorder=6, linewidth=1, markersize=4, color=colors[1]
        )


    # 自适应边界（含最小可视范围）
    max_x, min_x = float(np.max(locX)), float(np.min(locX))
    max_y, min_y = float(np.max(locZ)), float(np.min(locZ))
    min_x = min(-30.0, min_x)
    max_x = max(30.0, max_x)
    x_ = max(abs(min_x), abs(max_x))
    min_y = min(-10.0, min_y)
    max_y = max(50.0, max_y)

    traj_main_plt.set_xlabel("X")
    traj_main_plt.set_ylabel("Z")
    traj_main_plt.set_xlim([-x_-3, x_+3])
    traj_main_plt.set_ylim([min_y-3, max_y+3])

    handles, labels = traj_main_plt.get_legend_handles_labels()
    if handles:
        traj_main_plt.legend(loc=1, title="Legend",
                             borderaxespad=0., fontsize="medium", frameon=True)
    
    if plan_2s is not None:
        q3 = ''if q3 is None else q3
        q4 = ''if q4 is None else q4
        try:
            speed_plan, cur_cmd = plan_2s
            plt.title(f'plan for future 8 points: {speed_plan, cur_cmd}\n{q3}\n{q4}')
        except:
            speed_plan, cur_cmd, qwen_plan = plan_2s
            plt.title(f'plan for future 8 points: {speed_plan, cur_cmd}\n{qwen_plan}\n{q3}\n{q4}')

    # 保存 + 可视化
    fig.tight_layout()
    plt.savefig(outdir, bbox_inches="tight", pad_inches=0.1)
    plt.show()
    plt.close(fig)


import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image
def visualize_trajectory_v(
    trajectory,
    outdir,
    map_scale=1,
    f_path=None,                  # 可为 str(单图) 或 list[str](图序列)
    img_col_ratio=2,
    fps=2,                        # 降低帧率：8fps 通常足够
    output="mp4",                 # "mp4" | "gif"
    frame_step=1,                 # 每隔几帧取一帧（图像序列 & 轨迹都一起下采样）
    max_frames=121,               # 最多保留多少帧（None 表示不截断）
    px_width=1024,                 # 目标像素宽（总体画布宽）
    dpi=100,                      # 与 px_width 联合作用：inches = px/dpi
    gif_colors=128,               # GIF 自适应色表颜色数（<=256，越小越省）
    bitrate=1800                  # mp4 码率(kbps)，可再降到 1200~1500
):
    locX = np.asarray(trajectory)[:, 0].astype(float)
    locZ = np.asarray(trajectory)[:, 1].astype(float)

    mpl.rc("figure", facecolor="white")
    plt.style.use("seaborn-v0_8-whitegrid")

    # ---- 处理图片序列 ----
    # is_list = isinstance(f_path, (list, tuple))
    # img_paths = []
    # if is_list:
    #     img_paths = [p for p in f_path if isinstance(p, str) and os.path.exists(p)]
    # elif isinstance(f_path, str) and os.path.exists(f_path):
    #     img_paths = [f_path]  # 单图 → 也按动画管线走（只有1帧）
    img_paths = f_path

    # ---- 下采样/截断：对齐帧数，先处理图，再处理轨迹 ----
    if img_paths:
        if frame_step and frame_step > 1:
            img_paths = img_paths[::frame_step]
        if max_frames is not None and len(img_paths) > max_frames:
            idx = np.linspace(0, len(img_paths)-1, max_frames).astype(int)
            img_paths = [img_paths[i] for i in idx]

    if frame_step and frame_step > 1:
        locX = locX[::frame_step]
        locZ = locZ[::frame_step]
    if max_frames is not None and len(locX) > max_frames:
        idx = np.linspace(0, len(locX)-1, max_frames).astype(int)
        locX = locX[idx]; locZ = locZ[idx]

    # 帧对齐（补齐到同长度，重复最后帧）
    n_img = len(img_paths)
    n_traj = len(locX)
    n = max(n_img if n_img>0 else 1, n_traj)
    if n_img == 0: img_paths = [None] * n
    elif n_img < n: img_paths = img_paths + [img_paths[-1]] * (n - n_img)
    if n_traj < n:
        locX = np.concatenate([locX, np.full(n - n_traj, locX[-1])])
        locZ = np.concatenate([locZ, np.full(n - n_traj, locZ[-1])])

    # ---- 画布大小（像素控制体积）----
    # 两列布局下：总宽 px_width，按比例分给左右列
    width_in = px_width / dpi
    if img_paths and img_paths[0] is not None:
        fig = plt.figure(figsize=(width_in, width_in * 0.55), dpi=dpi)
        gs = fig.add_gridspec(1, 2, width_ratios=[img_col_ratio, 2])
        ax_img = fig.add_subplot(gs[0, 0])
        ax_traj = fig.add_subplot(gs[0, 1])
    else:
        fig = plt.figure(figsize=(width_in * 0.66, width_in * 0.55), dpi=dpi)
        ax_traj = fig.add_subplot(111)
        ax_img = None

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # 左图初始化
    im_artist = None
    if ax_img is not None:
        if img_paths[0] is not None:
            try:
                im0 = Image.open(img_paths[0]).convert("RGB")
                im_artist = ax_img.imshow(im0)
            except Exception:
                fig.delaxes(ax_img); ax_img = None
        if ax_img is not None:
            ax_img.set_title("Reference image", y=1.02)
            ax_img.axis("off")

    # 右图范围
    ax_traj.set_title("Trajectory (Z, X)", y=1.02)
    ax_traj.set_xlabel("X"); ax_traj.set_ylabel("Z")
    min_x = min(-30.0, float(locX.min())); max_x = max(30.0, float(locX.max()))
    x_ = max(abs(min_x), abs(max_x))
    min_y = min(-10.0, float(locZ.min())); max_y = max(50.0, float(locZ.max()))
    ax_traj.set_xlim([-x_-3, x_+3]); ax_traj.set_ylim([min_y-3, max_y+3])

    (line_plot,) = ax_traj.plot([], [], ".-", label="Ego",
                                zorder=6, linewidth=1, markersize=4, color=colors[0])
    cur_pt = ax_traj.plot([], [], "o", color=colors[0], markersize=5, zorder=7)[0]
    if ax_traj.get_legend_handles_labels()[0]:
        ax_traj.legend(loc=1, title="Legend", borderaxespad=0., fontsize="medium", frameon=True)

    def init():
        line_plot.set_data([], []); cur_pt.set_data([], [])
        return (line_plot, cur_pt) if im_artist is None else (im_artist, line_plot, cur_pt)

    def update(i):
        # 左图
        if im_artist is not None and img_paths[i] is not None:
            try:
                im = Image.open(img_paths[i]).convert("RGB")
                im_artist.set_data(im)
            except Exception:
                pass
        # 右图
        line_plot.set_data(locX[:i+1], locZ[:i+1])
        cur_pt.set_data([locX[i]], [locZ[i]])
        return (line_plot, cur_pt) if im_artist is None else (im_artist, line_plot, cur_pt)

    anim = FuncAnimation(fig, update, init_func=init, frames=n, interval=1000.0/fps, blit=True)

    # ---- 保存 ----
    os.makedirs(os.path.dirname(outdir) or ".", exist_ok=True)
    ext = os.path.splitext(outdir)[1].lower()

    if output.lower() == "mp4" or ext == ".mp4":
        # 更小更稳：H.264
        try:
            writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=bitrate)
            anim.save(outdir if ext==".mp4" else (os.path.splitext(outdir)[0]+".mp4"),
                      writer=writer, dpi=dpi)
        except Exception:
            # 回退 GIF
            writer = PillowWriter(fps=fps)  # loop 通过 metadata 传
            gif_path = os.path.splitext(outdir)[0] + ".gif"
            anim.save(gif_path, writer=writer, dpi=dpi)
            _quantize_gif_inplace(gif_path, fps=fps, colors=gif_colors)
    else:
        # 直接 GIF，再做量化压缩
        try:
            writer = PillowWriter(fps=fps, metadata={'loop': 0})
        except TypeError:
            writer = PillowWriter(fps=fps)
        gif_path = outdir if ext==".gif" else (os.path.splitext(outdir)[0] + ".gif")
        anim.save(gif_path, writer=writer, dpi=dpi)
        # _quantize_gif_inplace(gif_path, fps=fps, colors=gif_colors)

    plt.close(fig)

def _quantize_gif_inplace(path, fps=8, colors=128):
    """用 PIL 重新量化 GIF 帧，显著减小体积。"""
    try:
        im = Image.open(path)
        frames = []
        for f in ImageSequence.Iterator(im):
            fr = f.convert("RGB").quantize(colors=colors, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG)
            frames.append(fr)
        duration = int(1000 / max(1, fps))
        frames[0].save(path, save_all=True, append_images=frames[1:], optimize=True,
                       loop=0, duration=duration, disposal=2)
    except Exception:
        pass


from PIL import Image
from moviepy.editor import ImageSequenceClip
def images_to_video(image_folder, output_video, fps=20):
    # images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if (img.endswith(".png") or img.endswith(".jpg"))]
    # images.sort()  # Ensure the images are in the correct order
    images = image_folder
    # images.sort(key=lambda p: int(p.split('/')[-1].split('.')[0].split('_')[-1]))

    clip = ImageSequenceClip(images, fps=fps)
    clip.write_videofile(output_video, codec="libx264",
        verbose=False,   # 关闭 MoviePy 自己的进度条
        logger=None      # 禁止写自定义 logger（否则仍会输出）
    )


import os
import shutil
import numpy as np
from PIL import Image
import matplotlib.cm as cm

def _to_uint8_mask(mask):
    """mask can be bool / {0,1} / {0,255} / float. Output uint8 {0,255}."""
    m = np.asarray(mask)
    if m.ndim == 3 and m.shape[-1] == 1:
        m = m[..., 0]
    if m.dtype == np.bool_:
        m = m.astype(np.uint8) * 255
    else:
        m = m.astype(np.float32)
        # assume already 0/255 or 0/1 or arbitrary -> threshold >0
        m = (m > 0.5).astype(np.uint8) * 255 if m.max() <= 1.0 else (m > 0).astype(np.uint8) * 255
    return m

import numpy as np
import matplotlib.cm as cm
import cv2

def colorize(value: np.ndarray, vmin=None, vmax=None, cmap="magma_r", eps=1e-3, pmin=2, pmax=98):
    if value.ndim > 2:
        if value.shape[-1] > 1:
            return value
        value = value[..., 0]

    value = value.astype(np.float32)
    valid = np.isfinite(value) & (value > eps)

    # 关键：vmin/vmax 用 valid 的百分位，别用全图 min/max
    if vmin is None or vmax is None:
        if valid.any():
            lo, hi = np.percentile(value[valid], [pmin, pmax])
            vmin = lo if vmin is None else vmin
            vmax = hi if vmax is None else vmax
        else:
            vmin, vmax = 0.0, 1.0

    denom = max(vmax - vmin, 1e-6)
    norm = (value - vmin) / denom
    norm = np.clip(norm, 0, 1)

    cmapper = plt.get_cmap(cmap)
    rgba = cmapper(norm, bytes=True)
    rgba[~valid] = 0
    return rgba[..., :3]


def _overlay_mask_on_image(img_rgb, mask_u8, alpha=0.35):
    """Overlay mask (0/255) as red tint on RGB image."""
    img = img_rgb.astype(np.float32)
    m = (mask_u8 > 0)[..., None].astype(np.float32)
    red = np.zeros_like(img); red[..., 0] = 255.0
    out = img * (1 - alpha * m) + red * (alpha * m)
    return out.astype(np.uint8)

def save_depth_and_masks(cur_img_path, cur_depth, cur_d_mask, cur_g_mask, cur_s_mask,
                         gs_view_dir, token):
    os.makedirs(gs_view_dir, exist_ok=True)
    token_dir = os.path.join(gs_view_dir, token)
    os.makedirs(token_dir, exist_ok=True)

    if cur_depth is not None:
        tar_h, tar_w = int(cur_depth.shape[0]), int(cur_depth.shape[1])
    else:
        # fallback to dynamic mask
        tar_h, tar_w = int(cur_d_mask.shape[0]), int(cur_d_mask.shape[1])

    img_bgr = cv2.imread(cur_img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {cur_img_path}")

    img_bgr_rs = cv2.resize(img_bgr, (tar_w, tar_h), interpolation=cv2.INTER_AREA)

    dst = os.path.join(token_dir, 'rgb.jpg')
    cv2.imwrite(dst, img_bgr_rs)

    depth_path   = os.path.join(token_dir, f"depth.jpg")
    dynamic_path = os.path.join(token_dir, f"d_mask.jpg")
    ground_path  = os.path.join(token_dir, f"g_mask.jpg")
    sky_path     = os.path.join(token_dir, f"s_mask.jpg")

    # load rgb
    img_rgb = np.array(Image.open(dst).convert("RGB"))

    # depth colormap
    depth_rgb = colorize(cur_depth, cmap="jet")
    cv2.imwrite(depth_path, depth_rgb[..., ::-1])

    # masks
    dmask_u8 = _to_uint8_mask(cur_d_mask)
    gmask_u8 = _to_uint8_mask(cur_g_mask)
    smask_u8 = _to_uint8_mask(cur_s_mask)

    cv2.imwrite(dynamic_path, dmask_u8)
    cv2.imwrite(ground_path, gmask_u8)
    cv2.imwrite(sky_path, smask_u8)

    # optional overlays (helpful to verify alignment)
    cv2.imwrite(os.path.join(token_dir, f"d_overlay.jpg"),
                _overlay_mask_on_image(img_rgb, dmask_u8)[..., ::-1])
    cv2.imwrite(os.path.join(token_dir, f"g_overlay.jpg"),
                _overlay_mask_on_image(img_rgb, gmask_u8)[..., ::-1])
    cv2.imwrite(os.path.join(token_dir, f"s_overlay.jpg"),
                _overlay_mask_on_image(img_rgb, smask_u8)[..., ::-1])

    print("saved:", depth_path, dynamic_path, ground_path, sky_path)



from scipy.signal import savgol_filter
import numpy as np

def smooth_traj_sg(xy, dt=0.5, win_sec=1.5, poly=3, pin_first=True):
    xy = np.asarray(xy, float)
    T  = xy.shape[0]
    if T < 3:
        return xy

    # 计算窗口
    k = int(round(win_sec / dt))
    if k % 2 == 0: k += 1
    k = min(k, T if pin_first else T)          # 后面会对 diff 长度再处理
    if T == 4: k, poly = 3, 1
    elif poly >= k - 1: poly = max(1, k - 2)

    if not pin_first:
        # 直接平滑整条（首点会被改动）——不建议你的场景
        if k < 3: return xy
        return savgol_filter(xy, window_length=k, polyorder=poly, axis=0, mode="interp")

    # --- 关键：平滑增量，保证首点不变 ---
    dxy = np.diff(xy, axis=0)          # 形状 (T-1, 2)
    k_d = min(k, dxy.shape[0] if dxy.shape[0] % 2 == 1 else dxy.shape[0]-1)
    if k_d < 3:                         # 点太少，不平滑
        dxy_s = dxy
    else:
        poly_d = min(poly, max(1, k_d - 2))
        dxy_s = savgol_filter(dxy, window_length=k_d, polyorder=poly_d, axis=0, mode="interp")

    xy_s = np.empty_like(xy)
    xy_s[0] = xy[0]                     # 锚定起点
    xy_s[1:] = xy[0] + np.cumsum(dxy_s, axis=0)
    return xy_s

def setup_model(device: torch.device, depth_ckpt: str | None = None):
    """Initialize and setup the depth estimation model."""

    depth_ckpt = depth_ckpt or os.environ.get("DEPTH_ANYTHING_MODEL")
    if not depth_ckpt:
        raise ValueError(
            "Depth checkpoint is not configured. Set DEPTH_ANYTHING_MODEL "
            "in env.local.sh or pass depth_ckpt explicitly."
        )
    third_party_root = os.environ.get("GAUSSIAN_STORM_THIRD_PARTY")
    if third_party_root and third_party_root not in sys.path:
        sys.path.insert(0, third_party_root)
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "depth_anything_v2 is unavailable. Install it in the active "
            "environment or set GAUSSIAN_STORM_THIRD_PARTY in env.local.sh."
        ) from error

    depthv2 = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
    depthv2.load_state_dict(torch.load(depth_ckpt, map_location="cpu"))
    depthv2 = depthv2.eval().to(device)
    for param in depthv2.parameters():
        param.requires_grad = False
    return depthv2

IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets.folder import default_loader

class ListDataset:
    """Dataset class that loads images from a list of file paths."""
    
    def __init__(
        self,
        data_list: str,
        transform = None,
        return_path: bool = False,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_list: Path to text file containing image paths
            transform: Optional transforms to apply to images
            return_path: Whether to return image paths along with images
        """
        self.transform = transform
        self.return_path = return_path
        self.loader = default_loader
        self.samples = self._load_samples(data_list)

    def _load_samples(self, data_list: str) -> list:
        """Load image paths from the data list file."""
        # samples = []
        # with open(data_list, "r") as f:
        #     for line in f:
        #         file_path = line.strip()
        #         samples.append(file_path)
        # return samples
        return data_list

    def __getitem__(self, index: int):
        """Get an image and optionally its path."""
        img_pth = self.samples[index]
        try:
            img = self.loader(img_pth)
        except Exception as e:
            print(f"Error loading '{img_pth}': {e}")
            return self.__getitem__((index + 1) % len(self.samples))

        if self.transform is not None:
            img = self.transform(img)
            
        to_return = [img]
        if self.return_path:
            to_return.append(img_pth)
        return tuple(to_return) if len(to_return) > 1 else to_return[0]

    def __len__(self) -> int:
        return len(self.samples)

@torch.no_grad()
def get_sky_mask(dataloader, depthv2, gs_tar_shape) -> None:
    """
    Extract sky masks from images using depth estimation.
    
    Args:
        dataloader: DataLoader containing images
        depthv2: Depth estimation model
    """
    torch.cuda.empty_cache()
    device = next(depthv2.parameters()).device
    
    pbar = tqdm(dataloader, desc=f"Extracting sky masks")
    for samples, paths in pbar:
        samples = samples.to(device)
        # predict depth using the model
        with torch.autocast(device.type, dtype=torch.bfloat16):
            outputs = depthv2(samples)
        # identify sky regions (depth = 0)
        sky_masks = (outputs == 0).float()
        # sky_masks = sky_masks.cpu().numpy()
        if sky_masks.ndim == 3:
            sky_masks = sky_masks[:, None]     # (B,1,H,W)
        sky_masks = torch.nn.functional.interpolate(
            sky_masks, size=gs_tar_shape, mode="nearest"
        )
        sky_masks = (sky_masks.cpu().numpy()[:, 0] * 255).astype(np.uint8)

        return sky_masks.reshape(-1, 3, gs_tar_shape[0], gs_tar_shape[1])

class Config:
    # Trajectory planning parameters
    NUM_FUT = 4
    NUM_FUT_NAVI = 12
    VEL_NAVI_THRESH = 4.0
    VEL_DIFF_THRESH = 3.0
    VAL_STOP = 2.0
    LAT_THRESH = 1.5
    ANGLE_THRESH = 30.0
    ANGLE_THRESH_NAVI = 8.0  # abused
    DATA_FPS = 2
    TARGET_FPS = 2

    # Navigation command thresholds
    NAVI_DIS_FORWARD_THRESH = 20.0
    NAVI_DIS_THRESH = 10.0

    # Path and output configurations
    DATA_ROOT = qa_root
    QA_ROOT = qa_root
    BASE_PATH = f"{project_root}/data_qa_generate/"


pedal_status = {
    'const': 'KEEP',
    'accelerate': 'ACCELERATE',
    'decelerate': 'DECELERATE',
    'stop': 'STOP'
}

path_status = {
    'right turn': 'RIGHT_TURN',
    'right lane change': 'RIGHT_CHANGE',
    'left turn': 'LEFT_TURN',
    'left lane change': 'LEFT_CHANGE',
    'straight': 'STRAIGHT'
}
# Common prompts
image_prompt = """<FRONT VIEW>:\n<image>\n
<FRONT LEFT VIEW>:\n<image>\n
<FRONT RIGHT VIEW>:\n<image>\n
<LEFT VIEW>:\n<image>\n
<RIGHT VIEW>:\n<image>\n
<BACK LEFT VIEW>:\n<image>\n
<BACK RIGHT VIEW>:\n<image>\n
<BACK VIEW>:\n<image>\n"""


def inference_3(img_urls, prompt, system_prompt="You are a helpful assistant", max_new_tokens=1024):
    image_f = Image.open(img_urls[1])
    image_l = Image.open(img_urls[0])
    image_r = Image.open(img_urls[2])
    messages = [
        {
        "role": "system",
        "content": system_prompt
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": prompt
            },
            {
            "image": img_urls[0]
            },
            {
            "image": img_urls[1]
            },
            {
            "image": img_urls[2]
            }
        ]
        }
    ]
    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # print("input:\n",text)
    inputs = qwen_processor(text=[text], images=[image_l, image_f, image_r], padding=True, return_tensors="pt").to('cuda')

    output_ids = qwen_model.generate(**inputs, max_new_tokens=1024)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = qwen_processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # print("output:\n",output_text[0])

    input_height = inputs['image_grid_thw'][0][1]*14
    input_width = inputs['image_grid_thw'][0][2]*14

    return output_text[0]


# speed and path plan: explain the reason
def get_q3(speed, cur_command, speed_plan, img_paths):

    pedal_decision = {
        'KEEP': 'maintain the current speed',
        'ACCELERATE': 'accelerate',
        'DECELERATE': 'decelerate',
        'STOP': 'stop the car'
    }

    prompt = "You are driving, " \
        f"your current speed is {int(speed)} m/s, " \
        f"and the navigation command is {cur_command}. " \
        "Your driving decision for the next few seconds is to " \
        f"{pedal_decision[speed_plan]}. " \
        "Based on the provided images of the FRONT LEFT VIEW, FRONT VIEW and FRONT RIGHT VIEW, " \
        "explain the most likely reason for this decision in one or two concise sentence."

    global qwen_model, qwen_processor
    if qwen_model is None:
        init_qwen()

    meta_plan = inference_3(img_paths, prompt)

    return [prompt, meta_plan]

    qas = []
    pedal_decision = {
        'KEEP': 'maintain the current speed',
        'ACCELERATE': 'accelerate',
        'DECELERATE': 'decelerate',
        'STOP': 'stop the car'
    }

    path_decision = {
        'RIGHT_TURN': 'turn right',
        'RIGHT_CHANGE': 'change to the right lane',
        'LEFT_TURN': 'turn left',
        'LEFT_CHANGE': 'change to the left lane',
        'STRAIGHT': 'go straight'
    }
    for i, info in enumerate(infos):
        speed, speed_plan, path_plan, navigation_command = info

        if speed_plan == 'stop':
            decision = pedal_decision[pedal_status[speed_plan]]
        else:
            decision = pedal_decision[pedal_status[speed_plan]] + \
                ' and ' + path_decision[path_status[path_plan]]

        question = image_prompt + "You are driving, " \
            f"your current speed is {int(speed)} m/s, " \
            f"and the navigation command is {navigation_command} " \
            "your driving decision for the next three seconds is to " \
            f"{decision}. " \
            "Based on the provided image of the driving environment, " \
            "explain the most likely reason for this decision in one or two concise sentence."
        qas.append({"images": images[i], "messages": [
                   {"role": "user", "content": question}, {"role": "assistant", "content": ""}]})
    return qas


# traffic light
def get_q4(img_paths):

    prompt = "Given the provided image of the FRONT VIEW from a car's perspective, identify if there is a traffic light. Respond with 'Red', 'Green', 'Yellow', or 'None'."

    global qwen_model, qwen_processor
    if qwen_model is None:
        init_qwen()

    meta_plan = inference(img_paths[1], prompt)

    return [prompt, meta_plan]


    qas = []
    question = "Given the provided forward-facing image <image> from a car's perspective, identify if there is a traffic light that affects the car's behavior. Respond with 'Red', 'Green', 'Yellow', or 'None'."
    for imgs in images:
        qas.append({"images": [imgs[0]], "messages": [
                   {"role": "user", "content": question}, {"role": "assistant", "content": ""}]})
    return qas

# scene description
def q5(images, **kwargs):
    qas = []
    views = ["ring_front_center", "ring_front_left", "ring_front_right", "ring_side_left",
             "ring_side_right", "ring_rear_left", "ring_rear_right", "ring_back_center"]
    for imgs in images:
        for vi, view in enumerate(views):
            question = "Suppose you are driving, and I'm providing you with the image " \
                f"captured by the car's {view} <image>, generate a description of the driving scene " \
                "which includes the key factors for driving planning, including the positions " \
                "and movements of vehicles and pedestrians; prevailing weather conditions; " \
                "time of day, distinguishing between daylight and nighttime; road conditions, " \
                "indicating smooth surfaces or the presence of obstacles; and the status of traffic lights " \
                "which influence your decision making, specifying whether they are red or green. " \
                "The description should be concise, providing an accurate understanding " \
                "of the driving environment to facilitate informed decision-making."
            qas.append({"images": [imgs[vi]], "messages": [
                       {"role": "user", "content": question}, {"role": "assistant", "content": ""}]})
    return qas

# speed and path plan: why three seconds?
def get_q6(speed, cur_command, speed_plan):

    q = f"Your current speed is {int(speed)} m/s, the navigation command is {cur_command}," \
            f" based on the understanding of the driving scene and the navigation information," \
            f" what is your speed plan for the next few seconds?" \
            " Please answer your SPEED plan. SPEED includes KEEP, ACCELERATE and DECELERATE, and STOP." \
            " For example, a correct answer format is like 'KEEP'."

    a = f"{speed_plan}"

    return [q, a]

    qas = []

    for i, info in enumerate(infos):
        speed, speed_plan, path_plan, navigation_command = info

        question = image_prompt + f"Your current speed is {int(speed)} m/s, the navigation command is {navigation_command}," \
            f" based on the understanding of the driving scene and the navigation information," \
            f"what is your plan for the next three seconds?" \
            "Please answer your SPEED plan and your PATH plan. SPEED includes KEEP, ACCELERATE and DECELERATE, and STOP, " \
            "PATH includes STRAIGHT, RIGHT_CHANGE, LEFT_CHANGE, RIGHT_TURN, LEFT_TURN. " \
            "Based on the provided image of the driving environment, " \
            "For example, a correct answer format is like 'KEEP, LEFT_CHANGE'."
        answer = f"{pedal_status[speed_plan]},{path_status[path_plan]}"
        qas.append({"images": images[i], "messages": [
                   {"role": "user", "content": question}, {"role": "assistant", "content": answer}]})
    return qas


def calculate_angle(dx, dy):
    """
    x: forward, y: left
    返回 [0, 360) 度，0 度沿 +x，逆时针为正（左偏为正）
    """
    angle = math.degrees(math.atan2(dy, dx))  # 注意顺序：atan2(dy, dx)
    return angle if angle >= 0 else angle + 360

def angle_diff_deg(start_angle, end_angle):
    """
    返回 end - start 的差值，范围 [-180, 180]，>0 左转，<0 右转
    """
    diff = (end_angle - start_angle + 180) % 360 - 180
    return diff



def point_to_line_distance(points):
    """Calculate minimum distance from last point to the line formed by first two points"""
    if len(points) < 3:
        raise ValueError("At least three points required")

    points = np.asarray(points)
    (x1, y1), (x2, y2), (xn, yn) = points[0], points[1], points[-1]

    if np.allclose([x1, y1], [x2, y2]):
        # Fallback to point-to-point distance
        return np.sqrt((xn - x1)**2 + (yn - y1)**2)

    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2

    numerator = np.abs(A * xn + B * yn + C)
    denominator = np.sqrt(A**2 + B**2)
    return float(numerator / denominator) if denominator > 1e-10 else 0.0


def point_to_line_projection_distance(points):
    """Calculate projection distance of last point onto the line formed by first two points"""
    if len(points) < 3:
        raise ValueError("At least three points required")

    x1, y1 = points[0]
    x2, y2 = points[1]
    xn, yn = points[-1]
    dx, dy = x2 - x1, y2 - y1
    vx, vy = xn - x1, yn - y1

    line_length = (dx ** 2 + dy ** 2) ** 0.5
    if line_length == 0 or np.isnan(line_length):
        return 0

    return (vx * dx + vy * dy) / line_length

import numpy as np

def plan_to_path(raw_xy):
    xys = raw_xy

    # 开始方向（p0 -> p1）
    dx_start = xys[1][0] - xys[0][0]
    dy_start = xys[1][1] - xys[0][1]
    start_angle = calculate_angle(dx_start, dy_start)

    # 结束方向（p_{n-2} -> p_{n-1}）
    dx_end = xys[-1][0] - xys[-2][0]
    dy_end = xys[-1][1] - xys[-2][1]
    end_angle = calculate_angle(dx_end, dy_end)

    angle_diff = angle_diff_deg(start_angle, end_angle)
    dis = point_to_line_distance(xys) if len(xys) >= 3 else 0.0

    if dis < Config.LAT_THRESH and abs(angle_diff) < Config.ANGLE_THRESH:
        path_plan = "straight"
    else:
        path_plan = (
            "turn right" if angle_diff <= -Config.ANGLE_THRESH else
            "turn left" if angle_diff >= Config.ANGLE_THRESH else
            "right lane change" if angle_diff < 0 else
            "left lane change"
        )
    return (path_plan, dis, angle_diff)

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

qwen_model = None
qwen_processor = None

def init_qwen():

    global qwen_model, qwen_processor

    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"

    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map="auto"
    )
    qwen_processor = AutoProcessor.from_pretrained(model_path)

def inference(img_url, prompt, system_prompt="You are a helpful assistant", max_new_tokens=1024):
    image_f = Image.open(img_url)
    messages = [
        {
        "role": "system",
        "content": system_prompt
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": prompt
            },
            {
            "image": img_url
            }
        ]
        }
    ]
    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # print("input:\n",text)
    inputs = qwen_processor(text=[text], images=[image_f], padding=True, return_tensors="pt").to('cuda')

    output_ids = qwen_model.generate(**inputs, max_new_tokens=1024)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = qwen_processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # print("output:\n",output_text[0])

    input_height = inputs['image_grid_thw'][0][1]*14
    input_width = inputs['image_grid_thw'][0][2]*14

    return output_text[0]

# no need cmd + speed query next speed 
def plan_to_path_gpt(img_path, cur_command):
    
    global qwen_model, qwen_processor
    if qwen_model is None:
        init_qwen()

    if cur_command != 'keep straight':
        return cur_command  # left, right turn
    
    prompt = f"Your current navigation command is {cur_command}," \
            f" based on the given image of the driving scene and ego future trajectory," \
            f" what is your plan for the next few seconds? " \
            "Please only answer one of the three options: keep lane, left change lane, right change lane. " \
            "Left and right change lane means the ego is actually doing lane chage in the future trajectory. " \
            "For example, a correct answer format is like 'keep lane'."
    meta_plan = inference(img_path, prompt)

    print(meta_plan)

    return meta_plan


def plan_to_action(points, dt=0.5, v_stop=0.3, a_th=0.2):
    """
    points: (9, D), p0 is current, p8 is 4s future plan
    returns: KEEP / ACCELERATE / DECELERATE / STOP
    """
    p = np.asarray(points, dtype=float)
    # assert p.shape[0] == 9

    # segment speeds s0..s7
    v = (p[1:] - p[:-1]) / dt
    s = np.linalg.norm(v, axis=1)  # (8,)

    v_near  = s[0:2].mean()        # ~0-1s
    v_later = s[2:4].mean()        # ~1-2s

    # STOP intent: near future is basically stopped
    if v_near < v_stop and v_later < v_stop:
        return "STOP"

    # trend accel (m/s^2); window gap is ~1.0s
    a = (v_later - v_near) / 1.0

    if a > a_th:
        return "ACCELERATE"
    elif a < -a_th:
        return "DECELERATE"
    else:
        return "KEEP"   # must dominate


def get_plan(raw_poses, num_fut=Config.NUM_FUT, num_fut_navi=Config.NUM_FUT_NAVI,
             vel_navi_thresh=Config.VEL_NAVI_THRESH, vel_diff_thresh=Config.VEL_DIFF_THRESH,
             val_stop=Config.VAL_STOP, lat_thresh=Config.LAT_THRESH,
             angle_thresh=Config.ANGLE_THRESH, angle_thresh_navi=Config.ANGLE_THRESH_NAVI,
             data_fps=Config.DATA_FPS, target_fps=Config.TARGET_FPS):
    """Generate speed and path plans based on trajectory points"""
    # if not raw_poses or len(raw_poses) == 0:
    #     return [], [], [], []

    interval = max(int(data_fps / target_fps), 1)
    raw_xy = np.array(raw_poses)[::interval]

    # Speed calculations
    # speeds = np.zeros(len(raw_xy)) if len(raw_xy) <= 1 else np.append(
    #     np.sqrt(np.sum(np.diff(raw_xy, axis=0)**2, axis=1)) * target_fps,
    #     [0]
    # )

    # Speed planning
    # speed_plans = []
    # if len(speeds) > num_fut:
    #     speed_diffs = speeds[num_fut:] - speeds[:-num_fut]
    #     speed_plans = [
    #         'stop' if speeds[i] < val_stop else
    #         'accelerate' if diff >= vel_diff_thresh else
    #         'decelerate' if diff <= -vel_diff_thresh else 'const'
    #         for i, diff in enumerate(speed_diffs)
    #     ]
    #     speed_plans += [speed_plans[-1]] * (len(speeds) - len(speed_plans))
    # else:
    #     speed_plans = ['stop' if s < val_stop else 'const' for s in speeds]

    #TODO potential to more data since we only consider 2s
    speed_plan = plan_to_action(raw_xy)

    # use cmd
    # path_plan = None

    return speed_plan

    # Path planning
    path_plans = []
    if len(raw_xy) >= Config.NUM_FUT + 1:
        for i in range(len(raw_xy) - Config.NUM_FUT):
            xys = raw_xy[i:i+Config.NUM_FUT]
            start_angle = calculate_angle(
                xys[1][0]-xys[0][0], xys[1][1]-xys[0][1])
            end_angle = calculate_angle(
                xys[-1][0]-xys[-2][0], xys[-1][1]-xys[-2][1])
            angle_diff = end_angle - start_angle
            dis = point_to_line_distance(xys) if len(xys) >= 2 else 0.0

            if dis < Config.LAT_THRESH:
                path_plan = "straight"
            else:
                path_plan = (
                    "right turn" if angle_diff <= -Config.ANGLE_THRESH else
                    "left turn" if angle_diff >= Config.ANGLE_THRESH else
                    "right lane change" if angle_diff < 0 else "left lane change"
                )
            path_plans.append(path_plan)
        path_plans += [path_plans[-1]] * (len(raw_xy) - len(path_plans))
    else:
        path_plans = ["straight"] * len(raw_xy)

    # Navigation commands
    # navi_commands = []
    # if len(raw_xy) >= Config.NUM_FUT_NAVI + 1:
    #     for i in range(len(raw_xy) - Config.NUM_FUT_NAVI):
    #         xys = raw_xy[i:i+Config.NUM_FUT_NAVI]
    #         start_angle = calculate_angle(
    #             xys[1][0]-xys[0][0], xys[1][1]-xys[0][1])
    #         end_angle = calculate_angle(
    #             xys[-1][0]-xys[-2][0], xys[-1][1]-xys[-2][1])
    #         angle_diff = end_angle - start_angle
    #         dis = point_to_line_distance(xys)
    #         dis_forward = point_to_line_projection_distance(xys)

    #         if dis < Config.NAVI_DIS_THRESH:
    #             navi_command = 'go straight'
    #         elif dis_forward >= Config.NAVI_DIS_FORWARD_THRESH and dis >= Config.NAVI_DIS_THRESH:
    #             navi_command = f"go straight and turn {'left' if angle_diff > 0 else 'right'}"
    #         else:
    #             navi_command = f"turn {'left' if angle_diff > 0 else 'right'}"
    #         navi_commands.append(navi_command)
    #     navi_commands += [navi_commands[-1]] * \
    #         (len(raw_xy) - len(navi_commands))
    # else:
    #     navi_commands = ["go straight"] * len(raw_xy)

    return speeds.tolist(), speed_plans, path_plans, navi_commands


def plot_positions(positions, title, save_path):
    """Helper function to plot positions with start and end points marked"""
    # x, y = zip(*positions) if positions else ([], [])

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, c=range(len(positions)), cmap='viridis', s=10)
    plt.plot(x, y, alpha=0.3)  # Connect points with a line

    # Mark start and end points
    if len(positions) > 0:
        plt.scatter([x[0]], [y[0]], c='green', s=100, label='Start')
        plt.scatter([x[-1]], [y[-1]], c='red', s=100, label='End')

    plt.title(title)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.colorbar(label='Frame index')

    # Save the plot
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def project_by_cam2global(pts_ego, ego2global, cam2global, K):
    N = pts_ego.shape[0]
    pts_ego_h = np.concatenate([pts_ego, np.ones((N,1),np.float32)], axis=1)  # (N,4)
    pts_w = (ego2global @ pts_ego_h.T).T                                      # (N,4)
    w2c = np.linalg.inv(cam2global).astype(np.float32)
    pts_c = (w2c @ pts_w.T).T[:, :3]                                          # (N,3)
    z = pts_c[:, 2]
    uv = (K @ (pts_c / (z[:,None] + 1e-8)).T).T[:, :2]
    return uv, z

import numpy as np
import torch

def reproj_err_plucker(dd, lidar_idx=0, view_name="cam_f0", n=20000, use_canonical=True, device="cuda"):
    view_order = dd.get("view_order", ["cam_f0","cam_l0","cam_r0"])
    vid = view_order.index(view_name)

    K  = torch.from_numpy(dd["intrinsic"][lidar_idx, vid]).float().to(device)          # (3,3)
    c2w = torch.from_numpy(dd["cam2globals"][lidar_idx, vid]).float().to(device)       # (4,4)

    # optional canonicalize like STORM
    if use_canonical:
        c2w_ref = torch.from_numpy(dd["cam2globals"][0, view_order.index("cam_f0")]).float().to(device)
        w2canon = torch.linalg.inv(c2w_ref)
        c2w = w2canon @ c2w

    depth = torch.from_numpy(dd["depth"][lidar_idx, vid, ..., 0]).float().to(device)   # (H,W)
    H, W = depth.shape

    # sample valid pixels
    valid = depth > 0
    ys, xs = torch.where(valid)
    if ys.numel() == 0:
        raise RuntimeError("no valid depth")
    idx = torch.randint(0, ys.numel(), (min(n, ys.numel()),), device=device)
    ys = ys[idx].float()
    xs = xs[idx].float()
    z  = depth[ys.long(), xs.long()]  # (N,)

    # rays in world (same math as PluckerEmbedder, patch_size=1)
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    x_cam = (xs - cx + 0.5) / fx
    y_cam = (ys - cy + 0.5) / fy
    dirs_cam = torch.stack([x_cam, y_cam, torch.ones_like(x_cam)], dim=-1)             # (N,3)

    R = c2w[:3,:3]
    t = c2w[:3, 3]
    dirs_w = dirs_cam @ R.T                                                            # (N,3)
    orig_w = t[None].expand_as(dirs_w)                                                  # (N,3)

    # lift to world using z (NOTE: this matches PluckerEmbedder's dirs convention: z-scales the ray)
    pts_w = orig_w + dirs_w * z[:, None]                                                # (N,3)

    # project back
    w2c = torch.linalg.inv(c2w)
    pts_c = (torch.cat([pts_w, torch.ones((pts_w.shape[0],1), device=device)], dim=-1) @ w2c.T)[:, :3]
    uv = (pts_c @ K.T)
    u = uv[:,0] / (uv[:,2] + 1e-8)
    v = uv[:,1] / (uv[:,2] + 1e-8)

    err = torch.sqrt((u - xs)**2 + (v - ys)**2)
    import pdb
    pdb.set_trace()
    print(float(err.mean()), float(err.median()), float(err.quantile(0.9)))

# 用法
# dd = pickle.load(open(pkl_path,"rb"))
# print("plucker reproj err:", reproj_err_plucker(dd, lidar_idx=3, view_name="cam_f0", use_canonical=True))



import pickle
processed_data_root = os.environ.get(
    "DATA_ROOT",
    str(project_root / "navsim_dataset"),
)
video_root = os.environ.get(
    "NAVSIM_VIDEO_ROOT",
    str(Path(processed_data_root) / "navsim_video"),
)
load_video = 0
load_lidar = 0
only_lidar = 0
only_lidar_root = os.environ.get(
    "NAVSIM_LIDAR_DEPTH_ROOT",
    str(Path(processed_data_root) / "navsim_gt_depth_144256_single_frame"),
)
from data_engine.datasets.navsim.loaders.navsim.visualization.camera import _transform_pcs_to_images
gs_root = os.environ.get(
    "NAVSIM_GS_ROOT",
    str(Path(processed_data_root) / "navsim_storm_144256_12frame_1209"),
)
gs_tar_h, gs_tar_w = 144, 256
# gs_tar_h, gs_tar_w = 160*2, 240*2
import sys
try:
    local_rank = int(sys.argv[1])
except (IndexError, ValueError):
    local_rank = 0

load_ann = 0
if load_ann:
    from data_engine.datasets.navsim.anno_utils import *

load_q = 0
q_root = os.environ.get(
    "NAVSIM_VQA_ROOT",
    str(Path(processed_data_root) / "navsim_vqa_ours"),
)

mini_test_datalist = os.environ.get(
    "NAVSIM_MINI_TEST_DATALIST",
    str(project_root / "mini_meta.json"),
)
with open(mini_test_datalist, "r") as f:
    meta_test_mini_list = json.load(f)

if load_lidar:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depthv2 = setup_model(device)
import time
missing_video = {}

import cv2, numpy as np

def bev_rgb_to_occ(bev_rgb):  # (500,500,3) uint8
    gray = cv2.cvtColor(bev_rgb[:496, :496], cv2.COLOR_RGB2GRAY)
    occ = (gray < 255).astype(np.uint8)          # 白底≈255，框/线更暗
    # occ = cv2.morphologyEx(occ, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    return occ*255  # (500,500) {0,1}

load_bev = 0
def gen_info(root):
    os.makedirs(root, exist_ok=True)
    # for mode in ["test", "train"]:
    # for mode in ["navhard_two_stage"]:  # 220, 5912 #confirmed
    for mode in ["test"]:

        all_x = []
        all_y = []
        all_smooth_l2 = []

        meta_dir = os.path.join(root, mode)
        os.makedirs(meta_dir, exist_ok=True)
        dataset = VLMNavsim(mode=mode)
        video_dir = os.path.join(video_root, mode)
        os.makedirs(video_dir, exist_ok=True)
        gs_dir = os.path.join(gs_root, mode)
        os.makedirs(gs_dir, exist_ok=True)
        only_lidar_dir = os.path.join(only_lidar_root, mode)
        os.makedirs(only_lidar_dir, exist_ok=True)
        q_dir = os.path.join(q_root, mode)
        os.makedirs(q_dir, exist_ok=True)
        enu_positions = []

        q6_dir = os.path.join(q_dir, 'q6')
        os.makedirs(q6_dir, exist_ok=True)

        q3_dir = os.path.join(q_dir, 'q3')
        os.makedirs(q3_dir, exist_ok=True)

        q4_dir = os.path.join(q_dir, 'q4')
        os.makedirs(q4_dir, exist_ok=True)

        valid_lens = np.arange(len(dataset))
        valid_len = np.array_split(valid_lens, 10)[local_rank]
        valid_len = set(valid_len)

        for sid in tqdm(range(len(dataset)), desc=f"Processing {mode} samples"):
            # if sid not in valid_len:
            #     continue

            a = time.time()
            container = dataset.get_container_in(sid, load_lidar or only_lidar, load_bev, meta_test_mini_list)

            # print(time.time()-a)

            # not in mini test
            if container is None:
                continue

            
            # if container['token'] not in ['c59175106e2f5b26','0a678d2136b35b56']:
            #     continue

            # tolist [x, y]
            # xy1 = container["frame_data"][3]["ego_status"].ego_pose[:2].tolist()    # global
            # enu_positions.append(xy1)  # add like  [[x1, y1], [x2, y2], ...]
            this_glo_status = {}
            this_glo_images = {}
            # if len(container["ego_status"]) != 14 or len(container["ego_status"]) != 12:
            if False:
                import pdb
                pdb.set_trace()
                print(f'ignore sample {sid} with len {len(container["ego_status"])}')
                continue
            
            if load_bev:
                frame_data = container["frame_data"][3]
                ann = frame_data.get("annotations", {})
                bev = container['bev']
                if sid < 3:
                    print(f"Sample {sid}")
                    # print(f"  Total: {ann_data['num_total']}, Dynamic: {ann_data['num_dynamic']}")
                    # print(f"  Vehicles: {ann_data['num_vehicles']}, Peds: {ann_data['num_pedestrians']}")
                    # vis here
                    frame_data = container["frame_data"][3]
                    mask_255 = bev_rgb_to_occ(bev)
                    import pdb
                    pdb.set_trace()
                    cv2.imwrite(f'mask_{sid}.jpg', mask_255)
                    img = frame_data["cameras"]["cam_f0"]["image_path"]
                    os.system(f'cp {img} img_{sid}.jpg')
                
                with open(f"{meta_dir}/{container['token']}-bev.pkl", "wb") as f:
                        pickle.dump(bev, f, protocol=pickle.HIGHEST_PROTOCOL)
                continue

            if load_ann:
                assert False
                # ========== 处理 Annotations ==========
                all_frame_ann_data = []
                
                for frame_idx in range(min(12, len(container["frame_data"]))):
                    if frame_idx != 3:
                        continue
                    frame_data = container["frame_data"][frame_idx]
                    ann = frame_data.get("annotations", {})
                    
                    # 转换annotations格式
                    ann_dict = {
                            'gt_boxes': ann.boxes,
                            'gt_names': ann.names,
                            'gt_velocity_3d': ann.velocity_3d
                            # 'instance_tokens': ann.get('instance_tokens', []),
                            # 'track_tokens': ann.get('track_tokens', []),
                        }
                    
                    # 处理annotation
                    ann_data = process_frame_annotations(ann_dict)
                    all_frame_ann_data.append(ann_data)
                    
                    # 打印调试信息
                    import pdb
                    pdb.set_trace()
                    if frame_idx == 3 and sid < 3:
                        print(f"Sample {sid}, Frame {frame_idx}:")
                        # print(f"  Total: {ann_data['num_total']}, Dynamic: {ann_data['num_dynamic']}")
                        # print(f"  Vehicles: {ann_data['num_vehicles']}, Peds: {ann_data['num_pedestrians']}")
                        # vis here
                        mask = ann_data['bev_raster'].reshape(50, 50, 1)
                        mask_255 = mask.astype(np.uint8)
                        mask_255 = np.squeeze(mask_255) # Removes the 1-dim
                        cv2.imwrite(f'mask_{sid}.jpg', mask_255)
                        img = frame_data["cameras"]["cam_f0"]["image_path"]
                        os.system(f'cp {img} img_{sid}.jpg')
                        # bev_occupancy = ann_data['bev_occupancy']
                        # mask_255 = (bev_occupancy * 255).astype(np.uint8)
                        # cv2.imwrite('bev_occ.jpg', mask_255)
                
                    # ========== 保存数据 ==========
                    
                    with open(f"{meta_dir}/{container['token']}-det.pkl", "wb") as f:
                        pickle.dump(ann_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            for idx, ego_status in enumerate(container["ego_status"]):
                # if mode not in ['train']:
                #     if idx >= 4:
                #         break
                this_glo_status[idx] = {
                    "global_pose": ego_status.ego_pose,
                    "velocity": ego_status.ego_velocity,
                    "acceleration": ego_status.ego_acceleration,
                    "command": ego_status.driving_command
                }
            # stack status in time
            glo_status = {
                    "global_poses": np.array([this_glo_status[i]["global_pose"] for i in range(len(this_glo_status))]),
                    "velocities": np.array([this_glo_status[i]["velocity"] for i in range(len(this_glo_status))]),
                    "accelerations": np.array([this_glo_status[i]["acceleration"] for i in range(len(this_glo_status))]),
                    "commands": np.array([this_glo_status[i]["command"] for i in range(len(this_glo_status))]),
                }

            views = ['cam_f0', 'cam_l0', 'cam_r0', 'cam_l1', 'cam_r1', 'cam_l2', 'cam_r2', 'cam_b0']
            for idx, frame_data in enumerate(container["images"]):
                if idx == 13:
                    # no frames for last frame
                    continue
                if mode not in ['train']:
                    if idx >= 4:
                        break
                for view in views:
                    if idx not in this_glo_images:
                        this_glo_images[idx] = {}
                    try:
                        this_glo_images[idx][view] = {
                            "image_path": frame_data[view]["image_path"],
                            "sensor2lidar_rotation": frame_data[view]["sensor2lidar_rotation"],
                            "sensor2lidar_translation": frame_data[view]["sensor2lidar_translation"],
                            "intrinsics": frame_data[view]["intrinsics"],
                            "distortion": frame_data[view]["distortion"]
                        }
                    except:
                        import pdb
                        pdb.set_trace()

            glo_images = {}
            for view in views:
                glo_images[view] = {
                            "image_paths": [this_glo_images[i][view]["image_path"] for i in range(len(this_glo_images))],
                            "sensor2lidar_rotations": np.array([this_glo_images[i][view]["sensor2lidar_rotation"] for i in range(len(this_glo_images))]),
                            "sensor2lidar_translations": np.array([this_glo_images[i][view]["sensor2lidar_translation"] for i in range(len(this_glo_images))]),
                            "intrinsics": np.array([this_glo_images[i][view]["intrinsics"] for i in range(len(this_glo_images))]),
                            "distortions": np.array([this_glo_images[i][view]["distortion"] for i in range(len(this_glo_images))]),
                        }
            meta_res = {
                "glo_status": glo_status,
                "glo_images": glo_images,
            }
            # print(time.time()-a)
            # if not os.path.exists(f"{meta_dir}/{container['token']}.pkl"):
            import pdb
            pdb.set_trace()
            with open(f"{meta_dir}/{container['token']}.pkl", "wb") as f:
                pickle.dump(meta_res, f, protocol=pickle.HIGHEST_PROTOCOL)
                # print(f'writing: {container['token']}.pkl')

            import pdb
            pdb.set_trace()
            
            # print(time.time()-a)

            save_path = f"{meta_dir}/{container['token']}.gif"

            # if mode == "mini":
            if True:

                if load_q:
                    # get meta plan
                    fut_pose = gt_2_ego_xy(glo_status['global_poses'][3:12,:2])
                    fut_vel = glo_status['velocities'][3:12,:2]
                    fut_command = glo_status['commands'][3:12].argmax(1)
                    
                    LABELS = ["turn left", "keep straight", "turn right", "unknown"]

                    fut_command = [LABELS[f_c] for f_c in fut_command]

                    speed_plan = get_plan(fut_pose)

                    cur_command = fut_command[0]

                    speed = np.linalg.norm(fut_vel[:1])

                    if cur_command == 'unknown':
                        import pdb
                        pdb.set_trace()

                    
                    if not os.path.exists(f"{q6_dir}/{container['token']}.pkl"):

                        q6 = get_q6(speed, cur_command, speed_plan)

                        # print(q6)

                        with open(f"{q6_dir}/{container['token']}.pkl", "wb") as f:
                                pickle.dump(q6, f, protocol=pickle.HIGHEST_PROTOCOL)

                    # plan_decision = plan_to_path(fut_pose)

                    if not os.path.join(f"{q3_dir}/{container['token']}.pkl"):

                        q3 = get_q3(speed, cur_command, speed_plan,
                            [
                                glo_images['cam_l0']['image_paths'][3],
                                glo_images['cam_f0']['image_paths'][3],
                                glo_images['cam_r0']['image_paths'][3]
                            ]
                        )

                        print(q3)

                        with open(f"{q3_dir}/{container['token']}.pkl", "wb") as f:
                                pickle.dump(q3, f, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    else:
                        with open(f"{q3_dir}/{container['token']}.pkl", "rb") as f:
                                q3 = pickle.load(f)

                    # if not os.path.exists(f"{q4_dir}/{container['token']}.pkl"):
                    if True:
                        q4 = get_q4([
                                glo_images['cam_l0']['image_paths'][3],
                                glo_images['cam_f0']['image_paths'][3],
                                glo_images['cam_r0']['image_paths'][3]
                            ])
                        
                        print(q4)

                        with open(f"{q4_dir}/{container['token']}.pkl", "wb") as f:
                                pickle.dump(q4, f, protocol=pickle.HIGHEST_PROTOCOL)
                    else:
                        with open(f"{q4_dir}/{container['token']}.pkl", "rb") as f:
                                q4 = pickle.load(f)
                

                # 2d gen
                # traj_2 = smooth_traj_sg(gt_2_ego(glo_status['global_poses'][3:12,:2]))
                # visualize_trajectory_v(gt_2_ego(glo_status['global_poses'][3:12,:2]), save_path, f_path=glo_images['cam_f0']['image_paths'][3:12])
                save_path = f"{meta_dir}/{container['token']}_smooth.png"
                # visualize_trajectory(gt_2_ego(glo_status['global_poses'][3:12,:2]), save_path, trajectory2=traj_2, f_path=glo_images['cam_f0']['image_paths'][3], plan_2s=(speed_plan,cur_command), q3=q3[1], q4=q4[1])

                # visualize_trajectory(gt_2_ego(glo_status['global_poses'][3:12,:2]), save_path, trajectory2=traj_2, f_path=glo_images['cam_f0']['image_paths'][3])

                # meta_plan = plan_to_path_gpt(save_path, cur_command)

                # print(time.time()-a)

                # visualize_trajectory(gt_2_ego(glo_status['global_poses'][3:12,:2]), save_path, trajectory2=traj_2, f_path=glo_images['cam_f0']['image_paths'][3], plan_2s=(speed_plan,cur_command,meta_plan))

                if load_video:
                    for view in ['cam_f0', 'cam_l0', 'cam_r0']:
                        try:
                            img_list = glo_images[view]['image_paths'][3:12]
                            video_view_dir = os.path.join(video_dir, view)
                            os.makedirs(video_view_dir, exist_ok=True)
                            images_to_video(img_list, os.path.join(video_view_dir, container['token']+'.mp4'), fps=2)
                            ext = img_list[0][-4:]
                            shutil.copy2(
                                img_list[0],
                                os.path.join(video_view_dir, container['token'] + ext),
                            )
                        except Exception as e:
                            # print(glo_images[view]['image_paths'][3:12])
                            print(f'data missing: {e}')
                            if container['token'] in missing_video:
                                missing_video[container['token']].append(e)
                            else:
                                missing_video[container['token']] = [e]

                    # print(time.time()-a)

                

                # 3d gen
                if only_lidar:
                    depth = {}
                    # only consider 3 so 0 is idx
                    lidar_idx = 3
                    lidar = container['frame_data'][lidar_idx]['lidar']
                    for view in ['cam_f0', 'cam_l0', 'cam_r0']:
                            r = container["images"][lidar_idx][view]["sensor2lidar_rotation"]
                            t = container["images"][lidar_idx][view]["sensor2lidar_translation"]
                            k = container["images"][lidar_idx][view]["intrinsics"]
                            # img_h, img_w = 1080, 1920

                            img_path = container["images"][lidar_idx][view]["image_path"]  # 这个key名你得用下面print确认
                            img = cv2.imread(img_path)
                            img_h, img_w = img.shape[:2]

                            pixel_coords, valid_mask, depth_values = _transform_pcs_to_images(lidar, r, t, k, (img_h, img_w), (gs_tar_h, gs_tar_w))

                            # Get valid depth points and corresponding coordinates
                            valid_depth_points = depth_values[valid_mask]
                            valid_cam_coords = pixel_coords[valid_mask]
                            # Convert coordinates to integer indices
                            x_indices = valid_cam_coords[:, 0].astype(np.int32)
                            y_indices = valid_cam_coords[:, 1].astype(np.int32)
                            # Initialize arrays to accumulate depth sums and counts
                            depth_sums = np.zeros((gs_tar_h, gs_tar_w))
                            depth_counts = np.zeros((gs_tar_h, gs_tar_w))

                            np.add.at(depth_sums, (y_indices, x_indices), valid_depth_points)
                            np.add.at(depth_counts, (y_indices, x_indices), 1)

                            depth_map = np.divide(depth_sums, depth_counts, where=depth_counts > 0)

                            depth_map[depth_counts == 0] = 0

                            depth[view] = depth_map
                    with open(f"{only_lidar_dir}/{container['token']}.pkl", "wb") as f:
                        pickle.dump(depth, f, protocol=pickle.HIGHEST_PROTOCOL)
                    continue
                if load_lidar:
                    depths = [] # 5 * view
                    cam2egos = [] # 5 * view
                    intrinsics = [] # 5 * view
                    ego2globals = [] # 5 repeat to 5 * view
                    cam2globals = [] # 5 * view
                    dynamic_masks = [] # 5 * view
                    ground_masks = [] 
                    for lidar_idx in range(12):  # 12 for all
                        lidar = container['frame_data'][lidar_idx]['lidar']
                        # ego 2 global
                        ego2global = container['frame_data'][lidar_idx]["ego2global"]

                        # ego2globals.append([ego2global]*len(views))
                        ego2global_mat = np.array(container['frame_data'][lidar_idx]["ego2global"], dtype=np.float32)
                        ego2globals.append(np.repeat(ego2global_mat[None], len(['cam_f0', 'cam_l0', 'cam_r0']), axis=0))

                        depth = []  # need scale
                        cam2ego = []
                        intrinsic = []
                        cam2global = []
                        dynamic_mask = []   # 
                        ground_mask = []    # need scale

                        for view in ['cam_f0', 'cam_l0', 'cam_r0']:
                            r = container["images"][lidar_idx][view]["sensor2lidar_rotation"]
                            t = container["images"][lidar_idx][view]["sensor2lidar_translation"]
                            k = container["images"][lidar_idx][view]["intrinsics"]
                            # img_h, img_w = 1080, 1920

                            img_path = container["images"][lidar_idx][view]["image_path"]  # 这个key名你得用下面print确认
                            img = cv2.imread(img_path)
                            img_h, img_w = img.shape[:2]

                            pixel_coords, valid_mask, depth_values = _transform_pcs_to_images(lidar, r, t, k, (img_h, img_w), (gs_tar_h, gs_tar_w))

                            # if lidar_idx == 3:
                            #     print(view, "valid ratio:", float(valid_mask.mean()),
                            #         "z_med:", float(np.median(depth_values[valid_mask])) if valid_mask.any() else None)

                            # check opencv
                            # try:
                            #     import pdb
                            #     pdb.set_trace()
                            #     pkl_path = os.path.join(gs_root, "mini", f"{container['token']}.pkl")
                            #     with open(pkl_path, "rb") as f:
                            #         dd = pickle.load(f)

                            #     reproj_err_plucker(dd, lidar_idx=3, view_name="cam_f0", use_canonical=True)
                            #     reproj_err_plucker(dd, lidar_idx=3, view_name="cam_f0", use_canonical=False)
                            #     import pdb
                            #     pdb.set_trace()

                            #     uv_gt = pixel_coords[valid_mask]  # tar-space from your existing projection pipeline

                            #     # dd['cam2globals'] expected shape: (5, 3, 4, 4) in your saved format
                            #     view_order = dd.get("view_order", ["cam_f0", "cam_l0", "cam_r0"])
                            #     vid = view_order.index(view)
                            #     c2g_saved = np.asarray(dd["cam2globals"][lidar_idx][vid], np.float32)
                            #     K_tar = np.asarray(dd["intrinsic"][lidar_idx][vid], np.float32)
                            #     pts = lidar[:3, :].T.astype(np.float32)
                            #     uv_saved, z_saved = project_by_cam2global(pts, ego2global_mat, c2g_saved, K_tar)
                            #     # uv_now,   z_now   = project_by_cam2global(pts, ego2global_mat, c2g,       K_tar)

                            #     e_saved = np.linalg.norm(uv_saved[valid_mask] - uv_gt, axis=1)
                            #     # e_now   = np.linalg.norm(uv_now[mask]   - uv_gt, axis=1)

                            #     print("reproj err vs pixel_coords (saved) mean/med/p90:",
                            #         e_saved.mean(), np.median(e_saved), np.percentile(e_saved, 90))
                            #     # print("reproj err vs pixel_coords (now)   mean/med/p90:",
                            #     #     e_now.mean(), np.median(e_now), np.percentile(e_now, 90))

                            #     print("neg depth ratio saved/now:",
                            #         (z_saved[mask] < 0).mean())


                            #     diff = np.abs(c2g_saved - c2g).max()
                            #     print(f"  cam2global vs saved max|diff| = {diff:.6f}")
                            # except:
                            #     import pdb
                            #     pdb.set_trace()



                            # Get valid depth points and corresponding coordinates
                            valid_depth_points = depth_values[valid_mask]
                            valid_cam_coords = pixel_coords[valid_mask]
                            # Convert coordinates to integer indices
                            x_indices = valid_cam_coords[:, 0].astype(np.int32)
                            y_indices = valid_cam_coords[:, 1].astype(np.int32)
                            # Initialize arrays to accumulate depth sums and counts
                            depth_sums = np.zeros((gs_tar_h, gs_tar_w))
                            depth_counts = np.zeros((gs_tar_h, gs_tar_w))

                            np.add.at(depth_sums, (y_indices, x_indices), valid_depth_points)
                            np.add.at(depth_counts, (y_indices, x_indices), 1)

                            depth_map = np.divide(depth_sums, depth_counts, where=depth_counts > 0)

                            depth_map[depth_counts == 0] = 0

                            # depth_map[depth_map>=80] = 0
                            # depth_map[0:int(gs_tar_h*0.4),:] = 0


                            # valid_mask[depth_values>=80] = False


                            # depth_map = np.full((gs_tar_h, gs_tar_w), np.inf, dtype=np.float32)
                            # pixel_indices = np.round(pixel_coords[valid_mask]).astype(np.int32)
                            # pixel_indices[:, 0] = np.clip(pixel_indices[:, 0], 0, gs_tar_w - 1)  # x in [0, W-1]
                            # pixel_indices[:, 1] = np.clip(pixel_indices[:, 1], 0, gs_tar_h - 1)  # y in [0, H-1]

                            # np.minimum.at(depth_map, (pixel_indices[:, 1], pixel_indices[:, 0]), depth_values[valid_mask])
                            # depth_map[np.isinf(depth_map)] = 0.0

                            # cam to ego
                            R_s2l = np.asarray(r, dtype=np.float32)      # (3, 3)
                            t_s2l = np.asarray(t, dtype=np.float32)   # (3,) 或 (3,1)

                            T_s2l = np.eye(4, dtype=np.float32)
                            T_s2l[:3, :3] = R_s2l
                            T_s2l[:3, 3] = t_s2l.reshape(3)   # 确保是一维 (3,)

                            # cam2world
                            c2g = ego2global @ T_s2l

                            # scene flow define to o now
                            scene_flow = np.zeros(
                                (depth_map.shape[0], depth_map.shape[1], 3), 
                                dtype=np.float32
                            )

                            # 拼成 (H, W, 4): [depth, flow_x, flow_y, flow_z]
                            concate_image = np.concatenate(
                                [depth_map[:, :, None], scene_flow],
                                axis=-1
                            ).astype(np.float32)

                            d_mask = container['frame_data'][lidar_idx]['dynamic_mask'][view]
                            d_mask = cv2.resize(d_mask, (gs_tar_w, gs_tar_h), interpolation=cv2.INTER_NEAREST)

                            # g_mask = get_ground_np(lidar[:3,:].T)
                            # ground_image = np.zeros((img_h, img_w))
                            # ground_image[pixel_indices[:, 1], pixel_indices[:, 0]] = g_mask[valid_mask][:,0]
                            # ground_image = (ground_image * 255).astype(np.uint8)

                            pts = lidar[:3, :].T

                            g_mask = get_ground_np(pts).reshape(-1).astype(np.uint8)  # (N,) 0/1
                            if lidar_idx == 3:
                                print(view)
                                print("z stats", pts[:,2].min(), pts[:,2].max(), np.median(pts[:,2]))
                                print("ground ratio", g_mask.mean())

                            # pixel_coords 已经是 tar-space (x,y)
                            pix = np.round(pixel_coords[valid_mask]).astype(np.int32)
                            x = np.clip(pix[:, 0], 0, gs_tar_w - 1)
                            y = np.clip(pix[:, 1], 0, gs_tar_h - 1)

                            ground_image = np.zeros((gs_tar_h, gs_tar_w), dtype=np.uint8)
                            np.maximum.at(ground_image, (y, x), g_mask[valid_mask] * 255)


                            depth.append(concate_image)
                            cam2ego.append(T_s2l)

                            K_tar = np.array(k, dtype=np.float32).copy()
                            K_tar[0,0] *= gs_tar_w / img_w
                            K_tar[0,2] *= gs_tar_w / img_w
                            K_tar[1,1] *= gs_tar_h / img_h
                            K_tar[1,2] *= gs_tar_h / img_h
                            intrinsic.append(K_tar) # tar size

                            cam2global.append(c2g)
                            dynamic_mask.append(d_mask)
                            ground_mask.append(ground_image)
                        depths.append(depth)
                        cam2egos.append(cam2ego)
                        intrinsics.append(intrinsic)
                        cam2globals.append(cam2global)
                        dynamic_masks.append(dynamic_mask)
                        ground_masks.append(ground_mask)

                    depths = np.array(depths)
                    cam2egos = np.array(cam2egos)
                    intrinsics = np.array(intrinsics)
                    ego2globals = np.array(ego2globals)
                    cam2globals = np.array(cam2globals)
                    dynamic_masks = np.array(dynamic_masks)
                    ground_masks = np.array(ground_masks)

                    # get sky
                    img_transformation = transforms.Compose([
                        transforms.Resize([518, 518], interpolation=Image.BICUBIC, antialias=True),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=IMGNET_MEAN, std=IMGNET_STD),
                    ])

                    img_list = []
                    for idx in range(12):
                        for view in ['cam_f0', 'cam_l0', 'cam_r0']:
                            cur_img = glo_images[view]['image_paths'][idx]
                            img_list.append(cur_img)
                    
                    sky_dataset = ListDataset(data_list=img_list, transform=img_transformation, return_path=True)
                    sky_data_loader = torch.utils.data.DataLoader(
                        sky_dataset,
                        batch_size=12*3,
                        num_workers=1,
                        shuffle=False,
                        drop_last=False,
                    )
                    
                    sky_masks = get_sky_mask(sky_data_loader, depthv2, (gs_tar_h, gs_tar_w))

                    # vis for debug
                    gs_rgbs = []
                    trans = transforms.Resize((gs_tar_h, gs_tar_w), interpolation=Image.BICUBIC, antialias=True)
                    for idx in range(12):
                        this_gs_rgbs = []
                        for view_id, view in enumerate(['cam_f0', 'cam_l0', 'cam_r0']):
                            cur_img = glo_images[view]['image_paths'][idx]
                            cur_img = Image.open(cur_img).convert("RGB")
                            cur_img = np.asarray(trans(cur_img))
                            this_gs_rgbs.append(cur_img) # h, w, 3
                        gs_rgbs.append(this_gs_rgbs)
                    gs_rgbs = np.array(gs_rgbs)


                    for view_id, view in enumerate(['cam_f0', 'cam_l0', 'cam_r0']):
                        gs_view_dir = os.path.join(gs_dir, view)
                        os.makedirs(gs_view_dir, exist_ok=True)
                        cur_img = glo_images[view]['image_paths'][11]
                        cur_depth = depths[11,view_id,:,:,:1]
                        cur_d_mask = dynamic_masks[11,view_id]
                        cur_g_mask = ground_masks[11,view_id]
                        cur_s_mask = sky_masks[11,view_id]
                        
                        save_depth_and_masks(cur_img, cur_depth, cur_d_mask, cur_g_mask, cur_s_mask, gs_view_dir, container['token'])
                    gs_pkl = {
                        'view_order': ['cam_f0', 'cam_l0', 'cam_r0'],
                        'rgb': gs_rgbs,
                        'depth': depths,
                        'cam2ego': cam2egos,
                        'intrinsic': intrinsics,
                        'ego2global': ego2globals,
                        'cam2globals': cam2globals,
                        'dynamic_mask': dynamic_masks,
                        'ground_mask': ground_masks,
                        'sky_masks': sky_masks,
                    }

                    with open(f"{gs_dir}/{container['token']}.pkl", "wb") as f:
                        pickle.dump(gs_pkl, f, protocol=pickle.HIGHEST_PROTOCOL)


                all_smooth_l2.append(np.linalg.norm(traj_2-gt_2_ego(glo_status['global_poses'][3:12,:2])))
            all_x.append(gt_2_ego(glo_status['global_poses'][3:12,:2])[1:,0])
            all_y.append(gt_2_ego(glo_status['global_poses'][3:12,:2])[1:,1])
        
        all_x = np.array(all_x)
        all_y = np.array(all_y)
        std_x = np.std(all_x.reshape(-1, 1))
        print(all_x.shape)
        std_y = np.std(all_y.reshape(-1, 1))
        scale = std_y / (std_x+1e-6)
        print(scale)
        all_smooth_l2 = np.array(all_smooth_l2).mean()
        # with open('missing_video.json', 'w') as json_file:
        import pdb
        pdb.set_trace()
        # 4.4798 for train
        # title = f"{mode} Trajectory"
        # save_path = os.path.join(OUTPUT_DIR, f"{mode}_trajectory.png")
        # plot_positions(enu_positions, title, save_path)
        continue

        speeds, speed_plans, path_plans, navi_commands = get_plan(
            enu_positions)
        results = [(float(speed), sp, pp, nc) for speed, sp, pp,
                   nc in zip(speeds, speed_plans, path_plans, navi_commands)]
        
        import pdb
        pdb.set_trace()

        with open(f"{root}/{mode}_ego_results.json", "w") as f:
            json.dump(results, f)


def gen_qa(data_root, qa_root,data_fps =2, target_fps = 2):
    for mode in ["test", "train" ]:

        # q3s = []
        q4s = []
        # q5s = []
        q6s = []

        with open(f'{data_root}/{mode}_ego_results.json', "r") as f:
            ego = json.load(f)

        views = ['cam_f0', 'cam_10', 'cam_r0', 'cam_11', 'cam_r1', 'cam_12', 'cam_r2', 'cam_b0']
        dataset = VLMNavsim(mode=mode)
        images=[]
        for sid in tqdm(range(len(dataset)), desc=f"Processing {mode} samples"):

            images.append([
                os.path.join(Config.BASE_PATH, cam["image_path"])
                for cam in dataset.get_container_in(sid)["frame_data"][3]["cameras"].values()
                if "image_path" in cam
            ])

        q3s += q3(images, ego)
        q4s += q4(images)
        q5s += q5(images)
        q6s += q6(images, ego)
        print(mode, len(q3s), len(q4s), len(q5s),len(q6s))
        os.makedirs(qa_root, exist_ok=True)
        with open(f"{qa_root}/{mode}_q3.json", "w") as f:
            json.dump(q3s, f)
        with open(f"{qa_root}/{mode}_q4.json", "w") as f:
            json.dump(q4s, f)
        with open(f"{qa_root}/{mode}_q5.json", "w") as f:
            json.dump(q5s, f)
        with open(f"{qa_root}/{mode}_q6.json", "w") as f:
            json.dump(q6s, f)

gen_info(root=os.environ.get("NAVSIM_META_ROOT", str(Path(processed_data_root) / "meta")))
# gen_qa()
