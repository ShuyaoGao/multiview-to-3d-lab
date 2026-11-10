"""路线 B：转台视频 → 抽帧 → 无位姿重建（VGGT 估计相机+深度）→ TSDF 融合 → marching cubes 网格。

视频来源：
  --source blender   用真值网格在 Blender 里渲染转台视频（带程序化噪声纹理），验证“视频→重建”这一段
  --source video     任意 mp4（例如 MiniMax/Hailuo 生成的旋转视频，见 src/p6/video/minimax.py）

  python -m p6.routes.video_recon --cases gear,spot --frames 24
  python -m p6.routes.video_recon --video some.mp4 --frames 24

重建在 VGGT 的坐标系里（第一帧相机系，尺度任意），评测时用 align.py 做相似变换对齐（aligned=True）。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import trimesh

from .. import blender
from ..evaluate import eval_case, list_cases, summarize
from ..paths import BENCH_DATA, MODELS, TOOLS
from ..stamp import new_run_dir, stamp_run, write_json


# ----------------------------------------------------------------------------- 视频
def frames_to_mp4(frames: list[Path], mp4: Path, fps: int = 12) -> None:
    w = imageio.get_writer(str(mp4), fps=fps, codec="libx264", quality=7, macro_block_size=8)
    for f in frames:
        w.append_data(imageio.imread(f)[:, :, :3])
    w.close()


def extract_frames(mp4: Path, out: Path, n: int) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    rd = imageio.get_reader(str(mp4))
    allf = [fr for fr in rd]
    rd.close()
    idx = np.linspace(0, len(allf), n, endpoint=False).astype(int)
    paths = []
    for k, i in enumerate(idx):
        p = out / f"f_{k:03d}.png"
        imageio.imwrite(p, allf[i])
        paths.append(p)
    return paths


# ----------------------------------------------------------------------------- VGGT
def run_vggt(frames: list[Path], out_npz: Path, weights: Path | None = None) -> dict:
    import torch

    sys.path.insert(0, str(TOOLS / "vggt"))
    from safetensors.torch import load_file
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    weights = weights or (MODELS / "vggt" / "model.safetensors")
    t0 = time.time()
    model = VGGT()
    model.load_state_dict(load_file(str(weights)))
    model = model.to("cuda").eval()
    imgs = load_and_preprocess_images([str(f) for f in frames]).to("cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(imgs)
    ext, intr = pose_encoding_to_extri_intri(pred["pose_enc"], imgs.shape[-2:])
    res = {
        "images": imgs.float().cpu().numpy(),                 # (S,3,H,W) 0..1
        "extrinsic": ext[0].float().cpu().numpy(),            # (S,3,4) OpenCV cam-from-world
        "intrinsic": intr[0].float().cpu().numpy(),           # (S,3,3)
        "depth": pred["depth"][0, ..., 0].float().cpu().numpy(),        # (S,H,W)
        "depth_conf": pred["depth_conf"][0].float().cpu().numpy(),      # (S,H,W)
    }
    peak = torch.cuda.max_memory_allocated() / 2**30
    np.savez_compressed(out_npz, **res)
    del model
    torch.cuda.empty_cache()
    return {"seconds": round(time.time() - t0, 1), "peak_vram_gb": round(peak, 2), "n_frames": len(frames)}


# ----------------------------------------------------------------------------- 融合
def fg_masks(images: np.ndarray, thresh: float = 0.94) -> np.ndarray:
    """白背景抠图：任一通道 < thresh 即前景，再做一次开运算去噪。"""
    m = images.min(axis=1) < thresh
    k = np.ones((3, 3), np.uint8)
    return np.stack([cv2.morphologyEx(x.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool) for x in m])


def tsdf_fuse(d: dict, res: int = 192, trunc_vox: float = 3.0, conf_pct: float = 20.0) -> tuple[trimesh.Trimesh, dict]:
    """GPU TSDF：深度 + 前景掩码。掩码外的像素当作“很远”（自由空间），等价于同时做轮廓雕刻。"""
    import torch

    dev = "cuda"
    E = torch.tensor(d["extrinsic"], device=dev)
    K = torch.tensor(d["intrinsic"], device=dev)
    D = torch.tensor(d["depth"], device=dev)
    C = torch.tensor(d["depth_conf"], device=dev)
    M = torch.tensor(fg_masks(d["images"]), device=dev)
    S, H, W = D.shape
    thr = torch.quantile(C[M].float(), conf_pct / 100) if M.any() else 0
    # 前景点云 → 包围盒
    v, u = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    pts = []
    for s in range(S):
        ok = M[s] & (C[s] > thr)
        z = D[s][ok]
        x = (u[ok] - K[s, 0, 2]) / K[s, 0, 0] * z
        y = (v[ok] - K[s, 1, 2]) / K[s, 1, 1] * z
        pc = torch.stack([x, y, z], -1)
        R, t = E[s, :, :3], E[s, :, 3]
        pts.append((pc - t) @ R)  # world = R^T (pc - t)
    P = torch.cat(pts)
    lo = torch.quantile(P, 0.005, dim=0)
    hi = torch.quantile(P, 0.995, dim=0)
    ctr, ext = (lo + hi) / 2, (hi - lo).max() * 0.6
    vox = 2 * ext / res
    g = (torch.arange(res, device=dev, dtype=torch.float32) + 0.5) * vox - ext
    X, Y, Z = torch.meshgrid(g + ctr[0], g + ctr[1], g + ctr[2], indexing="ij")
    Pw = torch.stack([X, Y, Z], -1).reshape(-1, 3)
    tsdf = torch.zeros(Pw.shape[0], device=dev)
    wsum = torch.zeros(Pw.shape[0], device=dev)
    outside = torch.zeros(Pw.shape[0], device=dev, dtype=torch.int32)
    trunc = trunc_vox * vox
    for s in range(S):
        pc = Pw @ E[s, :, :3].T + E[s, :, 3]
        z = pc[:, 2]
        uu = (pc[:, 0] / z * K[s, 0, 0] + K[s, 0, 2]).round().long()
        vv = (pc[:, 1] / z * K[s, 1, 1] + K[s, 1, 2]).round().long()
        inimg = (z > 1e-4) & (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
        uu, vv = uu.clamp(0, W - 1), vv.clamp(0, H - 1)
        fg = M[s][vv, uu] & inimg
        # 投影到背景 = 该体素在物体外（轮廓雕刻）
        outside += (inimg & ~M[s][vv, uu]).int()
        dobs = D[s][vv, uu]
        good = fg & (C[s][vv, uu] > thr)
        sdf = (dobs - z).clamp(-trunc, trunc) / trunc
        upd = good & (dobs - z > -trunc)
        tsdf[upd] += sdf[upd]
        wsum[upd] += 1
    # 没被任何深度观测覆盖的体素（实体内部、截断带之后）先当作“内部”，再由轮廓雕刻判空
    val = torch.where(wsum > 0, tsdf / wsum.clamp(min=1), -torch.ones_like(tsdf))
    val[outside > max(1, S // 12)] = 1.0          # 多个视角判为背景 → 强制空
    vol = val.reshape(res, res, res).cpu().numpy()
    from skimage import measure

    occ = -vol  # 内部为正
    vv_, ff, _, _ = measure.marching_cubes(np.pad(occ, 1, constant_values=-1), 0.0, spacing=(vox.item(),) * 3)
    vv_ = vv_ - vox.item() + (ctr.cpu().numpy() - ext.item() + vox.item() / 2)
    mesh = trimesh.Trimesh(vv_, ff[:, ::-1], process=True)
    # 去掉小碎块
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        mesh = max(parts, key=lambda m: len(m.faces))
    info = {"res": res, "voxel": vox.item(), "n_points": int(P.shape[0]), "conf_thr": float(thr), "n_parts": len(parts)}
    return mesh, info


def canonicalize_from_cameras(mesh: trimesh.Trimesh, extrinsic: np.ndarray) -> tuple[trimesh.Trimesh, dict]:
    """只用视频本身的相机轨迹确定朝向（不看真值）：
    转台视频的相机中心落在与物体竖直轴垂直的圆（或多圈平行圆）上 → 圆所在平面的法向 = 竖直轴；
    相机“上”方向（OpenCV 相机系 -y）的平均值定法向符号；第一帧相机的水平方位 = 正视方向（-Y）。"""
    R = extrinsic[:, :, :3]
    t = extrinsic[:, :, 3]
    C = -np.einsum("sji,sj->si", R, t)                 # 相机中心 = -R^T t
    up_cam = -R[:, 1, :]                               # 相机 -y 轴在世界系的方向
    c0 = C.mean(0)
    _, _, Vt = np.linalg.svd(C - c0)
    up = Vt[2]
    if np.dot(up, up_cam.mean(0)) < 0:
        up = -up
    ctr = mesh.bounds.mean(0)
    f = C[0] - ctr
    f = f - np.dot(f, up) * up
    f /= np.linalg.norm(f)                             # 指向第一帧相机 = 世界 -Y
    x = np.cross(-f, up)                               # 右手系：X = Y × Z，其中 Y = -f
    M = np.stack([x, -f, up])                          # 行向量 = 新坐标轴
    T = np.eye(4)
    T[:3, :3] = M
    T[:3, 3] = -M @ ctr
    out = mesh.copy()
    out.apply_transform(T)
    planarity = float(_[2] / _[0]) if _[0] > 0 else None
    return out, {"up_in_vggt": up.tolist(), "planarity_ratio": planarity, "transform": T.tolist()}


# ----------------------------------------------------------------------------- 主流程
def recon_from_frames(frames: list[Path], out: Path) -> tuple[Path, dict]:
    npz = out / "vggt.npz"
    vinfo = run_vggt(frames, npz)
    d = dict(np.load(npz))
    mesh, finfo = tsdf_fuse(d)
    mesh.export(out / "pred_vggt_frame.obj")
    mesh, cinfo = canonicalize_from_cameras(mesh, d["extrinsic"])
    finfo["canonicalize"] = {k: v for k, v in cinfo.items() if k != "transform"}
    p = out / "pred.obj"
    mesh.export(p)
    # 保存点云便于检查
    return p, {"vggt": vinfo, "fuse": finfo}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--refuse", default=None, help="复用已有 run 目录里的 vggt.npz，只重做融合+定向+评测")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--render-frames", type=int, default=72)
    ap.add_argument("--rot-search", action="store_true", help="评测对齐时再做 24 种轴向旋转搜索（默认只用相机轨迹定向 + ICP）")
    ap.add_argument("--elevs", default="20,-20", help="每个仰角渲染一圈转台视频；单圈 20 看不到底面")
    a = ap.parse_args()
    run = new_run_dir("B_video_recon", f"vggt_f{a.frames}_e{a.elevs.replace(',', '_')}")
    stamp_run(run, {"route": "B_video_recon", "args": vars(a), "blender": blender.blender_version(),
                    "vggt_weights": "facebook/VGGT-1B (CC-BY-NC-4.0)"})
    if a.refuse:
        src = Path(a.refuse)
        rows = {}
        for npz in sorted(src.glob("*/vggt.npz")):
            case = npz.parent.name
            cd = run / case
            cd.mkdir(parents=True, exist_ok=True)
            d = dict(np.load(npz))
            mesh, finfo = tsdf_fuse(d)
            mesh, cinfo = canonicalize_from_cameras(mesh, d["extrinsic"])
            finfo["canonicalize"] = {k: v for k, v in cinfo.items() if k != "transform"}
            mesh.export(cd / "pred.obj")
            ev = eval_case(case, cd / "pred.obj", align=True, save_aligned=cd / "pred_aligned.obj", allow_rot=a.rot_search)
            ev.update({"recon": {"fuse": finfo, "vggt_npz": str(npz)}})
            write_json(cd / "result.json", ev)
            rows[case] = ev
            print(f"{case:14s} CD={ev['chamfer']:.4f} F@.02={ev['fscore@0.02']:.3f} vIoU={ev['vol_iou']:.3f} silIoU={ev['sil_iou_mean']:.3f}")
        write_json(run / "result.json", {"route": "B_video_recon", "args": vars(a), "per_case": rows, "mean": summarize(rows)})
        print("mean", summarize(rows), "->", run)
        return
    if a.video:
        cd = run / "custom"
        frames = extract_frames(Path(a.video), cd / "frames", a.frames)
        p, info = recon_from_frames(frames, cd)
        write_json(run / "result.json", {"custom": info})
        return
    rows = {}
    for case in list_cases(a.cases or "all"):
        t0 = time.time()
        cd = run / case
        elevs = [float(e) for e in a.elevs.split(",")]
        frames = []
        for k, el in enumerate(elevs):
            tt = blender.render_turntable(BENCH_DATA / case / "gt.obj", cd / f"render{k}", frames=a.render_frames,
                                          elev=el, res=512)
            mp4 = cd / f"turntable_elev{el:+.0f}.mp4"
            frames_to_mp4(sorted((cd / f"render{k}").glob("frame_*.png")), mp4)
            shutil.rmtree(cd / f"render{k}", ignore_errors=True)
            frames += extract_frames(mp4, cd / f"frames{k}", a.frames // len(elevs))
        p, info = recon_from_frames(frames, cd)
        ev = eval_case(case, p, align=True, save_aligned=cd / "pred_aligned.obj", allow_rot=a.rot_search)
        ev.update({"recon": info, "seconds": round(time.time() - t0, 1), "render_engine": tt.get("engine")})
        write_json(cd / "result.json", ev)
        rows[case] = ev
        print(f"{case:14s} CD={ev['chamfer']:.4f} F@.02={ev['fscore@0.02']:.3f} vIoU={ev['vol_iou']:.3f} "
              f"silIoU={ev['sil_iou_mean']:.3f} vggt {info['vggt']['seconds']}s peak {info['vggt']['peak_vram_gb']}GB")
    write_json(run / "result.json", {"route": "B_video_recon", "args": vars(a), "per_case": rows,
                                     "mean": summarize(rows)})
    print("mean", summarize(rows), "\n->", run)


if __name__ == "__main__":
    main()
