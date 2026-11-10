"""无头 Blender 渲染：标准正交三视图（ortho3）或转台视频帧（turntable）。

ortho3：  blender -b --factory-startup -P render_views.py -- --mesh m.obj --outdir d --mode ortho3 [--res 512]
turntable：... --mode turntable --frames 36 --elev 20 --res 512 [--texture noise]
网格按 p6 约定是 Z-up 世界坐标 OBJ（build_mesh.py 导出的格式）。
正交窗口 = [-1,1]^2（ortho_scale=2），与 src/p6/views.py 的 numpy 轮廓完全一致。
"""
import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Vector


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--mode", default="ortho3", choices=["ortho3", "turntable"])
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--frames", type=int, default=36)
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--dist", type=float, default=3.0)
    ap.add_argument("--lens", type=float, default=50.0)
    ap.add_argument("--texture", default="noise", choices=["noise", "none"])
    ap.add_argument("--engine", default="auto")
    return ap.parse_args(argv)


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_mesh(path):
    bpy.ops.wm.obj_import(filepath=path, forward_axis="Y", up_axis="Z")
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    for o in objs:
        for p in o.data.polygons:
            p.use_smooth = False
    return objs


def white_world(light_strength=0.35):
    """相机直接看到纯白背景；给物体打光的环境光只有 light_strength 的灰（Light Path: Is Camera Ray）。"""
    w = bpy.data.worlds.new("w")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    out = nt.nodes.get("World Output")
    bg = nt.nodes.get("Background")
    bg.inputs[0].default_value = (1, 1, 1, 1)
    bg.inputs[1].default_value = light_strength
    white = nt.nodes.new("ShaderNodeBackground")
    white.inputs[0].default_value = (1, 1, 1, 1)
    white.inputs[1].default_value = 1.0
    lp = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs[0])
    nt.links.new(bg.outputs[0], mix.inputs[1])
    nt.links.new(white.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    bpy.context.scene.view_settings.view_transform = "Standard"
    return w


def add_cam(name, loc, rot_euler, ortho=True, scale=2.0, lens=50.0):
    cd = bpy.data.cameras.new(name)
    if ortho:
        cd.type = "ORTHO"
        cd.ortho_scale = scale
    else:
        cd.lens = lens
    cd.clip_start = 0.01
    cd.clip_end = 100
    co = bpy.data.objects.new(name, cd)
    bpy.context.scene.collection.objects.link(co)
    co.location = loc
    co.rotation_euler = rot_euler
    return co


def look_at(obj, target=(0, 0, 0)):
    d = Vector(target) - obj.location
    obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def ortho3(a):
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sh = sc.display.shading
    sh.light = "STUDIO"
    sh.color_type = "SINGLE"
    sh.single_color = (0.75, 0.75, 0.75)
    sh.show_cavity = True
    sh.cavity_type = "BOTH"
    sh.show_object_outline = True
    sh.object_outline_color = (0, 0, 0)
    sc.render.film_transparent = True
    sc.render.resolution_x = sc.render.resolution_y = a.res
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    cams = {
        "front": ((0, -10, 0), (math.radians(90), 0, 0)),
        "side": ((10, 0, 0), (math.radians(90), 0, math.radians(90))),
        "top": ((0, 0, 10), (0, 0, 0)),
    }
    out = {}
    for name, (loc, rot) in cams.items():
        cam = add_cam(name, loc, rot, True, 2.0)
        sc.camera = cam
        p = os.path.join(a.outdir, f"{name}_rgba.png")
        sc.render.filepath = p
        bpy.ops.render.render(write_still=True)
        out[name] = p
    return out


def noise_material():
    m = bpy.data.materials.new("noise")
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    n1 = nt.nodes.new("ShaderNodeTexNoise")
    n1.inputs["Scale"].default_value = 12.0
    n1.inputs["Detail"].default_value = 8.0
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 25.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[1].color = (0.55, 0.45, 0.35, 1)  # 避免纯白斑点被当成背景
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    nt.links.new(tc.outputs["Object"], n1.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], vor.inputs["Vector"])
    nt.links.new(n1.outputs["Color"], mix.inputs[6])
    nt.links.new(vor.outputs["Distance"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], mix.inputs[7])
    mix.inputs[0].default_value = 0.5
    nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.8
    return m


def turntable(a, objs):
    sc = bpy.context.scene
    eng = a.engine
    if eng == "auto":
        eng = "BLENDER_EEVEE" if "BLENDER_EEVEE" in [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items] else "BLENDER_EEVEE_NEXT"
    sc.render.engine = eng
    if eng == "CYCLES":
        sc.cycles.samples = 32
    white_world()
    if a.texture == "noise":
        mat = noise_material()
        for o in objs:
            o.data.materials.clear()
            o.data.materials.append(mat)
    # 太阳光（随相机固定，避免高光漂移太大）
    ld = bpy.data.lights.new("sun", "SUN")
    ld.energy = 3.0
    lo = bpy.data.objects.new("sun", ld)
    sc.collection.objects.link(lo)
    lo.rotation_euler = (math.radians(40), 0, math.radians(30))
    sc.render.film_transparent = False
    sc.render.resolution_x = sc.render.resolution_y = a.res
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    cam = add_cam("cam", (0, -a.dist, 0), (0, 0, 0), ortho=False, lens=a.lens)
    sc.camera = cam
    poses = []
    el = math.radians(a.elev)
    for i in range(a.frames):
        az = 2 * math.pi * i / a.frames
        # az=0 对应正视图方向（相机在 -Y）
        cam.location = (a.dist * math.cos(el) * math.sin(az), -a.dist * math.cos(el) * math.cos(az), a.dist * math.sin(el))
        look_at(cam)
        bpy.context.view_layer.update()
        p = os.path.join(a.outdir, f"frame_{i:03d}.png")
        sc.render.filepath = p
        bpy.ops.render.render(write_still=True)
        poses.append({"file": os.path.basename(p), "c2w_blender": [list(r) for r in cam.matrix_world]})
    sensor = cam.data.sensor_width
    f_px = a.lens / sensor * a.res
    return {"engine": eng, "frames": poses, "intrinsics": {"fx": f_px, "fy": f_px, "cx": a.res / 2, "cy": a.res / 2,
                                                       "w": a.res, "h": a.res}}


def main():
    a = args()
    a.outdir = os.path.abspath(a.outdir)
    a.mesh = os.path.abspath(a.mesh)
    os.makedirs(a.outdir, exist_ok=True)
    reset()
    objs = import_mesh(a.mesh)
    if a.mode == "ortho3":
        info = {"views": ortho3(a)}
    else:
        info = turntable(a, objs)
    info["blender"] = bpy.app.version_string
    info["mode"] = a.mode
    json.dump(info, open(os.path.join(a.outdir, f"render_{a.mode}.json"), "w"), indent=2)


main()
