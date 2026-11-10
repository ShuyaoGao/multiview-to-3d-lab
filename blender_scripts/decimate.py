"""无头 Blender：导入 GLB/OBJ → 合并 → Decimate(collapse) 到目标面数 → 按 p6 约定（Z-up 世界坐标）导出 OBJ。
用法：blender -b --factory-startup -P decimate.py -- --in x.glb --out y.obj --faces 200000
（trimesh 的二次误差简化依赖 fast_simplification 原生扩展，被本机“智能应用控制”拦截，所以放到 Blender 里做。）
"""
import argparse
import json
import os
import sys
import time

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--faces", type=int, default=200000)
a = ap.parse_args(argv)
t0 = time.time()
bpy.ops.wm.read_factory_settings(use_empty=True)
inp = os.path.abspath(a.inp)
if inp.lower().endswith((".glb", ".gltf")):
    bpy.ops.import_scene.gltf(filepath=inp)
else:
    bpy.ops.wm.obj_import(filepath=inp, forward_axis="Y", up_axis="Z")
objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
bpy.ops.object.select_all(action="DESELECT")
for o in objs:
    o.select_set(True)
bpy.context.view_layer.objects.active = objs[0]
if len(objs) > 1:
    bpy.ops.object.join()
ob = bpy.context.view_layer.objects.active
n0 = len(ob.data.polygons)
if n0 > a.faces:
    m = ob.modifiers.new("dec", "DECIMATE")
    m.ratio = a.faces / n0
    bpy.ops.object.modifier_apply(modifier="dec")
me = ob.data
me.calc_loop_triangles()
M = ob.matrix_world
with open(os.path.abspath(a.out), "w") as f:
    for v in me.vertices:
        p = M @ v.co
        f.write(f"v {p.x:.6f} {p.y:.6f} {p.z:.6f}\n")
    for t in me.loop_triangles:
        x, y, z = t.vertices
        f.write(f"f {x + 1} {y + 1} {z + 1}\n")
json.dump({"faces_in": n0, "tris_out": len(me.loop_triangles), "seconds": round(time.time() - t0, 1)},
          open(os.path.abspath(a.out) + ".json", "w"))
