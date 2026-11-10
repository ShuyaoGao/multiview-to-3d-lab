"""在无头 Blender 里执行一段建模脚本，把场景所有网格并成一个实体，导出 OBJ。

用法：blender -b --factory-startup -P build_mesh.py -- --script s.py --out m.obj --meta meta.json
      [--import-mesh x.obj]  （可选：先导入一个外部网格当作场景内容，用于 Spot 这类资产）
脚本里可以直接用 bpy 和 p6h 的全部函数（已 from p6h import *）。
"""
import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402

import p6h  # noqa: E402


def mesh_objects():
    return [o for o in bpy.context.scene.objects if o.type == "MESH" and len(o.data.polygons) > 0]


def finalize(union_mode="bool"):
    objs = mesh_objects()
    if not objs:
        return None, "场景里没有任何网格对象"
    # 先把对象变换烘焙进网格（脚本可能直接改了 location/rotation/scale）
    for o in objs:
        if o.matrix_world != o.matrix_world.Identity(4):
            o.data = o.data.copy()
            o.data.transform(o.matrix_world)
            o.matrix_world.identity()
    base = objs[0]
    note = ""
    if len(objs) > 1:
        if union_mode == "bool":
            try:
                for o in objs[1:]:
                    base = p6h._apply_bool(base, o, "UNION")
            except Exception as e:  # noqa
                note = f"布尔并集失败，退回 join: {e}"
                union_mode = "join"
        if union_mode == "join":
            objs = mesh_objects()
            base = objs[0]
            for o in objs[1:]:
                bm_data = o.data
                base.data = _join_meshes(base.data, bm_data)
                bpy.data.objects.remove(o, do_unlink=True)
    return base, note


def _join_meshes(a, b):
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(a)
    bm.from_mesh(b)
    me = bpy.data.meshes.new("joined")
    bm.to_mesh(me)
    bm.free()
    return me


def export_obj(obj, path):
    me = obj.data
    me.calc_loop_triangles()
    M = obj.matrix_world
    with open(path, "w", encoding="utf-8") as f:
        f.write("# p6 build_mesh export, Z-up world coords\n")
        for v in me.vertices:
            p = M @ v.co
            f.write(f"v {p.x:.6f} {p.y:.6f} {p.z:.6f}\n")
        for t in me.loop_triangles:
            a, b, c = t.vertices
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")
    return len(me.vertices), len(me.loop_triangles)


def import_obj(path, axis_forward="NEGATIVE_Z", axis_up="Y"):
    bpy.ops.wm.obj_import(filepath=path, forward_axis=axis_forward, up_axis=axis_up)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--union", default="bool", choices=["bool", "join"])
    a = ap.parse_args(argv)
    meta = {"blender": bpy.app.version_string, "script": a.script, "ok": False}
    t0 = time.time()
    p6h.clear()
    code = open(a.script, encoding="utf-8").read()
    g = {"__name__": "__p6_user__", "bpy": bpy,
         "P6_ROOT": os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}
    g.update({k: getattr(p6h, k) for k in p6h.__all__})
    try:
        exec(compile(code, os.path.basename(a.script), "exec"), g)
    except Exception:
        meta["error"] = traceback.format_exc(limit=6)
        meta["stage"] = "script"
        json.dump(meta, open(a.meta, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        sys.exit(2)
    meta["n_objects_before_union"] = len(mesh_objects())
    try:
        obj, note = finalize(a.union)
        if obj is None:
            raise RuntimeError(note)
        nv, nf = export_obj(obj, a.out)
        meta.update({"ok": True, "verts": nv, "tris": nf, "union_note": note})
    except Exception:
        meta["error"] = traceback.format_exc(limit=6)
        meta["stage"] = "finalize"
        json.dump(meta, open(a.meta, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        sys.exit(3)
    meta["seconds"] = round(time.time() - t0, 2)
    json.dump(meta, open(a.meta, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


main()
