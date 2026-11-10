"""p6h：在 Blender 内部使用的极简 CSG 建模助手（真值形体和 LLM 生成的脚本都用它）。

所有尺寸单位 = 世界单位；Z 朝上；正视图看 XZ 平面，侧视图看 YZ 平面，顶视图看 XY 平面。
函数都返回 bpy Object，布尔运算直接应用（EXACT 求解器），工具体自动删除。

API 一览（LLM 提示词里也会给出这份清单）：
  clear()
  box(size=(sx,sy,sz), center=(x,y,z))
  cylinder(r, h, center=(0,0,0), axis='Z', segments=64)
  cone(r1, r2, h, center=(0,0,0), axis='Z', segments=64)          # r1 在负端, r2 在正端
  sphere(r, center=(0,0,0), segments=48)
  ellipsoid(radii=(rx,ry,rz), center=(0,0,0), segments=48)
  torus(R, r, center=(0,0,0), axis='Z', seg_major=64, seg_minor=24, arc_deg=360)
  capsule(p0, p1, r, segments=32)
  prism(points, depth, plane='XZ', offset=0.0)
        # 把 2D 多边形沿平面法向挤出：plane='XZ' 时 points 是 (x,z)，沿 Y 挤出 [offset-depth/2, offset+depth/2]
        # plane='YZ' 时 points 是 (y,z) 沿 X 挤出；plane='XY' 时 points 是 (x,y) 沿 Z 挤出
  revolve(profile, axis='Z', center=(0,0,0), segments=64)   # profile=[(r,z),...] 绕轴旋转成回转体
  rotate(obj, deg=(rx,ry,rz), pivot=None)  move(obj, (dx,dy,dz))  copy(obj)  mirror(obj, axis='X')
  union(a, *others)  difference(a, *tools)  intersect(a, b)
最后不需要手动合并：运行器会把场景里所有网格对象做布尔并集后导出。
"""
from __future__ import annotations

import math

import bpy
from mathutils import Euler, Matrix, Vector

_AX = {"X": 0, "Y": 1, "Z": 2}


# ---------------------------------------------------------------- 基础
def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for m in list(bpy.data.meshes):
        bpy.data.meshes.remove(m)


def _obj(name, verts, faces):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in verts], [], [tuple(f) for f in faces])
    me.validate(clean_customdata=False)
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def _orient(verts, axis, center):
    """局部几何以 Z 为轴建好，再转到指定 axis 并平移到 center。"""
    axis = axis.upper()
    out = []
    for x, y, z in verts:
        if axis == "Z":
            p = (x, y, z)
        elif axis == "X":
            p = (z, x, y)
        elif axis == "Y":
            p = (y, z, x)
        else:
            raise ValueError("axis 必须是 X/Y/Z")
        out.append((p[0] + center[0], p[1] + center[1], p[2] + center[2]))
    return out


def _fix_normals(ob):
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()
    return ob


# ---------------------------------------------------------------- 图元
def box(size=(1, 1, 1), center=(0, 0, 0), name="box"):
    sx, sy, sz = (s / 2 for s in size)
    cx, cy, cz = center
    v = [(cx + dx * sx, cy + dy * sy, cz + dz * sz) for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)]
    f = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return _fix_normals(_obj(name, v, f))


def _ring_solid(rings, segments, name, axis, center):
    """rings: [(r, z)] 自下而上；r=0 的端点收成极点，否则加端盖。"""
    verts, faces, idx = [], [], []
    for r, z in rings:
        if r <= 1e-9:
            idx.append([len(verts)])
            verts.append((0.0, 0.0, z))
        else:
            start = len(verts)
            idx.append(list(range(start, start + segments)))
            for i in range(segments):
                a = 2 * math.pi * i / segments
                verts.append((r * math.cos(a), r * math.sin(a), z))
    for k in range(len(idx) - 1):
        A, B = idx[k], idx[k + 1]
        if len(A) == 1 and len(B) == 1:
            continue
        for i in range(segments):
            j = (i + 1) % segments
            if len(A) == 1:
                faces.append((A[0], B[i], B[j]))
            elif len(B) == 1:
                faces.append((A[i], A[j], B[0]))
            else:
                faces.append((A[i], A[j], B[j], B[i]))
    if len(idx[0]) > 1:
        faces.append(tuple(reversed(idx[0])))
    if len(idx[-1]) > 1:
        faces.append(tuple(idx[-1]))
    return _fix_normals(_obj(name, _orient(verts, axis, center), faces))


def cylinder(r, h, center=(0, 0, 0), axis="Z", segments=64, name="cyl"):
    return _ring_solid([(r, -h / 2), (r, h / 2)], segments, name, axis, center)


def cone(r1, r2, h, center=(0, 0, 0), axis="Z", segments=64, name="cone"):
    return _ring_solid([(r1, -h / 2), (r2, h / 2)], segments, name, axis, center)


def revolve(profile, axis="Z", center=(0, 0, 0), segments=64, name="rev"):
    return _ring_solid([(float(r), float(z)) for r, z in profile], segments, name, axis, center)


def ellipsoid(radii=(1, 1, 1), center=(0, 0, 0), segments=48, name="ellipsoid"):
    n_lat = max(8, segments // 2)
    rings = [(math.sin(math.pi * i / n_lat), -math.cos(math.pi * i / n_lat)) for i in range(n_lat + 1)]
    rings[0] = (0.0, -1.0)
    rings[-1] = (0.0, 1.0)
    ob = _ring_solid(rings, segments, name, "Z", (0, 0, 0))
    rx, ry, rz = radii
    for v in ob.data.vertices:
        v.co = Vector((v.co.x * rx + center[0], v.co.y * ry + center[1], v.co.z * rz + center[2]))
    ob.data.update()
    return ob


def sphere(r, center=(0, 0, 0), segments=48, name="sphere"):
    return ellipsoid((r, r, r), center, segments, name)


def torus(R, r, center=(0, 0, 0), axis="Z", seg_major=64, seg_minor=24, arc_deg=360.0, name="torus"):
    full = abs(arc_deg - 360.0) < 1e-6
    n_maj = seg_major if full else max(2, int(seg_major * arc_deg / 360.0)) + 1
    verts, faces = [], []
    for i in range(n_maj):
        a = (2 * math.pi * i / seg_major) if full else math.radians(arc_deg) * i / (n_maj - 1)
        for j in range(seg_minor):
            b = 2 * math.pi * j / seg_minor
            rr = R + r * math.cos(b)
            verts.append((rr * math.cos(a), rr * math.sin(a), r * math.sin(b)))
    rng = n_maj if full else n_maj - 1
    for i in range(rng):
        i2 = (i + 1) % n_maj
        for j in range(seg_minor):
            j2 = (j + 1) % seg_minor
            faces.append((i * seg_minor + j, i2 * seg_minor + j, i2 * seg_minor + j2, i * seg_minor + j2))
    if not full:
        faces.append(tuple(range(seg_minor)))
        faces.append(tuple((n_maj - 1) * seg_minor + j for j in reversed(range(seg_minor))))
    return _fix_normals(_obj(name, _orient(verts, axis, center), faces))


def capsule(p0, p1, r, segments=32, name="capsule"):
    p0, p1 = Vector(p0), Vector(p1)
    d = p1 - p0
    L = d.length
    n_cap = max(4, segments // 4)
    rings = []
    for i in range(n_cap + 1):
        t = math.pi / 2 * i / n_cap
        rings.append((r * math.sin(t), -r * math.cos(t)))
    for i in range(n_cap + 1):
        t = math.pi / 2 * i / n_cap
        rings.append((r * math.cos(t), L + r * math.sin(t)))
    rings[0] = (0.0, -r)
    rings[-1] = (0.0, L + r)
    ob = _ring_solid(rings, segments, name, "Z", (0, 0, 0))
    rot = Vector((0, 0, 1)).rotation_difference(d.normalized() if L > 0 else Vector((0, 0, 1))).to_matrix().to_4x4()
    ob.data.transform(Matrix.Translation(p0) @ rot)
    ob.data.update()
    return ob


def prism(points, depth, plane="XZ", offset=0.0, name="prism"):
    pts = [(float(a), float(b)) for a, b in points]
    # 去掉首尾重复点
    if len(pts) > 2 and abs(pts[0][0] - pts[-1][0]) < 1e-12 and abs(pts[0][1] - pts[-1][1]) < 1e-12:
        pts = pts[:-1]
    n = len(pts)
    plane = plane.upper()
    lo, hi = offset - depth / 2, offset + depth / 2

    def P(a, b, w):
        if plane == "XZ":
            return (a, w, b)
        if plane == "YZ":
            return (w, a, b)
        if plane == "XY":
            return (a, b, w)
        raise ValueError("plane 必须是 XZ/YZ/XY")

    verts = [P(a, b, lo) for a, b in pts] + [P(a, b, hi) for a, b in pts]
    faces = [tuple(range(n)), tuple(range(2 * n - 1, n - 1, -1))]
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
    ob = _obj(name, verts, faces)
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(ob.data)
    # 凹多边形的 ngon 端盖三角化，避免布尔出错
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()
    return ob


# ---------------------------------------------------------------- 变换
def bake(obj):
    """把对象的 location/rotation/scale 烘焙进网格，使之后的操作都在世界坐标里进行。"""
    M = obj.matrix_world.copy()
    if M != Matrix.Identity(4):
        if obj.data.users > 1:
            obj.data = obj.data.copy()
        obj.data.transform(M)
        obj.matrix_world = Matrix.Identity(4)
        obj.data.update()
    return obj


def rotate(obj, deg=(0, 0, 0), pivot=None):
    bake(obj)
    piv = Vector(pivot) if pivot is not None else _centroid(obj)
    R = Euler(tuple(math.radians(d) for d in deg), "XYZ").to_matrix().to_4x4()
    obj.data.transform(Matrix.Translation(piv) @ R @ Matrix.Translation(-piv))
    obj.data.update()
    return obj


def move(obj, d):
    bake(obj)
    obj.data.transform(Matrix.Translation(Vector(d)))
    obj.data.update()
    return obj


def scale(obj, s, pivot=(0, 0, 0)):
    bake(obj)
    s = (s, s, s) if isinstance(s, (int, float)) else s
    piv = Vector(pivot)
    S = Matrix.Diagonal((s[0], s[1], s[2], 1.0))
    obj.data.transform(Matrix.Translation(piv) @ S @ Matrix.Translation(-piv))
    if s[0] * s[1] * s[2] < 0:
        obj.data.flip_normals()
    obj.data.update()
    return obj


def copy(obj, name=None):
    ob = obj.copy()
    ob.data = obj.data.copy()
    if name:
        ob.name = name
    bpy.context.scene.collection.objects.link(ob)
    return ob


def mirror(obj, axis="X"):
    """返回关于该坐标平面（axis=0）镜像的副本。"""
    c = copy(obj)
    s = [1, 1, 1]
    s[_AX[axis.upper()]] = -1
    return scale(c, tuple(s))


def _centroid(obj):
    vs = obj.data.vertices
    if not vs:
        return Vector((0, 0, 0))
    acc = Vector((0, 0, 0))
    for v in vs:
        acc += v.co
    return acc / len(vs)


# ---------------------------------------------------------------- 布尔
def _apply_bool(a, b, op):
    bake(a)
    bake(b)
    mod = a.modifiers.new(name="bool", type="BOOLEAN")
    mod.operation = op
    mod.object = b
    mod.solver = "EXACT"
    dg = bpy.context.evaluated_depsgraph_get()
    ev = a.evaluated_get(dg)
    new = bpy.data.meshes.new_from_object(ev)
    a.modifiers.remove(mod)
    old = a.data
    a.data = new
    if old.users == 0:
        bpy.data.meshes.remove(old)
    bpy.data.objects.remove(b, do_unlink=True)
    return a


def union(a, *others):
    for b in others:
        a = _apply_bool(a, b, "UNION")
    return a


def difference(a, *tools):
    for b in tools:
        a = _apply_bool(a, b, "DIFFERENCE")
    return a


def intersect(a, b):
    return _apply_bool(a, b, "INTERSECT")


__all__ = [
    "clear", "box", "cylinder", "cone", "revolve", "ellipsoid", "sphere", "torus", "capsule", "prism",
    "bake", "rotate", "move", "scale", "copy", "mirror", "union", "difference", "intersect", "math",
]
