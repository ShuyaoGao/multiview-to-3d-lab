# 有机体：Keenan Crane 的 Spot 奶牛（CC0 1.0）。原始 OBJ 为 Y-up，按默认轴向导入后头朝 -Y（正视图看到脸）
import os
bpy.ops.wm.obj_import(filepath=os.path.join(P6_ROOT, "bench", "assets_src", "spot", "spot", "spot_triangulated.obj"),
                      forward_axis="NEGATIVE_Z", up_axis="Y")
# 导入后头朝 +Y，转 180° 让脸朝向正视图相机（-Y）
for o in [o for o in bpy.context.scene.objects if o.type == "MESH"]:
    rotate(o, (0, 0, 180), pivot=(0, 0, 0))
