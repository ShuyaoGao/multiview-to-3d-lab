# 有机体：Keenan Crane 的 Bob 橡皮鸭（CC0 1.0，亏格 1）
import os
bpy.ops.wm.obj_import(filepath=os.path.join(P6_ROOT, "bench", "assets_src", "bob", "bob", "bob_tri.obj"),
                      forward_axis="NEGATIVE_Z", up_axis="Y")
