# 带空腔的杯子：外轮廓三视图看不到内腔（专门考验“轮廓法”的盲区）
outer = cylinder(0.5, 1.0, (0, 0, 0.5), segments=96)
handle = torus(0.3, 0.06, (0.6, 0, 0.5), axis="Y", seg_major=64, seg_minor=24)
body = union(outer, handle)
inner = cylinder(0.43, 1.0, (0, 0, 0.58), segments=96)
difference(body, inner)
