# 硬表面：直齿轮（18 齿）+ 轮毂 + 轴孔 + 键槽 + 减重孔
n = 18
pts = []
for i in range(n):
    a0 = 2 * math.pi * i / n
    da = 2 * math.pi / n
    for f, r in ((0.0, 0.66), (0.18, 0.66), (0.32, 0.8), (0.68, 0.8), (0.82, 0.66)):
        a = a0 + f * da
        pts.append((r * math.cos(a), r * math.sin(a)))
g = prism(pts, 0.25, plane="XY", offset=0.0)
hub = cylinder(0.28, 0.5, (0, 0, 0.0))
g = union(g, hub)
bore = cylinder(0.12, 1.0, (0, 0, 0))
key = box((0.06, 0.08, 1.0), (0.13, 0, 0))
lights = [cylinder(0.09, 1.0, (0.47 * math.cos(math.radians(a)), 0.47 * math.sin(math.radians(a)), 0)) for a in (45, 135, 225, 315)]
difference(g, bore, key, *lights)
