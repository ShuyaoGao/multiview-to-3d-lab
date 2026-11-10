# 硬表面回转体：带法兰的阶梯轴 + 6 个螺栓孔 + 键槽
prof = [(0, -0.9), (0.18, -0.9), (0.18, -0.35), (0.6, -0.35), (0.6, -0.2), (0.3, -0.2),
        (0.3, 0.1), (0.22, 0.1), (0.22, 0.9), (0, 0.9)]
shaft = revolve(prof, axis="Z", segments=96)
holes = [cylinder(0.05, 0.4, (0.45 * math.cos(math.radians(a)), 0.45 * math.sin(math.radians(a)), -0.275))
         for a in range(0, 360, 60)]
key = box((0.08, 0.5, 0.35), (0, 0.22, 0.75))
difference(shaft, *holes, key)
