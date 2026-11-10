# 硬表面：L 形支架（底板+立板+三角筋+通孔）
base = box((1.6, 1.0, 0.12), (0, 0, 0.06))
wall = box((1.6, 0.12, 0.9), (0, 0.44, 0.45))
gus = prism([(0.38, 0.12), (-0.25, 0.12), (0.38, 0.72)], 0.1, plane="YZ", offset=0.0)
part = union(base, wall, gus)
holes = [cylinder(0.11, 0.5, (x, -0.15, 0.06), axis="Z") for x in (-0.5, 0.5)]
holes += [cylinder(0.1, 0.5, (x, 0.44, 0.6), axis="Y") for x in (-0.5, 0.5)]
difference(part, *holes)
