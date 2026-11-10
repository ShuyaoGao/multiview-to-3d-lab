# 角色类（有机体近似）：Q 版人形，A-pose
torso = ellipsoid((0.28, 0.18, 0.38), (0, 0, 0.95))
head = sphere(0.26, (0, 0, 1.55))
nose = sphere(0.05, (0, -0.25, 1.53))
ears = [ellipsoid((0.04, 0.07, 0.09), (x, 0, 1.56)) for x in (-0.26, 0.26)]
arms = [capsule((s * 0.28, 0, 1.2), (s * 0.62, 0, 0.78), 0.075) for s in (-1, 1)]
hands = [sphere(0.1, (s * 0.65, 0, 0.72)) for s in (-1, 1)]
legs = [capsule((s * 0.13, 0, 0.7), (s * 0.17, 0, 0.12), 0.095) for s in (-1, 1)]
feet = [ellipsoid((0.11, 0.18, 0.07), (s * 0.17, -0.07, 0.06)) for s in (-1, 1)]
union(torso, head, nose, *ears, *arms, *hands, *legs, *feet)
