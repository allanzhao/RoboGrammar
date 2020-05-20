import argparse
from design_search import make_graph, build_normalized_robot
from math import sin, cos, pi
import numpy as np
import os
import pyrobotdesign as rd

def make_affine_translation(t):
    m = np.eye(4)
    m[:3, 3] = t
    return m

def make_affine_rotation(r):
    m = np.eye(4)
    m[:3, :3] = r
    return m

def make_box_mesh():
    positions = [
        -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, # -X face
        -1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, # -Y face
        -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, # -Z face
        1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1,     # +X face
        1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1,     # +Y face
        1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1      # +Z face
    ]
    normals = [
        -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0,     # -X face
        0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0,     # -Y face
        0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1,     # -Z face
        1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,         # +X face
        0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0,         # +Y face
        0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1          # +Z face
    ]
    indices = [
        0, 1, 2, 3, 0, 2,                           # -X face
        4, 5, 6, 7, 4, 6,                           # -Y face
        8, 9, 10, 11, 8, 10,                        # -Z face
        12, 13, 14, 15, 12, 14,                     # +X face
        16, 17, 18, 19, 16, 18,                     # +Y face
        20, 21, 22, 23, 20, 22                      # +Z face
    ]

    return np.array(positions).reshape(-1, 3), \
           np.array(normals).reshape(-1, 3), \
           np.array(indices, dtype=int)

def make_tube_mesh(n_segments):
    positions = []
    normals = []
    indices = []

    # Define two rings of vertices
    for i in range(2):
        for j in range(n_segments):
            theta = (2 * pi) * j / n_segments
            pos = [-1.0 if (i == 0) else 1.0, cos(theta), sin(theta)]
            nor = [0, cos(theta), sin(theta)]
            positions.extend(pos)
            normals.extend(nor)

    # Define triangles
    for j in range(n_segments):
        idx_00 = j
        idx_01 = (j + 1) % n_segments
        idx_10 = n_segments + j
        idx_11 = n_segments + (j + 1) % n_segments
        idx = [idx_00, idx_01, idx_10, idx_11, idx_10, idx_01]
        indices.extend(idx)

    return np.array(positions).reshape(-1, 3), \
           np.array(normals).reshape(-1, 3), \
           np.array(indices, dtype=int)

def make_capsule_end_mesh(n_segments, n_rings):
    positions = []
    indices = []

    # Define rings of vertices
    for i in range(n_rings):
        for j in range(n_segments):
            theta = (2 * pi) * j / n_segments
            phi = (pi / 2) * i / n_rings
            pos = [sin(phi), cos(phi) * cos(theta), cos(phi) * sin(theta)]
            positions.extend(pos)

    # Define zenith vertex
    pos = [1.0, 0.0, 0.0]
    positions.extend(pos)

    # Define triangles for every ring except the last
    for i in range(n_rings - 1):
        for j in range(n_segments):
            idx_00 = i * n_segments + j
            idx_01 = i * n_segments + (j + 1) % n_segments
            idx_10 = (i + 1) * n_segments + j
            idx_11 = (i + 1) * n_segments + (j + 1) % n_segments
            idx = [idx_00, idx_01, idx_10, idx_11, idx_10, idx_01]
            indices.extend(idx)

    # Define triangles for last ring
    for j in range(n_segments):
        idx = [(n_rings - 1) * n_segments + j,
               (n_rings - 1) * n_segments + (j + 1) % n_segments,
               n_rings * n_segments]
        indices.extend(idx)

    # Note: the positions and normals of points on a unit sphere are equal
    return np.array(positions).reshape(-1, 3), \
           np.array(positions).reshape(-1, 3), \
           np.array(indices, dtype=int)

def make_cylinder_end_mesh(n_segments):
    positions = []
    normals = []
    indices = []

    # Define a ring of vertices
    for j in range(n_segments):
        theta = (2 * pi) * j / n_segments
        pos = [0.0, cos(theta), sin(theta)]
        nor = [1.0, 0.0, 0.0]
        positions.extend(pos)
        normals.extend(nor)

    # Define a center vertex
    pos = [0.0, 0.0, 0.0]
    nor = [1.0, 0.0, 0.0]
    positions.extend(pos)
    normals.extend(nor)

    # Define triangles
    for j in range(n_segments):
        idx = [j, (j + 1) % n_segments, n_segments]
        indices.extend(idx)

    return np.array(positions).reshape(-1, 3), \
           np.array(normals).reshape(-1, 3), \
           np.array(indices, dtype=int)

def transform_mesh(positions, normals, indices, tf):
    normal_tf = np.linalg.inv(tf).T
    positions = tf[:3,:3].dot(positions.T) + tf[:3,3:]
    normals = normal_tf[:3,:3].dot(normals.T) + normal_tf[:3,3:]
    return positions.T, normals.T, indices

class ObjDumper:
    def __init__(self, obj_file, mtl_file, n_segments=32, n_rings=8):
        self.obj_file = obj_file
        self.mtl_file = mtl_file

        self.box_mesh = make_box_mesh()
        self.tube_mesh = make_tube_mesh(n_segments)
        self.capsule_end_mesh = make_capsule_end_mesh(n_segments, n_rings)
        self.cylinder_end_mesh = make_cylinder_end_mesh(n_segments)

        # Indices start at 1
        self.pos_count = 1
        self.nor_count = 1

        # All unique colors used
        self.colors = []

    def set_proc_texture_type(self, i):
        # Not implemented yet
        pass

    def set_object_color(self, color):
        color = tuple(color)
        if color not in self.colors:
            self.colors.append(color)

        self.obj_file.write("usemtl mtl{}\n".format(self.colors.index(color)))

    def write_mesh(self, positions, normals, indices):
        for position in positions:
            self.obj_file.write("v {:6f} {:6f} {:6f}\n".format(*position))
        for normal in normals:
            self.obj_file.write("vn {:6f} {:6f} {:6f}\n".format(*normal))
        for tri_indices in indices.reshape(-1, 3):
            args = np.empty(6, dtype=int)
            args[0::2] = tri_indices + self.pos_count
            args[1::2] = tri_indices + self.nor_count
            self.obj_file.write("f {}//{} {}//{} {}//{}\n".format(*args))

        self.pos_count += len(positions)
        self.nor_count += len(positions)

    def draw_tube_based_shape(self, tf, half_length, radius, end_mesh):
        right_tf = tf.dot(make_affine_translation([half_length, 0.0, 0.0])) \
            .dot(np.diag([radius, radius, radius, 1.0]))
        left_tf = tf.dot(make_affine_translation([-half_length, 0.0, 0.0])) \
            .dot(np.diag([-radius, radius, -radius, 1.0]))
        middle_tf = tf.dot(np.diag([half_length, radius, radius, 1.0]))

        self.write_mesh(*transform_mesh(*end_mesh, right_tf))
        self.write_mesh(*transform_mesh(*end_mesh, left_tf))
        self.write_mesh(*transform_mesh(*self.tube_mesh, middle_tf))

    def draw_capsule(self, tf, half_length, radius):
        self.draw_tube_based_shape(tf, half_length, radius,
                                   self.capsule_end_mesh)

    def draw_cylinder(self, tf, half_length, radius):
        self.draw_tube_based_shape(tf, half_length, radius,
                                   self.cylinder_end_mesh)

    def draw_box(self, tf, half_extents):
        self.write_mesh(*transform_mesh(*self.box_mesh,
                                        tf.dot(np.diag([*half_extents, 1.0]))))

    def finish(self):
        for i, color in enumerate(self.colors):
            self.mtl_file.write("newmtl mtl{}\n".format(i))
            self.mtl_file.write("Ka {:3f} {:3f} {:3f}\n".format(*color))
            self.mtl_file.write("Kd {:3f} {:3f} {:3f}\n".format(*color))
            self.mtl_file.write("Ks {:3f} {:3f} {:3f}\n".format(*color))
            self.mtl_file.write("Ns {}\n".format(32))

def dump_sim(sim, dumper):
    for robot_idx in range(sim.get_robot_count()):
        dump_robot(robot_idx, sim, dumper)

    for prop_idx in range(sim.get_prop_count()):
        dump_prop(prop_idx, sim, dumper)

def dump_robot(robot_idx, sim, dumper):
    robot = sim.get_robot(robot_idx)

    for link_idx, link in enumerate(robot.links):
        link_transform = np.zeros((4, 4), order='f')
        sim.get_link_transform(robot_idx, link_idx, link_transform)

        # Draw the link's collision shape
        if link.shape == rd.LinkShape.CYLINDER:
            # Checkerboard (YZ) texture for cylinders
            dumper.set_proc_texture_type(1)
        else:
            # No texture for other shapes
            dumper.set_proc_texture_type(0)
        dumper.set_object_color(link.color)
        if link.shape == rd.LinkShape.CAPSULE:
            dumper.draw_capsule(link_transform, link.length / 2, link.radius)
        elif link.shape == rd.LinkShape.CYLINDER:
            dumper.draw_cylinder(link_transform, link.length / 2, link.radius)
        else:
            raise ValueError("Unexpected link shape")

        # Draw the link's joint
        dumper.set_proc_texture_type(0) # No texture
        dumper.set_object_color(link.joint_color)
        joint_axis_rotation = rd.Quaterniond.from_two_vectors(link.joint_axis,
                                                              [1.0, 0.0, 0.0])
        joint_transform = link_transform \
            .dot(make_affine_translation([-link.length / 2, 0, 0])) \
            .dot(make_affine_rotation(joint_axis_rotation.to_rotation_matrix()))

        if link.parent >= 0:
            parent_link_radius = robot.links[link.parent].radius
            joint_size = min(link.radius, parent_link_radius)
        else:
            joint_size = link.radius
        joint_size *= 1.05

        if link.joint_type == rd.JointType.FREE:
            # Nothing to draw
            pass
        elif link.joint_type == rd.JointType.HINGE:
            dumper.draw_cylinder(joint_transform, joint_size, joint_size)
        elif link.joint_type == rd.JointType.FIXED:
            dumper.draw_box(joint_transform, np.full(3, joint_size))
        else:
            raise ValueError("Unexpected joint type")

def dump_prop(prop_idx, sim, dumper):
    prop = sim.get_prop(prop_idx)
    prop_transform = np.zeros((4, 4), order='f')
    sim.get_prop_transform(prop_idx, prop_transform)

    # Draw the prop's collision shape
    if prop.density == 0.0:
        # Checkerboard (XZ) texture for static shapes
        dumper.set_proc_texture_type(2)
    else:
        # No texture for dynamic shapes
        dumper.set_proc_texture_type(0)
    dumper.set_object_color(prop.color)
    if prop.shape == rd.PropShape.BOX:
        dumper.draw_box(prop_transform, prop.half_extents)
    elif prop.shape == rd.PropShape.HEIGHTFIELD:
        raise NotImplementedError("Heightfields are not supported yet")
    else:
        raise ValueError("Unexpected prop shape")

def main():
    parser = argparse.ArgumentParser(
        description="Export a robot design as a mesh.")
    parser.add_argument("grammar_file", type=str, help="Grammar file (.dot)")
    parser.add_argument("rule_sequence", nargs="+", help="Rule sequence")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output file")
    args = parser.parse_args()

    graphs = rd.load_graphs(args.grammar_file)
    rules = [rd.create_rule_from_graph(g) for g in graphs]
    rule_sequence = [int(s.strip(",")) for s in args.rule_sequence]
    graph = make_graph(rules, rule_sequence)
    robot = build_normalized_robot(graph)

    # Simulation is only used to get link/joint transforms
    sim = rd.BulletSimulation()
    sim.add_robot(robot, [0.0, 0.0, 0.0], rd.Quaterniond(1.0, 0.0, 0.0, 0.0))

    obj_file_name = args.output_file
    mtl_file_name = os.path.splitext(args.output_file)[0] + '.mtl'
    with open(obj_file_name, 'w') as obj_file, \
         open(mtl_file_name, 'w') as mtl_file:
        dumper = ObjDumper(obj_file, mtl_file)
        obj_file.write("mtllib {}\n".format(mtl_file_name))
        dump_sim(sim, dumper)
        dumper.finish()

if __name__ == '__main__':
    main()
