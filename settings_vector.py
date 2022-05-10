import os
from mgen import rotation_around_x, rotation_around_y, rotation_around_z
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
import matplotlib.cbook

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from PIL import Image, ImageFont, ImageDraw
import textwrap

# to draw online or not
online_draw = os.environ.get("online_draw", False)

# Delay between draws
refresh_rate = 0.05

if online_draw:
    plt.ion()


def show(pause=refresh_rate):
    plt.pause(pause)
    plt.show()


rotate_methods = {
    'x': rotation_around_x,
    'y': rotation_around_y,
    'z': rotation_around_z
}

axis_numbers = {
    'x': 0,
    'y': 1,
    'z': 2
}

default_hand_length = np.array([[0.4], [0.4], [0.3], [0.3],
                                [1.0], [0.4], [0.3], [0.2],
                                [1.0], [0.4], [0.3], [0.2],
                                [1.0], [0.4], [0.3], [0.2],
                                [1.0], [0.4], [0.3], [0.2]])

first_base_vector_number = 5
second_base_vector_number = 17


def set_standard_size(points):
    points = np.copy(points)
    base_point = points[:1]
    points = points[1:]
    cos_list = get_cos_list(points - base_point)
    points = hand_restore(cos_list)
    points = np.append(base_point, points + base_point, axis=0)
    return points


def get_cos_list(points):
    array_vector = points[1:] - points[:-1]
    array_vector = np.vstack((points[0], array_vector))
    array_vector[::4] = points[::4]
    length_vector = get_length_2d(array_vector, np.array([0, 0, 0]))
    cos = array_vector / np.reshape(length_vector, (len(length_vector), 1))
    return cos


def hand_restore(cos_list):
    restore_vectors = cos_list * default_hand_length
    # array_vector = restore_vectors[1:] + restore_vectors[:-1]
    # array_vector = np.vstack((restore_vectors[0], array_vector))
    # array_vector[::4] = restore_vectors[::4]
    for i in range(0, 3):
        restore_vectors[i + 1::4] += restore_vectors[i::4]
    return restore_vectors


def get_length_2d(array1, array2):
    vectors = array1[:, np.newaxis] - array2
    length = np.sqrt(np.sum(np.power(vectors, 2), axis=2)).T.ravel()
    return length


def set_standard_position(points):
    points = np.copy(points)
    base_point = points[0]
    vectors = points - base_point
    rotated_vectors = rotate_vectors(vectors, first_base_vector_number, 'z', 'y')
    rotated_vectors = rotate_vectors(rotated_vectors, first_base_vector_number, 'x', 'y')
    rotated_vectors = rotate_vectors(rotated_vectors, second_base_vector_number, 'y', 'x')
    return rotated_vectors


def rotate_vectors(vectors, rotate_vector_number, ax_rotate, to_ax, around_point=None, alfa=None):
    vectors = np.copy(vectors)
    if around_point is not None:
        base_vector = around_point
        rotate_vector = np.copy(around_point)
    else:
        base_vector = np.copy(vectors[rotate_vector_number])
        rotate_vector = np.copy(vectors[rotate_vector_number])
    base_vector[axis_numbers[ax_rotate]] = 0
    if alfa is None:
        alfa = np.arccos(get_cos(base_vector))[axis_numbers[to_ax]]
        if np.isnan(alfa):
            return vectors
        matrix = rotate_methods[ax_rotate](alfa)
        value = np.delete(matrix.dot(rotate_vector), [axis_numbers[ax_rotate], axis_numbers[to_ax]])
        if np.abs(value) > 10 ** -7:
            matrix = rotate_methods[ax_rotate](-alfa)
        rotated_vectors = np.array([matrix.dot(vector) for vector in vectors])
    else:
        matrix = rotate_methods[ax_rotate](alfa)
        vectors -= around_point
        rotated_vectors = np.array([matrix.dot(vector) for vector in vectors])
        rotated_vectors += around_point

    min_value = 10 ** -14
    rotated_vectors[np.abs(rotated_vectors) < min_value] = 0.0
    return rotated_vectors


def get_numpy_points(landmarks):
    if type(landmarks) == np.ndarray:
        points = np.copy(landmarks)
    elif landmarks is not None:
        points = np.array([[point.x, point.y, point.z] for point in list(landmarks.landmark)], dtype='float16')
    else:
        points = None
    return points


def get_cos(vector, base_point=np.array([0, 0, 0])):
    vector = vector - base_point[:len(vector)]
    length_vector = get_length(vector)
    cos = vector / length_vector
    return cos


def get_length(vector, base_point=np.array([0, 0, 0])):
    length = np.sqrt(np.sum(np.power(vector - base_point[:len(vector)], 2)))
    return length


def get_length_between_points(points1, points2):
    array_length = list()
    for p1 in points1:
        for p2 in points2:
            array_length.append(get_length(p1, p2))
    return np.array(array_length)


def draw_hand(points, base_point=(0, 0, 0), color='red', refresh=False, draw_axis=1):
    """
    Draw hand by MediaPipe hand model points.
    :param points: 21 point from MediaPipe hand model
    :param base_point: first point from MediaPipe hand model or any point
    :param color: color to draw lines
    :param refresh: clear draw space
    :param draw_axis: 0 - do not draw axis; 1 - draw axis from (0, 0, 0); 2 - draw axis from (0, 0, 1)
    """
    points = np.copy(points) + base_point
    if refresh:
        set_axis(draw_axis)
    if len(points) == 20:
        points = np.vstack((np.array([0, 0, 0]), points))

    draw_vector(points[1], points[0], color)
    draw_vector(points[5], points[0], color)
    draw_vector(points[17], points[0], color)

    draw_vector(points[1], points[2], color)
    draw_vector(points[2], points[3], color)
    draw_vector(points[3], points[4], color)

    draw_vector(points[5], points[6], color)
    draw_vector(points[6], points[7], color)
    draw_vector(points[7], points[8], color)

    draw_vector(points[9], points[10], color)
    draw_vector(points[10], points[11], color)
    draw_vector(points[11], points[12], color)

    draw_vector(points[13], points[14], color)
    draw_vector(points[14], points[15], color)
    draw_vector(points[15], points[16], color)

    draw_vector(points[17], points[18], color)
    draw_vector(points[18], points[19], color)
    draw_vector(points[19], points[20], color)

    draw_vector(points[5], points[9], color)
    draw_vector(points[9], points[13], color)
    draw_vector(points[13], points[17], color)

    draw_vector(points[4], points[8], color)

    if refresh and plt.isinteractive():
        show(refresh_rate)


def draw_vector(first_point, second_point=(0, 0, 0), color='red', refresh=False, draw_axis=1):
    """
    Draw line between two points.
    :param first_point: first point coordinates
    :param second_point: second point coordinates
    :param color: color to draw line
    :param refresh: clear draw space
    :param draw_axis: 0 - do not draw axis; 1 - draw axis from (0, 0, 0); 2 - draw axis from (0, 0, 1)
    """
    if refresh:
        set_axis(draw_axis)

    ax.plot(np.linspace(first_point[0], second_point[0]),
            np.linspace(first_point[1], second_point[1]),
            np.linspace(first_point[2], second_point[2]),
            color=color)
    if refresh and plt.isinteractive():
        show(refresh_rate)


def get_clear_space():
    """
    Take a clear space
    :return: coordinate chart
    """
    ax = Axes3D(fig)
    ax.view_init(elev=-60, azim=-80)
    return ax


def get_standard_axis():
    """
    sets standard axes
    :return: coordinate chart
    """
    # ax = fig.add_subplot(projection='3d')
    ax = Axes3D(fig)
    ax.plot(np.linspace(0, 1), np.linspace(0, 0), np.linspace(0, 0), color='black')
    ax.plot(np.linspace(0, 0), np.linspace(0, 1), np.linspace(0, 0), color='orange')
    ax.plot(np.linspace(0, 0), np.linspace(0, 0), np.linspace(0, 1), color='blue')
    ax.view_init(elev=-60, azim=-80)
    return ax


def get_z_up_axis():
    ax = Axes3D(fig)
    ax.plot(np.linspace(0, 1), np.linspace(0, 0), np.linspace(1, 1), color='black')
    ax.plot(np.linspace(0, 0), np.linspace(0, 1), np.linspace(1, 1), color='orange')
    ax.plot(np.linspace(0, 0), np.linspace(0, 0), np.linspace(1, 2), color='blue')
    ax.view_init(elev=-60, azim=-80)
    return ax


def set_axis(draw_axis, refresh=True):
    global ax, fig
    if refresh:
        fig.clf()
    ax = axis[draw_axis]()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


axis = {
    0: get_clear_space,
    1: get_standard_axis,
    2: get_z_up_axis
}

fig = plt.figure()
ax = get_clear_space()

if online_draw:
    set_axis(1)

