#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches


plt.rcParams['figure.figsize'] = (4, 4)
plt.rcParams['font.size'] = 10

lw = 0.6
lw_highlight = 3 * lw
plt.rcParams['lines.linewidth'] = lw
plt.rcParams['lines.solid_capstyle'] = 'round'
plt.rcParams['patch.linewidth'] = lw

# legend
plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.borderaxespad'] = 1.6

# set the color cycle used by ax.plot():
plt.rcParams['axes.prop_cycle'] = "cycler('color', ['black'])"
plt.rcParams['patch.edgecolor'] = 'black'


class Cube:
    def __init__(self, center_xy, side_half_length):
        self._lower_left_xy = [center_xy[0] - side_half_length,
                               center_xy[1] - side_half_length]
        self._width = 2 * side_half_length
        self._height = 2 * side_half_length

    def draw(self, ax):
        ax.add_artist(patches.Rectangle(self._lower_left_xy,
                                        self._width, self._height, fill=False))

    def highlight_side(self, ax, color, label):
        x0 = self._lower_left_xy[0]
        y0 = self._lower_left_xy[1]
        y1 = y0 + self._height
        return ax.plot([x0, x0], [y0, y1], lw=lw_highlight, color=color,
                       label=label)

    def connect_to_cube(self, ax, cube, this_side_only='both'):
        if (this_side_only not in ['both', 'left', 'right']):
            print("Expect 'this_side_only' from {'both', 'left', 'right'}.")
            this_side_only = 'both'
        x0 = self._lower_left_xy[0]
        x1 = x0 + self._width
        y0 = self._lower_left_xy[1]
        y1 = y0 + self._height
        cx0 = cube._lower_left_xy[0]
        cx1 = cx0 + cube._width
        cy0 = cube._lower_left_xy[1]
        cy1 = cy0 + cube._height
        if (this_side_only in ['both', 'left']):
            ax.plot([x0, cx0], [y0, cy0])
            ax.plot([x0, cx0], [y1, cy1])
        if (this_side_only in ['both', 'right']):
            ax.plot([x1, cx1], [y0, cy0])
            ax.plot([x1, cx1], [y1, cy1])

    def connect_to_sphere(self, ax, sphere):
        x0 = self._lower_left_xy[0]
        x1 = x0 + self._width
        y0 = self._lower_left_xy[1]
        y1 = y0 + self._height
        sin45 = cos45 = np.cos(np.pi/4)
        sx0 = sphere._center_xy[0] - cos45 * sphere._radius
        sx1 = sphere._center_xy[0] + cos45 * sphere._radius
        sy0 = sphere._center_xy[1] - sin45 * sphere._radius
        sy1 = sphere._center_xy[1] + sin45 * sphere._radius
        ax.plot([x0, sx0], [y0, sy0])
        ax.plot([x1, sx1], [y0, sy0])
        ax.plot([x0, sx0], [y1, sy1])
        ax.plot([x1, sx1], [y1, sy1])


class Sphere:
    def __init__(self, center_xy, radius):
        self._center_xy = center_xy
        self._radius = radius

    def draw(self, ax):
        ax.add_artist(patches.Circle(self._center_xy, self._radius,
                                     fill=False))

    def highlight_side(self, ax, color, label):
        theta_rad = np.linspace(0.75 * np.pi, 1.25 * np.pi, 91)
        xs = self._center_xy[0] + self._radius * np.cos(theta_rad)
        ys = self._center_xy[1] + self._radius * np.sin(theta_rad)
        ax.plot(xs, ys, lw=lw_highlight, color=color, label=label)

    def draw_broken_cutting_plane(self, ax, inner_right_cube, outer_cube):
        x0 = self._center_xy[0]
        x_shift = inner_right_cube._lower_left_xy[0]
        y0 = self._center_xy[1] - self._radius
        y1 = outer_cube._lower_left_xy[1]
        y2 = inner_right_cube._lower_left_xy[1]
        y3 = inner_right_cube._lower_left_xy[1] + inner_right_cube._height
        y4 = outer_cube._lower_left_xy[1] + outer_cube._height
        y5 = self._center_xy[1] + self._radius
        # draw cutting plane segments from bottom up
        ax.plot([x0, x0], [y0, y1])
        ax.plot([x0, x_shift], [y1, y2])
        ax.plot([x_shift, x0], [y3, y4])
        ax.plot([x0, x0], [y4, y5])

    def connect_to_sphere(self, ax, sphere):
        sin45 = cos45 = np.cos(np.pi/4)
        x0 = self._center_xy[0] - cos45 * self._radius
        x1 = self._center_xy[0] + cos45 * self._radius
        y0 = self._center_xy[1] - sin45 * self._radius
        y1 = self._center_xy[1] + sin45 * self._radius
        sx0 = sphere._center_xy[0] - cos45 * sphere._radius
        sx1 = sphere._center_xy[0] + cos45 * sphere._radius
        sy0 = sphere._center_xy[1] - sin45 * sphere._radius
        sy1 = sphere._center_xy[1] + sin45 * sphere._radius
        ax.plot([x0, sx0], [y0, sy0])
        ax.plot([x1, sx1], [y0, sy0])
        ax.plot([x0, sx0], [y1, sy1])
        ax.plot([x1, sx1], [y1, sy1])


def bco_domain_figure():

    # distance between compact objects
    distance_AB = 4.8

    # shift of central blocks relative to origin
    shift = [-1.0, 0.8]

    # blocks around compact-object A
    A_center = [0.5 * distance_AB + shift[0], shift[1]]
    radius_A0 = 0.7
    radius_A1 = 1.4
    half_length_A2 = 0.5 * distance_AB
    A0 = Sphere(A_center, radius_A0)
    A1 = Sphere(A_center, radius_A1)
    A2 = Cube(A_center, half_length_A2)

    # blocks around compact-object B
    B_center = [-0.5 * distance_AB + shift[0], shift[1]]
    half_length_B0 = 0.6
    radius_B1 = 2.0
    half_length_B2 = half_length_A2
    B0 = Cube(B_center, half_length_B0)
    B1 = Sphere(B_center, radius_B1)
    B2 = Cube(B_center, half_length_B2)

    # blocks around both A and B
    C_center = [0.0, 0.0]
    half_length_C1 = 7.0
    radius_C2 = 14.0 - 0.05
    C1 = Cube(C_center, half_length_C1)
    C2 = Sphere(C_center, radius_C2)

    # this song and dance means there will be no axes, no axis padding, and
    # no whitespace beyond that which is explicitly added in the figure content
    fig = plt.figure()
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_aspect('equal')
    ax.set_xlim(-14.0, 14.0)
    ax.set_ylim(-14.0, 14.0)
    fig.add_axes(ax)
    plt.axis('off')

    # draw the blocks
    for block in [A0, A1, A2, B0, B1, B2, C1, C2]:
        block.draw(ax)

    # draw the connections between blocks
    A1.connect_to_sphere(ax, A0)
    A2.connect_to_sphere(ax, A1)
    A2.connect_to_cube(ax, C1, this_side_only='right')
    B2.connect_to_cube(ax, B0)
    B2.connect_to_cube(ax, C1, this_side_only='left')
    C1.connect_to_sphere(ax, C2)
    C2.draw_broken_cutting_plane(ax, inner_right_cube=A2, outer_cube=C1)

    # highlight sides with colors
    C2.highlight_side(ax, 'red', 'L4')
    C1.highlight_side(ax, 'orange', 'L3')
    B2.highlight_side(ax, 'lime', 'L2')
    B1.highlight_side(ax, 'blue', 'L1')
    B0.highlight_side(ax, 'darkorchid', 'L0')

    ax.legend()

    fig.savefig('binary_compact_object_domain.png', bbox_inches=0)


bco_domain_figure()
