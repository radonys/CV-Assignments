import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image

def get_input_lines(im, min_lines=3):

    n = 0
    lines = np.zeros((3, 0))
    centers = np.zeros((3, 0))

    plt.figure()
    plt.imshow(im)
    print('Set at least %d lines to compute vanishing point' % min_lines)
    while True:
        print('Click the two endpoints, use the right key to undo, and use the middle key to stop input')
        clicked = plt.ginput(2, timeout=0, show_clicks=True)
        if not clicked or len(clicked) < 2:
            if n < min_lines:
                print('Need at least %d lines, you have %d now' % (min_lines, n))
                continue
            else:
                # Stop getting lines if number of lines is enough
                break

        # Unpack user inputs and save as homogeneous coordinates
        pt1 = np.array([clicked[0][0], clicked[0][1], 1])
        pt2 = np.array([clicked[1][0], clicked[1][1], 1])
        # Get line equation using cross product
        # Line equation: line[0] * x + line[1] * y + line[2] = 0
        line = np.cross(pt1, pt2)
        lines = np.append(lines, line.reshape((3, 1)), axis=1)
        # Get center coordinate of the line segment
        center = (pt1 + pt2) / 2
        centers = np.append(centers, center.reshape((3, 1)), axis=1)

        # Plot line segment
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='b')

        n += 1

    return n, lines, centers

def plot_lines_and_vp(im, lines, vp):

    bx1 = min(1, vp[0] / vp[2]) - 10
    bx2 = max(im.shape[1], vp[0] / vp[2]) + 10
    by1 = min(1, vp[1] / vp[2]) - 10
    by2 = max(im.shape[0], vp[1] / vp[2]) + 10

    plt.figure()
    plt.imshow(im)
    for i in range(lines.shape[1]):
        if lines[0, i] < lines[1, i]:
            pt1 = np.cross(np.array([1, 0, -bx1]), lines[:, i])
            pt2 = np.cross(np.array([1, 0, -bx2]), lines[:, i])
        else:
            pt1 = np.cross(np.array([0, 1, -by1]), lines[:, i])
            pt2 = np.cross(np.array([0, 1, -by2]), lines[:, i])
        pt1 = pt1 / pt1[2]
        pt2 = pt2 / pt2[2]
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'g')

    plt.plot(vp[0] / vp[2], vp[1] / vp[2], 'ro')
    plt.show()

def get_top_and_bottom_coordinates(im, obj):

    plt.figure()
    plt.imshow(im)

    print('Click on the top coordinate of %s' % obj)
    clicked = plt.ginput(1, timeout=0, show_clicks=True)
    x1, y1 = clicked[0]
    # Uncomment this line to enable a vertical line to help align the two coordinates
    # plt.plot([x1, x1], [0, im.shape[0]], 'b')
    print('Click on the bottom coordinate of %s' % obj)
    clicked = plt.ginput(1, timeout=0, show_clicks=True)
    x2, y2 = clicked[0]

    plt.plot([x1, x2], [y1, y2], 'b')

    return np.array([[x1, x2], [y1, y2], [1, 1]])

def get_vanishing_point(lines):

    vp = np.zeros((3, 1))
    #vp[0] = (lines[0])/(-1*lines[2])
    #vp[1] = (lines[1])/(-1*lines[3])
    #vp[0:1] = (lines[0:1,:])/(-1*lines[2,:])
    b = []
    print(lines)
    for i in range (0,2):
        b.append(lines[i])
    c = lines[2]
    #print(lines.shape())

    b = np.array(b)
    c = np.array(c)
    #b = b.reshape(1,-1)
    c = c.reshape(1,-1)
    print ("c",c)
    print ("b",b)
    
    b = b.T
    c = c.T
    print ("c_new",c)
    print ("b_new",b)
    #print("b and c",b.T," ", c)
    d = np.matmul(b.T,c)
    print ("d",d)
    
    a = np.matmul(np.linalg.inv(np.matmul(np.transpose(b),b)), np.matmul(b.T,c))
 #   print (c)
  #  print (b)
    print ("a",a)
    vp[2] = 1
    pass

def get_horizon_line(vpts):

    print("Hello!")
    print("vpts",vpts)
    horizon = real(np.matmul(np.transpose(vpts[2]), np.transpose(vpts[0])))
    length = sqrt(horizon[0]^2 + horizon[1]^2)
    horizon = horizon/length
    print(horizon)
    pass

def plot_horizon_line(im,vpts):

	plt.figure()
    plt.imshow(im)

    plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='b')
    plt.show()
    
    pass

im = np.asarray(Image.open('CSL.jpg'))

# Part 1
# Get vanishing points for each of the directions
num_vpts = 3
vpts = np.zeros((3, num_vpts))

for i in range(num_vpts):

    print('Getting vanishing point %d' % i)
    # Get at least three lines from user input
    n, lines, centers = get_input_lines(im)
    # <YOUR IMPLEMENTATION> Solve for vanishing point
    vpts[:, i] = get_vanishing_point(lines)
    # Plot the lines and the vanishing point
    plot_lines_and_vp(im, lines, vpts[:, i])

# <YOUR IMPLEMENTATION> Get the ground horizon line
horizon_line = get_horizon_line(vpts)
# <YOUR IMPLEMENTATION> Plot the ground horizon line
plot_horizon_line(im,vpts)

# Part 2
# <YOUR IMPLEMENTATION> Solve for the camera parameters (f, u, v)
f, u, v = get_camera_parameters()