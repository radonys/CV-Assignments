import cv2
import math

boxes = []
iter = 0

def on_mouse(event, x, y, flags, params):
    
    global iter

    if event == cv2.EVENT_LBUTTONDOWN:

        print ('Start Mouse Position: ' + str(x) + ', ' + str(y))

        start_point = (x, y)
        boxes.append(start_point)

    elif event == cv2.EVENT_LBUTTONUP:

        print ('End Mouse Position: '+str(x)+', '+str(y))

        end_point = (x, y)
        boxes.append(end_point)

        cv2.line(img, boxes[-1], boxes[-2],(0,255,0),10)

        iter += 1


def line_intersection(line1, line2):

    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) 

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)

    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    return x, y

def norm(point1, point2):

    xdiff = point1[0] - point2[0]
    ydiff = point1[1] - point2[1]

    norm = math.sqrt(xdiff*xdiff + ydiff*ydiff)

    return norm

print ("-------------------------INSTRUCTIONS----------------------------")
print ("Draw 8 line segments, holding mouse while drawing")
print ("First two lines for a pair of parallel lines")
print ("Next two lines for another pair of parallel lines")
print ("Finally two lines for a third pair of parallel lines to find Vz")
print ("Now, two lines for objects whose lengths are to be compared")
print ("First draw line for shorter object in image plane starting from bottom")
print ("Then for other object again starting from bottom")
print ("-----------------------------END---------------------------------")
font = cv2.FONT_HERSHEY_SIMPLEX

count = 0
img = cv2.imread('img2.jpg')
number_of_objects = int(raw_input("Number of Objects to be measured:"))

while(1):

    #if iter == 7 + number_of_objects:
    #    break

    count += 1

    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', on_mouse, 0)
    cv2.imshow('image', img)

    if count < 50:

        if cv2.waitKey(33) == 27:

            cv2.destroyAllWindows()
            break

    elif count >= 50:
        count = 0

    if iter==2:

        parallel_1 = line_intersection( [boxes[0],boxes[1]], [boxes[2],boxes[3]] )
        cv2.circle(img,parallel_1,10,(0,0,255),20)

    if iter==4:

        parallel_2 = line_intersection( [boxes[4],boxes[5]], [boxes[6],boxes[7]] )
        cv2.circle(img,parallel_2,10,(0,0,255),10)

        x1 = img.shape[0]
        y1 = ((parallel_1[1]-parallel_2[1])*(x1-parallel_2[0])/(parallel_1[0]-parallel_2[0])) + parallel_2[1]

        x2 = 0
        y2 = ((parallel_1[1]-parallel_2[1])*(x2-parallel_2[0])/(parallel_1[0]-parallel_2[0])) + parallel_2[1]

        cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 20)

parallel_1 = line_intersection( [boxes[0],boxes[1]], [boxes[2],boxes[3]] )
print (parallel_1)

parallel_2 = line_intersection( [boxes[4],boxes[5]], [boxes[6],boxes[7]] )
print (parallel_2)

Vz_Parallel = line_intersection( [boxes[8],boxes[9]], [boxes[10],boxes[11]] )
print (Vz_Parallel)

for i in range(0,number_of_objects):

    print ("Assuming bottom is given as first input for each object")
    vertex = line_intersection( [parallel_1,parallel_2], [boxes[12],boxes[13+(2*i)+1]] )

    bot = boxes[13+(2*i)+1]
    ref = line_intersection( [vertex,boxes[13]], [boxes[13+(2*i)+1],boxes[13+(2*i)+2]] )
    top = boxes[13+(2*i)+2]

    response1 = float(raw_input("Please enter height of shorter object, enter 0 if unknown: "))
    response2 = float(raw_input("Please enter height of other object, enter 0 if unknown: "))

    response = response1 + response2

    print ("Length of unknown object is")
    print (( (norm(top,bot)/norm(ref,bot))*(norm(Vz_Parallel,ref)/norm(Vz_Parallel,top))*response ) )
    value = str(( (norm(top,bot)/norm(ref,bot))*(norm(Vz_Parallel,ref)/norm(Vz_Parallel,top))*response ) )
    cv2.putText(img,value,top, font, 4,(0,0,0),10,cv2.LINE_AA)

cv2.namedWindow('image1',cv2.WINDOW_NORMAL)
cv2.imshow('image1', img)
k=0
while k!=27:
    k = cv2.waitKey(33)   # Esc key to stop
    