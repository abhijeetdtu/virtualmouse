import cv2
import random

from virtualmouse.constants import ScreenSize , Constants
import time


def resize(frame):
    h,w = frame.shape[:2]
    wn = ScreenSize.WIDTH
    r = wn/w
    hn = int(r*h)
    return cv2.resize(frame,(wn,hn) , interpolation=cv2.INTER_AREA)

def randomCircle(frame):
    rx = random.random()
    ry = random.random()
    h,w = frame.shape[:2]
    rh = int(h*ry)
    rw = int(w*rx)
    circle = [(rw,rh), 5, (255,123,123), 5]
    # frame = cv2.circle(frame, (rw,rh), 5, (255,123,123), 5)
    # return frame
    return circle



def capture(circleFunc):
    c = None
    INTVAL = 2
    data = []

    keybreak = False
    while not keybreak:

        tstart = time.time()

        ret, frame = vid.read()
        c = circleFunc(frame)
        print("New Loop")
        while time.time() < tstart + INTVAL:
            print(time.time() , tstart + INTVAL)
            ret, frame = vid.read()
            frame = resize(frame)
            frame = cv2.flip(frame,1)
            frame = cv2.circle(frame, *c)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                keybreak= True
                break
        data.append([frame , c])

    return data


if __name__ == "__main__":

    vid = cv2.VideoCapture(0)
    capture(randomCircle)

# After the loop release the cap object
    vid.release()
# Destroy all the windows
    cv2.destroyAllWindows()

    import pickle
    import datetime

    ts = datetime.datetime.today().strftime("%d_%m_%y_%H_%M")

    with open(f"{Constants.IMAGE_DUMP_FILE}_{ts}", "wb") as f:
        pickle.dump(data,f)
