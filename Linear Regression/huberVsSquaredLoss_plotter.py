
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def plotHuberLoss():
    xaxis = []
    yaxis = []
    delta = 3
    for x in range(-9, 10):
        xaxis.append(x)

    for x in xaxis:
        if np.abs(x) <= delta:
            z = 0.5*np.square(x)
            yaxis.append(z)
        else:
            z = delta*(np.abs(x) - 0.5*delta)
            yaxis.append(z)
    print(xaxis)
    print(yaxis)
    plt.plot(xaxis,yaxis)
    plt.savefig("huberplot")

def plotSquaredLoss():
    xaxis = []
    yaxis = []
    delta = 3
    for x in range(-9, 10):
        xaxis.append(x)

    for x in xaxis:
        z = 0.5*np.square(x)
        yaxis.append(z)

    print(xaxis)
    print(yaxis)

    plt.xlabel("y")
    plt.ylabel("Loss")
    plt.text(0.15, 15, "Blue: Huber Loss")
    plt.text(0.1, 20,"Orange: Squared Loss")
    plt.plot(xaxis,yaxis)
    plt.savefig("sqlossplot")

def loadpic():
    pic = Image.open("output.pgm", "r")
    pic.load()
    data = np.asarray(pic, dtype="int32")
    pic.close()
    print(data)

def savepic() :
    data1 = [[255 , 0, 255],
            [0, 255, 0],
            [100, 20, 100]]

    data2 = [[0, 123, 255, 123, 0],
             [123, 0, 123, 0, 123],
             [255, 123, 0, 123, 255]]
    img = Image.fromarray(np.asarray(data2, dtype="uint8"),"L")
    img.save("input.pgm")

if __name__ == "__main__":
    #plotHuberLoss()
    #plotSquaredLoss()
    savepic()
    #loadpic()