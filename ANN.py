from Nodes import *
import csv
from random import *
from math import *

def main():
    count = 0
    j = 0
    same = 0.0
    alpha = .05
    delta = 0.0
    value = 0.0
    outputList = []
    outputs = []
    RMSE = 0.0
    errors = []
    max = 20.0
    min = -20.0
    # Array for each hidden node with as many values input + bias nodes
    weightsHI = [[1,-.5],[.25,-1]]
    # Array for each output node with as many values as hidden nodes + bias Nodes
    weightsOH = [[1,-1,1]]
    nodesI = [InputNode(),BiasNode()]
    nodesH = [HiddenNode(),HiddenNode(),BiasNode()]
    nodesO = [OutputNode()]

    #data = open("small.csv",'r')
    data = open("f(x)2x.csv",'r')
    #data = open("sin(x).csv",'r')
    with data as d:
        reader = csv.reader(d)
        for row in reader:
            print(row)
            outputList.append(float((float(row[0]) - min)/(max - min)))
    # with data as d:
    #     reader = csv.reader(d)
    #     for row in reader:
    #         outputList.append(float(row[0]))

    counter = len(outputList)

    while count != 10000 and same != 10:
        ###################################
        #FORWARD
        ###################################

        # reset value to 0
        value = 0

        #Set inputs for all non-bias nodes Input Layer
        for i in range(0,len(nodesI)):
            if not type(nodesI[i]) is BiasNode:
                nodesI[i].setInput(outputList[j])

        #Set inputs for all non-bias nodes Hidden Layer
        for k in range(0,len(nodesH)):
            if not type(nodesH[k]) is BiasNode:
                for i in range(0,len(nodesI)):
                    value =  value + nodesI[i].getInput() * weightsHI[k][i]
                nodesH[k].setInput(value)
                value = 0

        #Calculate Output for all non-bias nodes Hidden Layer
        for i in range(0,len(nodesH)):
            if not type(nodesH[i]) is BiasNode:
                nodesH[i].sigmoid()

        #Set inputs for Output Layer
        for k in range(0,len(nodesO)):
            for i in range(0,len(nodesH)):
                value = value + nodesH[i].getOutput() * weightsOH[k][i]
            nodesO[k].setInput(value)
            value = 0

        #Calculate Output for Output Layer
        for i in range(0,len(nodesO)):
            nodesO[i].sigmoid()

        ###################################
        #CHECK
        ###################################
        for i in range(0,len(nodesO)):
            outputs.append(nodesO[i].getOutput())
            errors.append((nodesO[i].getOutput() - outputList[j]*2)**2)
            if nodesO[i].getOutput() == outputList[j]*2 or abs(nodesO[i].getOutput() - outputList[j]*2) < .01:
                 same = same + 1
            else:
                 same = 0
            #print("beginning input: " + str(outputList[j] * (max - min) + min) + " current result: " + str(nodesO[i].getOutput() * (max - min) + min) + " actual: " + str((outputList[j] * (max - min) + min)*2) + " iteration: " + str(count))
            print("beginning input: " + str(outputList[j]) + " current result: " + str(nodesO[i].getOutput()) + " actual: " + str(outputList[j]*2) + " iteration: " + str(count))

        ###################################
        #BACKWARD
        ###################################

        #Calculate Delta for Output Layer
        for i in range(0,len(nodesO)):
            nodesO[i].setDelta(outputList[j]*2)

        #Calculate Delta for all non-bias nodes Hidden Layer
        for k in range(0,len(nodesH)):
            if not type(nodesH[k]) is BiasNode:
                for i in range(0,len(nodesO)):
                    nodesH[k].setDelta(weightsOH[i][k] * nodesO[i].getDelta())

        #Calculate New Weights for Hidden Layer
        for i in range(0,len(nodesO)):
            for k in range(0,len(nodesH)):
                weightsOH[i][k] = weightsOH[i][k] - (alpha) * nodesO[i].getDelta() * nodesH[k].getOutput()

        #Calculate New Weights for Input Layer
        for i in range(0,len(nodesH)):
            if not type(nodesH[i]) is BiasNode:
                for k in range(0,len(nodesI)):
                    weightsHI[i][k] = weightsHI[i][k] - 2*(alpha) * nodesH[i].getDelta() * nodesI[k].getOutput()
        j = j + 1
        if j == counter:
            RMSE = sqrt(np.sum(errors).mean())
            if RMSE < .1:
                myData =[[RMSE]]
                myFile = myFile = open('RMSE.csv','a')
                with myFile:
                    writer = csv.writer(myFile)
                    writer.writerows(myData)
                break
            if count % 100 == 0:
                myData =[[RMSE]]
                myFile = myFile = open('RMSE.csv','a')
                with myFile:
                    writer = csv.writer(myFile)
                    writer.writerows(myData)
            j = 0
            count = count + 1
            if count != 10000:
                errors = []
                outputs = []
            shuffle(outputList)
    myData = [[]]
    myData = [["weightsHO:"]]
    myFile = myFile = open('final.csv','a')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(myData)

    for i in range(0,len(weightsOH)):
        for k in range(0,len(weightsOH[i])):
            myData = [[weightsOH[i][k]]]
            myFile = myFile = open('final.csv','a')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(myData)

    myData = [["weightsIH:"]]
    myFile = myFile = open('final.csv','a')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(myData)

    for i in range(0,len(weightsHI)):
        for k in range(0,len(weightsHI[i])):
            myData = [[weightsHI[i][k]]]
            myFile = myFile = open('final.csv','a')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(myData)

    myData = [["output","target"]]
    myFile = myFile = open('final.csv','a')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(myData)

    for i in range(0,len(outputs)):
        myData = [[outputs[i],(outputList[i])*2]]
        myFile = myFile = open('final.csv','a')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(myData)

main()
