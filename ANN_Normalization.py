from Nodes import *
import csv
import os
from random import *
from math import *

def main():
    os.remove('outputs.csv') if os.path.exists('outputs.csv') else None
    os.remove('RMSE.csv') if os.path.exists('RMSE.csv') else None
    os.remove('test.csv') if os.path.exists('test.csv') else None
    os.remove('final.csv') if os.path.exists('final.csv') else None
    count = 0
    j = 0
    alpha = .09
    value = 0.0
    outputList = []
    outputList1 = []
    outputs = []
    RMSE = 0.0
    oldRMSE = 0.0
    errors = []
    max = 0.0
    min = 0.0
    # Array for each hidden node with as many values input + bias nodes
    weightsHI = [[uniform(-1,1),uniform(-1,1)],[uniform(-1,1),uniform(-1,1)],[uniform(-1,1),uniform(-1,1)],[uniform(-1,1),uniform(-1,1)]]
    # Array for each output node with as many values as hidden nodes + bias Nodes
    weightsOH = [[uniform(-1,1),uniform(-1,1),uniform(-1,1),uniform(-1,1),uniform(-1,1)]]
    nodesI = [InputNode(),BiasNode()]
    nodesH = [HiddenNode(),HiddenNode(),HiddenNode(),HiddenNode(),BiasNode()]
    nodesO = [OutputNode()]

    myData =[["input","output","actual"]]
    myFile = myFile = open('outputs.csv','a')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(myData)

    myData =[["input","output","actual"]]
    myFile = myFile = open('test.csv','a')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(myData)

    #data = open("small.csv",'r')
    data = open("f(x)2x.csv",'r')
    #data = open("sin(x).csv",'r')
    with data as d:
        reader = csv.reader(d)
        for row in reader:
            outputList1.append(float(row[0]))
    max = np.amax(outputList1)*2
    min = np.amin(outputList1)*2

    #data = open("small.csv",'r')
    data = open("f(x)2x.csv",'r')
    #data = open("sin(x).csv",'r')
    with data as d:
        reader = csv.reader(d)
        for row in reader:
            outputList.append(float((float(row[0]) - min)/(max - min)))

    shuffle(outputList)
    l = len(outputList)/2
    train = outputList[:int(l)]
    test = outputList[int(l):]

    counter = len(train)

    while count != 10000:
        ###################################
        #FORWARD
        ###################################
        #Set inputs for all non-bias nodes Input Layer
        for i in range(0,len(nodesI)):
            if type(nodesI[i]) is not  BiasNode:
                nodesI[i].setInput(train[j])

        #Set inputs for all non-bias nodes Hidden Layer
        for k in range(0,len(nodesH)):
            value = 0
            if not type(nodesH[k]) is BiasNode:
                for i in range(0,len(nodesI)):
                    if type(nodesI[i]) is BiasNode:
                        value += nodesI[i].getOutput(max,min) * weightsHI[k][i]
                    else:
                        value += nodesI[i].getOutput() * weightsHI[k][i]
                nodesH[k].sigmoid(value)

        #Set inputs for Output Layer
        for k in range(0,len(nodesO)):
            value = 0
            for i in range(0,len(nodesH)):
                if type(nodesH[i]) is BiasNode:
                    value += nodesH[i].getOutput(max,min) * weightsOH[k][i]
                else:
                    value += nodesH[i].getOutput() * weightsOH[k][i]
            nodesO[k].sigmoid(value)

        ###################################
        #CHECK
        ###################################
        for i in range(0,len(nodesO)):
            outputs.append(2*(nodesO[i].getOutput() * (max - min) + min))
            errors.append(pow(train[j]-nodesO[i].getOutput(),2))
            if RMSE <= oldRMSE:
                myData =[[train[j] * (max - min) + min,2*(nodesO[i].getOutput() * (max - min) + min),2*(train[j] * (max - min) + min)]]
                myFile = myFile = open('outputs.csv','a')
                with myFile:
                    writer = csv.writer(myFile)
                    writer.writerows(myData)
            print("beginning input: " + str(train[j] * (max - min) + min) + " current result: " + str(2*(nodesO[i].getOutput() * (max - min) + min)) + " actual: " + str(2*(train[j] * (max - min) + min)) + " epoch: " + str(count))
            #print("beginning input: " + str(train[j]) + " current result: " + str(nodesO[i].getOutput()) + " actual: " + str(train[j]*2) + " iteration: " + str(count))


        ###################################
        #BACKWARD
        ###################################

        #Calculate Delta for Output Layer
        for i in range(0,len(nodesO)):
            nodesO[i].setDelta(train[j],max,min)

        #Calculate Delta for all non-bias nodes Hidden Layer
        for k in range(0,len(nodesH)):
            value = 0;
            if not type(nodesH[k]) is BiasNode:
                for i in range(0,len(nodesO)):
                    value += nodesO[i].getDelta() * weightsOH[i][k]
                nodesH[k].setDelta(value)

        #Calculate New Weights for Hidden Layer
        for i in range(0,len(nodesO)):
            for k in range(0,len(nodesH)):
                if type(nodesH[k]) is BiasNode:
                    weightsOH[i][k] = weightsOH[i][k] - (alpha) * nodesO[i].getDelta() * nodesH[k].getOutput(max,min)
                else:
                    weightsOH[i][k] = weightsOH[i][k] - (alpha) * nodesO[i].getDelta() * nodesH[k].getOutput()

        #Calculate New Weights for Input Layer
        for i in range(0,len(nodesH)):
            if not type(nodesH[i]) is BiasNode:
                for k in range(0,len(nodesI)):
                    if type(nodesI[k]) is BiasNode:
                        weightsHI[i][k] = weightsHI[i][k] - 2*(alpha) * nodesH[i].getDelta() * nodesI[k].getOutput(max,min)
                    else:
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
            oldRMSE = RMSE
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
            #shuffle(train)

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
        myData = [[outputs[i],2*(train[i] * (max - min) + min)]]
        myFile = myFile = open('final.csv','a')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(myData)

    ###################################
    #RUN TESTING SET
    ###################################
    counter = len(test)
    j = 0
    while j != counter:
        ###################################
        #FORWARD TESTING SET
        ###################################
        #Set inputs for all non-bias nodes Input Layer
        for i in range(0,len(nodesI)):
            if type(nodesI[i]) is not  BiasNode:
                nodesI[i].setInput(test[j])

        #Set inputs for all non-bias nodes Hidden Layer
        for k in range(0,len(nodesH)):
            value = 0
            if not type(nodesH[k]) is BiasNode:
                for i in range(0,len(nodesI)):
                    if type(nodesI[i]) is BiasNode:
                        value += nodesI[i].getOutput(max,min) * weightsHI[k][i]
                    else:
                        value += nodesI[i].getOutput() * weightsHI[k][i]
                nodesH[k].sigmoid(value)

        #Set inputs for Output Layer
        for k in range(0,len(nodesO)):
            value = 0
            for i in range(0,len(nodesH)):
                if type(nodesH[i]) is BiasNode:
                    value += nodesH[i].getOutput(max,min) * weightsOH[k][i]
                else:
                    value += nodesH[i].getOutput() * weightsOH[k][i]
            nodesO[k].sigmoid(value)

        ###################################
        #CHECK
        ###################################
        for i in range(0,len(nodesO)):
            myData =[[test[j] * (max - min) + min,2*(nodesO[i].getOutput() * (max - min) + min),2*(test[j] * (max - min) + min)]]
            myFile = myFile = open('test.csv','a')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(myData)
            print("beginning input: " + str(test[j] * (max - min) + min) + " current result: " + str(2*(nodesO[i].getOutput() * (max - min) + min)) + " actual: " + str(2*(test[j] * (max - min) + min)))
            #print("beginning input: " + str(test[j]) + " current result: " + str(nodesO[i].getOutput()) + " actual: " + str(test[j]*2) + " iteration: " + str(count))
        j = j + 1
    exit()

main()
