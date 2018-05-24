from Nodes_Normalization import *
import csv
import os
from random import *
from math import *

def main():
    os.remove('RMSE.csv') if os.path.exists('RMSE.csv') else None
    os.remove('test.csv') if os.path.exists('test.csv') else None
    os.remove('final.csv') if os.path.exists('final.csv') else None
    count = 0
    j = 0
    value = 0.0
    population = []
    inputOutput = []
    crimes = []
    RMSE = 0.0
    errors = []
    # Array for each hidden node with as many values input + bias nodes
    weightsHI = [[uniform(-1,1),uniform(-1,1)],[uniform(-1,1),uniform(-1,1)],[uniform(-1,1),uniform(-1,1)]]
    # Array for each output node with as many values as hidden nodes + bias Nodes
    weightsOH = [[uniform(-1,1),uniform(-1,1),uniform(-1,1),uniform(-1,1)]]
    nodesI = [InputNode(),BiasNode()]
    nodesH = [HiddenNode(),HiddenNode(),HiddenNode(),BiasNode()]
    nodesO = [OutputNode()]

    myData =[["Input","OutputActual","OutputObserved"]]
    myFile = myFile = open('test.csv','a')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(myData)

    data = open("population_numbUrban.csv",'r')
    with data as d:
        reader = csv.reader(d)
        for row in reader:
            inputOutput.append([float(row[0]),float(row[1])])

    shuffle(inputOutput)
    inputOutput_l = len(inputOutput)/2
    inputOutput_train = inputOutput[:int(inputOutput_l)]
    inputOutput_test = inputOutput[int(inputOutput_l):]

    counter = len(inputOutput_train)
    Epochs = int(input("Enter the number of Epochs to train: "))
    alpha = float(input("Enter the learning rate: "))
    while count != Epochs:
        ###################################
        #FORWARD
        ###################################
        #Set inputs for all non-bias nodes Input Layer
        nodesI[0].setInput(inputOutput_train[j][0])

        #Set inputs for all non-bias nodes Hidden Layer
        for k in range(0,len(nodesH)):
            value = 0
            if not type(nodesH[k]) is BiasNode:
                for i in range(0,len(nodesI)):
                    value += nodesI[i].getOutput() * weightsHI[k][i]
                nodesH[k].sigmoid(value)

        #Set inputs for Output Layer
        for k in range(0,len(nodesO)):
            value = 0
            for i in range(0,len(nodesH)):
                value += nodesH[i].getOutput() * weightsOH[k][i]
            nodesO[k].sigmoid(value)

        ###################################
        #CHECK
        ###################################
        errors.append(pow(nodesO[0].getOutput() - inputOutput_test[j][1],2))
        print("Input: " + str(inputOutput_train[j][0]) + " OutputActual: " + str(inputOutput_train[j][1])+  " OutputObserved: " +str(nodesO[0].getOutput()) + " epoch: " + str(count))


        ###################################
        #BACKWARD
        ###################################

        #Calculate Delta for Output Layer
        nodesO[0].setDelta(inputOutput_train[j][1])

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
                weightsOH[i][k] = weightsOH[i][k] - (alpha) * nodesO[i].getDelta() * nodesH[k].getOutput()

        #Calculate New Weights for Input Layer
        for i in range(0,len(nodesH)):
            if not type(nodesH[i]) is BiasNode:
                for k in range(0,len(nodesI)):
                    weightsHI[i][k] = weightsHI[i][k] - 2*(alpha) * nodesH[i].getDelta() * nodesI[k].getOutput()
        j+=1
        if j == counter:
            RMSE = sqrt(np.average(errors))
            if RMSE < .1:
                myData =[[RMSE]]
                myFile = myFile = open('RMSE.csv','a')
                with myFile:
                    writer = csv.writer(myFile)
                    writer.writerows(myData)
                break
            myData =[[RMSE]]
            myFile = myFile = open('RMSE.csv','a')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(myData)
            j = 0
            count = count + 1
            if count != Epochs:
                errors = []
            shuffle(inputOutput_train)

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

    ###################################
    #RUN TESTING SET
    ###################################
    counter = len(inputOutput_test)
    j = 0
    count = 0
    while j != counter: ###################################
        #FORWARD TESTING SET
        ###################################
        #Set inputs for all non-bias nodes Input Layer
        nodesI[0].setInput(inputOutput_test[j][0])

        #Set inputs for all non-bias nodes Hidden Layer
        for k in range(0,len(nodesH)):
            value = 0
            if not type(nodesH[k]) is BiasNode:
                for i in range(0,len(nodesI)):
                    value += nodesI[i].getOutput() * weightsHI[k][i]
                nodesH[k].sigmoid(value)

        #Set inputs for Output Layer
        for k in range(0,len(nodesO)):
            value = 0
            for i in range(0,len(nodesH)):
                value += nodesH[i].getOutput() * weightsOH[k][i]
            nodesO[k].sigmoid(value)

        ###################################
        #CHECK
        ###################################
        myData =[[inputOutput_test[j][0],inputOutput_test[j][1],nodesO[0].getOutput()]]
        myFile = myFile = open('test.csv','a')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(myData)
        print("Input: " + str(inputOutput_test[j][0]) + " OutputActual: " + str(inputOutput_test[j][1])+  " OutputObserved: " +str(nodesO[0].getOutput()) + " epoch: " + str(count))
        j += 1
    exit()
main()
