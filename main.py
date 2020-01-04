# -*- coding: utf-8 -*-

from ML import PreProcessing, Processing
import sys, os


P = PreProcessing()
D = Processing()


#file1 = ".\\datasets\\lymphography.csv"
#file2 = ".\\datasets\\adult.csv"
#
#
#dataset = P.getData(file1)
#
##precision LabelEncoder
#lblEncoderDataSet = P.labelEncoder(dataset['descriptive'])
#ds1 = D.splitDataset(lblEncoderDataSet,dataset['target'],0.5,0)
#
##labelEncoder + standardScaler
#lblEncoderStandardScalerDataSet = P.standarScaler(lblEncoderDataSet)
#ds2 = D.splitDataset(lblEncoderStandardScalerDataSet,dataset['target'],0.5,0)
##labelEncder + oneHotEncoder
#lblOneHotDataSet = P.oneHotEncoder(lblEncoderDataSet)
#ds3 = D.splitDataset(lblOneHotDataSet,dataset['target'],0.5,0)
##lableEnconder + OneHot + standar
#allDataset = P.standarScaler(lblOneHotDataSet)
#ds4 =  D.splitDataset(allDataset,dataset['target'],0.5,0)
#
#
#classifierNB = D.naiveBayes()
#
#prediction = D.getPrediction(classifierNB,ds3['descriptiveTraining'],ds3['descriptiveTest'], ds3['targetTraining'])
#
#result = D.getResults(ds1['targetTest'],prediction)



####
menu_actions  = {}

DataSet = {}
Data = []
Calc = {}



def exec_menu(choice):
    os.system('clear')
    ch = choice.lower()
    if ch == '':
        menu_actions['main_menu']()
    else:
        try:
            menu_actions[ch]()
        except KeyError:
            print ("Invalid selection, please try again.\n")
            menu_actions['main_menu']()
    return


def exec_menu_file(choice):
    global DataSet
    global P
    os.system('clear')
    ch = choice.lower()
    if ch == '':
        menu_actions['main_menu']()
    else:
        try:
           if ch == "1":
               File = ".\\datasets\\lymphography.csv"
               DataSet = P.getData(File)
               print("--->Lymphography selecionado")
               menu_actions['main_menu']()
           elif ch == "2":
               File = ".\\datasets\\adult.csv"
               DataSet = P.getData(File)
               print("--->Adult selecionado")
               menu_actions['main_menu']()
           elif ch == "9":
               back()
           elif ch == "0":
               exit()
           else:
               print ("Escolha inválida, tente novamente.\n")
               menu1()
        except KeyError:
            print ("Escolha inválida, tente novamente.\n")
            menu1()
    return


def exec_menu_precision(choice):
    global P
    global Data
    global DataSet
   
    
    os.system('clear')
    ch = choice.lower()
    
    if not DataSet:
        print("Deve selecionar um dataset primeiro!")
        menu1()
    else:
        if ch == '':
            menu_actions['main_menu']()
        else:
            try:
               if ch == "1":
                    Data = P.labelEncoder(DataSet['descriptive'])
                    print("--->labelEncoder selecionado")
                    menu_actions['main_menu']()
               elif ch == "2":
                    Data = P.labelEncoder(DataSet['descriptive'])
                    Data = P.standarScaler(Data)
                    print("--->labelEncoder + standardScaler")
                    menu_actions['main_menu']()
               elif ch == "3":
                    Data = P.labelEncoder(DataSet['descriptive'])
                    Data = P.oneHotEncoder(Data)
                    print("--->labelEncoder + oneHotEncoder")
                    menu_actions['main_menu']()
               elif ch == "4":
                    Data = P.labelEncoder(DataSet['descriptive'])
                    Data = P.oneHotEncoder(Data)
                    Data = P.standarScaler(Data)
                    print("--->labelEncoder + oneHotEncoder + standardScaler")
                    menu_actions['main_menu']()
               elif ch == "9":
                   back()
               elif ch == "0":
                   exit()
               else:
                    print("Escolha inválida, tente novamente\n")
                    menu1()
            
            except KeyError:
                print ("Invalid selection, please try again.\n")
                menu_actions['main_menu']()
    return


def exec_menu_algorithm(choice):
    global D
    global Data
    global DataSet
   
    
    os.system('clear')
    ch = choice.lower()
    
    if not DataSet:
        print("Deve selecionar um dataset primeiro!")
        menu1()
    elif not Data.any():
        print("Deve selecionar a precisão antes")
        menu2()
    else:
        if ch == '':
            menu_actions['main_menu']()
        else:
            testSize = float(input("Insira o tamanho de testes: " ))
            try:
               if ch == "1":
                    classifier = D.naiveBayes()
                    r=CalcResult(classifier, testSize)
                    print("-----resultado------\n")
                    print(r)
                    menu_actions['main_menu']()
               elif ch == "2":
                    classifier = D.decisionTree()
                    r=CalcResult(classifier, testSize)
                    print("-----resultado------\n")
                    print(r)
                    menu_actions['main_menu']()
               elif ch == "3":
                    classifier = D.randomForest()
                    r=CalcResult(classifier, testSize)
                    print("-----resultado------\n")
                    print(r)
                    menu_actions['main_menu']()
               elif ch == "4":
                    neighbors = input("Nr vizinhos: ")
                    classifier = D.kNN(neighbors)
                    r=CalcResult(classifier, testSize)
                    print("-----resultado------\n")
                    print(r)
                    menu_actions['main_menu']()
               elif ch == "9":
                   back()
               elif ch == "0":
                   exit()
               else:
                    print("Escolha inválida, tente novamente\n")
                    menu1()
            except KeyError:
                print ("Invalid selection, please try again.\n")
                menu_actions['main_menu']()
    return


def CalcResult(classifier, testSize):
    global Calc
    global D
    global Data
    global DataSet
    
    Calc =  D.splitDataset(Data,DataSet['target'],testSize,0)
    Prediction= D.getPrediction(classifier,Calc['descriptiveTraining'],Calc['descriptiveTest'], Calc['targetTraining'])
    result = D.getResults(Calc['targetTest'],Prediction)
     
    return result
    
    
def main_menu():
    os.system('clear')
    
    try:
      
        print ("Selcione uma opção:")
        print ("1. Escolher Dataset")
        print ("2. Escolher Precisão")
        print ("3. Escolher Algoritmo")
        print ("\n0. Sair")
        choice = input(" >>  ")
        exec_menu(choice)
    except SystemExit:
        print("Bye")
 
    return


def menu1():
    print ("Selecionar dataset")
    print ("1. Lymphography")
    print ("2. Adult")
    print ("9. Voltar")
    print ("0. Sair")
    choice = input(" >>  ")
    exec_menu_file(choice)
    return
 
 
# Menu 2
def menu2():
    print ("Selecionar precisão")
    print ("1. labelEncoder")
    print ("2. labelEncoder + standardScaler")
    print ("3. labelEncoder + oneHotEncoder")
    print ("4. labelEncoder + oneHotEncoder + standardScaler")
    print ("9. Voltar")
    print ("0. Sair" )
    choice = input(" >>  ")
    exec_menu_precision(choice)
    return


def menu3():
    print ("Selecionar algoritmo")
    print ("1. Naive Bayes")
    print ("2. Decision Tree")
    print ("3. Random Forest")
    print ("4. kNN")
    print ("9. Voltar")
    print ("0. Sair" )
    choice = input(" >>  ")
    exec_menu_algorithm(choice)
    return
 
# Back to main menu
def back():
    menu_actions['main_menu']()
 
# Exit program
def exit():
    sys.exit(1)
 
# =======================
#    MENUS DEFINITIONS
# =======================
 
# Menu definition
menu_actions = {
    'main_menu': main_menu,
    '1': menu1,
    '2': menu2,
    '3': menu3,
    '9': back,
    '0': exit,
}

if __name__ == "__main__":
    # Launch main menu
    main_menu()