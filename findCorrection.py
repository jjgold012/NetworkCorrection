import sys
sys.path.append('../')
import numpy as np
import argparse
from maraboupy import MarabouUtils
from maraboupy import Marabou
from WatermarkVerification import MarabouNetworkTFWeightsAsVar
from functools import reduce
# from gurobipy import *
from copy import deepcopy
from pprint import pprint

sat = 'SAT'
unsat = 'UNSAT'
class findCorrection:

    def __init__(self, epsilon_max, epsilon_interval):
        self.epsilon_max = epsilon_max
        self.epsilon_interval = epsilon_interval

    def epsilonABS(self, network, epsilon_var):
        epsilon2 = network.getNewVariable()
        MarabouUtils.addEquality(network, [epsilon2, epsilon_var], [1, -2], 0)
        
        relu_epsilon2 = network.getNewVariable()
        network.addRelu(epsilon2, relu_epsilon2)
        
        abs_epsilon = network.getNewVariable()
        MarabouUtils.addEquality(network, [abs_epsilon, relu_epsilon2, epsilon_var], [1, -1, 1], 0)
        return abs_epsilon

    def evaluateEpsilon(self, epsilon, network, prediction):
        outputVars = network.outputVars[0]
        abs_epsilons = list()
        for k in network.matMulLayers.keys():
            n, m = network.matMulLayers[k]['vals'].shape
            print(n,m)
            for i in range(n):
                for j in range(m):
                    # if j in [prediction, output]:
                    epsilon_var = network.epsilons[i][j]
                    network.setUpperBound(epsilon_var, epsilon)
                    network.setLowerBound(epsilon_var, -epsilon)
                    abs_epsilon_var = self.epsilonABS(network, epsilon_var)
                    abs_epsilons.append(abs_epsilon_var)
                    # else:
                    #     epsilon_var = network.epsilons[i][j]
                    #     network.setUpperBound(epsilon_var, 0)
                    #     network.setLowerBound(epsilon_var, 0)

        e = MarabouUtils.Equation(EquationType=MarabouUtils.MarabouCore.Equation.LE)
        for i in range(len(abs_epsilons)):
            e.addAddend(1, abs_epsilons[i])
        e.setScalar(epsilon)
        network.addEquation(e)

        MarabouUtils.addInequality(network, [outputVars[2], outputVars[0]], [1, -1], 0)
        MarabouUtils.addInequality(network, [outputVars[3], outputVars[0]], [1, -1], 0)
        MarabouUtils.addInequality(network, [outputVars[4], outputVars[0]], [1, -1], 0)
        MarabouUtils.addInequality(network, [outputVars[2], outputVars[1]], [1, -1], 0)
        MarabouUtils.addInequality(network, [outputVars[3], outputVars[1]], [1, -1], 0)
        MarabouUtils.addInequality(network, [outputVars[4], outputVars[1]], [1, -1], 0)
        vals = network.solve(verbose=True)
        if vals[0]:

            return sat, vals
        else:
            return unsat, vals
    
    # def getNetworkSolution(self, network):
    #     equations = network.equList
    #     numOfVar = network.numVars
    #     networkEpsilons = network.epsilons
    #     epsilonsShape = networkEpsilons.shape 
    #     model = Model("my model")
    #     modelVars = model.addVars(numOfVar, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    #     epsilon = model.addVar(name="epsilon")
    #     model.setObjective(epsilon, GRB.MINIMIZE)
    #     for i in range(epsilonsShape[0]):
    #         for j in range(epsilonsShape[1]):
    #             model.addConstr(modelVars[networkEpsilons[i][j]], GRB.LESS_EQUAL, epsilon)
    #             model.addConstr(modelVars[networkEpsilons[i][j]], GRB.GREATER_EQUAL, -1*epsilon)

    #     for eq in equations:
    #         addends = map(lambda addend: modelVars[addend[1]] * addend[0], eq.addendList)
    #         eq_left = reduce(lambda x,y: x+y, addends)
    #         if eq.EquationType == MarabouCore.Equation.EQ:
    #             model.addConstr(eq_left, GRB.EQUAL, eq.scalar)
    #         if eq.EquationType == MarabouCore.Equation.LE:
    #             model.addConstr(eq_left, GRB.LESS_EQUAL, eq.scalar)
    #         if eq.EquationType == MarabouCore.Equation.GE:
    #             model.addConstr(eq_left, GRB.GREATER_EQUAL, eq.scalar)
                
    #     model.optimize()
    #     epsilons_vals = np.array([[modelVars[networkEpsilons[i][j]].x for j in range(epsilonsShape[1])] for i in range(epsilonsShape[0])])
    #     all_vals = np.array([modelVars[i].x for i in range(numOfVar)])
    #     return epsilon.x, epsilons_vals, all_vals 

    # def findEpsilon(self, network, prediction):
    #     outputVars = network.outputVars
        
    #     predIndices = np.flip(np.argsort(prediction, axis=1), axis=1)        
    #     for i in range(outputVars.shape[0]):
    #         maxPred = predIndices[i][0]
    #         secondMaxPred = predIndices[i][1]
    #         MarabouUtils.addInequality(network, [outputVars[i][maxPred], outputVars[i][secondMaxPred]], [1, -1], 0)
    #     results = self.getNetworkSolution(network)
    #     newOutput = np.array([[results[2][outputVars[i][j]] for j in range(outputVars.shape[1])] for i in range(outputVars.shape[0])])
    #     return results, predIndices[:,0], predIndices[:,1], newOutput
    
    def findEpsilonInterval(self, network, prediction):
        sat_epsilon = self.epsilon_max
        unsat_epsilon = 0.0
        sat_vals = None
        epsilon = sat_epsilon
        while abs(sat_epsilon - unsat_epsilon) > self.epsilon_interval:
            status, vals = self.evaluateEpsilon(epsilon, deepcopy(network), prediction)
            if status == sat:
                sat_epsilon = epsilon
                sat_vals = (status, vals)
            else:
                unsat_epsilon = epsilon
            epsilon = (sat_epsilon + unsat_epsilon)/2
        return unsat_epsilon, sat_epsilon , sat_vals


    def run(self, model_name, model_dir):
        filename = './ProtobufNetworks/last.layer.{}.pb'.format(model_name)
        
        lastlayer_inputs = np.load('./{}.lastlayer.input.npy'.format(model_name))
        predictions = np.load('./{}.prediction.npy'.format(model_name))
        prediction = np.argmax(predictions)
        print('Prediction:')
        print(prediction)
        # inputVals = np.reshape(lastlayer_inputs, (1, lastlayer_inputs.shape[0]))
        network = MarabouNetworkTFWeightsAsVar.read_tf_weights_as_var(filename=filename, inputVals=lastlayer_inputs)
        
        unsat_epsilon, sat_epsilon, sat_vals = self.findEpsilonInterval(network, prediction)
        print('{},{}\n'.format(unsat_epsilon, sat_epsilon))
    
        all_vals = sat_vals[1][0]

        output_vars = network.outputVars[0]
        output_vals = np.array([all_vals[output_vars[i]] for i in range(len(output_vars))])    
        print('Output vector:')
        print(output_vals)
        print('Output vector min:')
        print(np.argmin(output_vals))

        epsilons_vars = network.matMulLayers[0]['epsilons']
        epsilons_vals = np.array([[all_vals[epsilons_vars[j][i]] for i in range(epsilons_vars.shape[1])] for j in range(epsilons_vars.shape[0])])    
        np.save('./{}.vals'.format(model_name), epsilons_vals)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='the name of the model')
    parser.add_argument('--epsilon_max', default=5, help='max epsilon value')
    parser.add_argument('--epsilon_interval', default=0.001, help='epsilon smallest change')
    
    args = parser.parse_args()
    epsilon_max = float(args.epsilon_max)
    epsilon_interval = float(args.epsilon_interval)  
    
    model_name = args.model
    MODELS_PATH = './Models'
    problem = findCorrection(epsilon_max, epsilon_interval)
    problem.run(model_name)