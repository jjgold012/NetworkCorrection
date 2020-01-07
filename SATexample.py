import sys
import warnings
import numpy as np
from maraboupy import MarabouUtils
from maraboupy import Marabou

nnet_file_name = "./Models/ACASXU_2_9.pb"

net1 = Marabou.read_tf(nnet_file_name)
# Bounds for input 0: [ -0.3284228772, 0.6798577687 ]
# Bounds for input 1: [ -0.5000000551, 0.5000000551 ]
# Bounds for input 2: [ -0.5000000551, 0.5000000551 ]
# Bounds for input 3: [ -0.5000000000, 0.5000000000 ]
# Bounds for input 4: [ -0.5000000000, 0.5000000000 ]
inputVars = net1.inputVars[0]
net1.setLowerBound(inputVars[0], -0.3284228772)
net1.setUpperBound(inputVars[0], 0.6798577687)
net1.setLowerBound(inputVars[1], -0.5)
net1.setUpperBound(inputVars[1], 0.5)
net1.setLowerBound(inputVars[2], -0.5)
net1.setUpperBound(inputVars[2], 0.5)
net1.setLowerBound(inputVars[3], -0.5)
net1.setUpperBound(inputVars[3], 0.5)
net1.setLowerBound(inputVars[4], -0.5)
net1.setUpperBound(inputVars[4], 0.5)

# property: output 3 is minimal
outputVars = net1.outputVars[0]
MarabouUtils.addInequality(net1, [outputVars[3], outputVars[0]], [1, -1], 0)
MarabouUtils.addInequality(net1, [outputVars[3], outputVars[1]], [1, -1], 0)
MarabouUtils.addInequality(net1, [outputVars[3], outputVars[2]], [1, -1], 0)
MarabouUtils.addInequality(net1, [outputVars[3], outputVars[4]], [1, -1], 0)

options = Marabou.createOptions(dnc=True, verbosity=0, initialDivides=2)
vals1, stats1 = net1.solve(options=options)


out_file = open('./input1.csv', 'w')
out_file.write('{},{},{},{},{}\n'.format(vals1[inputVars[0]],
                                         vals1[inputVars[1]],
                                         vals1[inputVars[2]],
                                         vals1[inputVars[3]],
                                         vals1[inputVars[4]]))
out_file.close()

out_file = open('./output1.csv', 'w')
out_file.write('{},{},{},{},{}\n'.format(vals1[outputVars[0]],
                                         vals1[outputVars[1]],
                                         vals1[outputVars[2]],
                                         vals1[outputVars[3]],
                                         vals1[outputVars[4]]))
out_file.close()

np.save('./vals1', np.array([vals1 for i in range(len(vals1))]))

