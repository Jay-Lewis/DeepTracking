require('torch')
cmd = torch.CmdLine()

require('lfs')
require('nngraph')
require('optim')
require('image')
require('SensorData')
require('WeightedBCECriterion')
require('Recurrent')

name = "/home/justin94lewis/Documents/DeepTracking/data.t7"
name = "/home/justin94lewis/Documents/DeepTracking/New_Tracking/tinydata.t7"
cmd:option('-gpu', 0, 'use GPU')
cmd:option('-iter', 100000, 'the number of training iterations')
cmd:option('-N', 100, 'training sequence length')
cmd:option('-model', 'model', 'neural network model')
cmd:option('-data',name, 'training data')
cmd:option('-learningRate', 0.01, 'learning rate')
cmd:option('-initweights', '', 'initial weights')

params = cmd:parse(arg)

DEFAULT_TENSOR_TYPE = 'torch.FloatTensor'
torch.setdefaulttensortype(DEFAULT_TENSOR_TYPE)


-- Load Data and then save smaller portion for testing

data = torch.load(name)
print((data[1]))

--print('Loading training data from file ' .. params.data)
----data = torch.load(params.data) -- load pre-processed 2D grid sensor input
--data = LoadSensorData(params.data, params)
--width  = (#data)[4] -- occupancy 2D grid width
--height = (#data)[3] -- occupancy 2D grid height
--print('Occupancy grid has size ' .. width .. 'x' .. height)
--M = math.floor((#data)[1] / params.N) -- total number of training sequences
--print('Number of sequences ' .. M)


--torch.save('process_data.t7',data)
--newdata = torch.load('process_data.t7')

print(#data)

--N = 100
--tinydata = {}

--for i=1,2*N do
--  tinydata[i] = data[i]
--end

--torch.save('tinydata.t7', tinydata)

