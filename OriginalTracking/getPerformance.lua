
require('torch')
cmd = torch.CmdLine()

require('lfs')
require('nngraph')
require('optim')
require('image')
require('SensorData')
require('WeightedBCECriterion')
require('Recurrent')

cmd:option('-gpu', 1, 'use GPU')
cmd:option('-iter', 100, 'the number of training iterations')
cmd:option('-N', 100, 'training sequence length')
cmd:option('-model', 'model', 'neural network model')
cmd:option('-data', 'data.t7', 'training data')
cmd:option('-learningRate', 0.01, 'learning rate')
cmd:option('-initweights', '', 'initial weights')
cmd:option('-savew', 1000, 'how often to save weights')

params = cmd:parse(arg)

cmd:log('log_' .. params.model .. '.txt', params)

-- switch to GPU
if params.gpu > 0 then
	print('Using GPU ' .. params.gpu)
	require('cunn')
	require('cutorch')
	cutorch.setDevice(params.gpu)
	DEFAULT_TENSOR_TYPE = 'torch.CudaTensor'
else
	print('Using CPU')
	DEFAULT_TENSOR_TYPE = 'torch.FloatTensor'
end

torch.setdefaulttensortype(DEFAULT_TENSOR_TYPE)


----Load training data-----------------------------------

DEFAULT_TENSOR_TYPE = 'torch.FloatTensor'
torch.setdefaulttensortype(DEFAULT_TENSOR_TYPE)

print('Loading training data from file ' .. params.data)

--Load full data set
if false then
--data = torch.load(params.data) -- load pre-processed 2D grid sensor input
data = LoadSensorData(params.data, params)
width  = (#data)[4] -- occupancy 2D grid width
height = (#data)[3] -- occupancy 2D grid height
print('Occupancy grid has size ' .. width .. 'x' .. height)
M = math.floor((#data)[1] / params.N) -- total number of training sequences
print('Number of sequences ' .. M)

else

--Load smaller data set
data = torch.load("tinydata.t7")
width = 51
height = 51
M = math.floor(#data / params.N)
end


-- load neural network model
require(params.model)
-- initial hidden state
h0 = getInitialState(width, height)
-- one step of RNN
step = getStepModule(width, height)

-- network weights + gradients
w, dw = step:getParameters()
print('Model has ' .. w:numel() .. ' parameters')

if #params.initweights > 0 then
	print('Loading weights ' .. params.initweights)
	w:copy(torch.load(params.initweights))
end

-- chain N steps into a recurrent neural network
model = Recurrent(step, params.N)

-- cost function
-- {y1, y2, ..., yN},{t1, t2, ..., tN} -> cost
criterion = nn.ParallelCriterion()
for i=1,params.N do
	criterion:add(WeightedBCECriterion(), 1/params.N)
end

-- return i-th training sequence
function getSequence(i)
	local input = {}
	for j = 1,params.N do
		input[j] = data[(i-1) * params.N + j]:type(DEFAULT_TENSOR_TYPE)
	end
	return input
end

-- filter and save model performance on a sample sequence
function evalModel(weights)
	input = getSequence(1)
	table.insert(input, h0)
	w:copy(weights)
	local output = model:forward(input)
	-- temporarily switch to FloatTensor as image does not work otherwise.
	torch.setdefaulttensortype('torch.FloatTensor')
	for i = 1,#input-1 do
		image.save('video_' .. params.model .. '/input' .. i .. '.png',  input[i][2] / 2 + input[i][1])
		image.save('video_' .. params.model .. '/output' .. i .. '.png', input[i][2] / 2 + output[i])
	end
	torch.setdefaulttensortype(DEFAULT_TENSOR_TYPE)
end

-- blanks part of the sequence for predictive training
function dropoutInput(target)
	local input = {}
	for i=1,#target do
		input[i] = target[i]:clone()
		if (i-1) % 20 >= 10 then
		    input[i]:zero()
		end
	end
	return input
end

-- evaluates model on a random input
function trainModel(weights)
	-- input and target
	local target = getSequence(torch.IntTensor().random(M))
	local input  = dropoutInput(target)
	table.insert(input, h0)
	-- forward pass
	w:copy(weights)
	local output = model:forward(input)
	local cost   = criterion:forward(output, target)

	return cost
end

-- create directory to save weights and videos
lfs.mkdir('weights_' .. params.model)
lfs.mkdir('video_'   .. params.model)

local total_cost, config, state = 0, { learningRate = params.learningRate }, {}
collectgarbage()




-- get average performance

start_i = 1
end_i = 50
COSTS = {}
for index = start_i,end_i do
  --print('Loading weights ')
  name = '/home/justin94lewis/Documents/DeepTracking/weights_model/'.. index*1000 ..".dat"
  --print(name)
  WEIGHT=torch.load(name)
  total_cost = 0
  number = 5
for k = 1,number do

    --get costs
--    local _, cost = optim.adagrad(trainModel, w, config, state)
    cost = trainModel(WEIGHT)
    total_cost = total_cost +cost
    
end
print(total_cost/number)
COSTS[index] = total_cost


torch.save('lin_costs.t7',COSTS)
end



