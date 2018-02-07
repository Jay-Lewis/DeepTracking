--[[
DeepTracking: Seeing Beyond Seeing Using Recurrent Neural Networks.
Copyright (C) 2016  Peter Ondruska, Mobile Robotics Group, University of Oxford
email:   ondruska@robots.ox.ac.uk.
webpage: http://mrg.robots.ox.ac.uk/

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
--]]

require('torch')
cmd = torch.CmdLine()

require('lfs')
require('nngraph')
require('optim')
require('image')
require('SensorData')
require('WeightedBCECriterion')
require('Recurrent2')

cmd:option('-gpu', 1, 'use GPU')
cmd:option('-iter', 60000, 'the number of training iterations')
cmd:option('-N', 100, 'training sequence length')
cmd:option('-model', 'model2', 'neural network model')
cmd:option('-data', 'data.t7', 'training data')
cmd:option('-learningRate', 0.01, 'learning rate')
cmd:option('-initweights', '', 'initial weights')
cmd:option('-saveiter', 1000, 'training iterations to save video/weights')

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

width = 51
height = 51
train_iter = 0
train_modulus = 20

modulus_sum = 0
for i = 0,math.ceil(train_modulus/2)-1 do
  modulus_sum = modulus_sum + i
end

-- load training data (new images)
data = {}
print('Loading training data from file')
torch.setdefaulttensortype('torch.FloatTensor')
for i=1,100 do

img = image.load("/home/justin94lewis/sketchbook/DeepTracking/linear_filtering/TestData/input"..i..".png",1)

data[i] = torch.CudaTensor(1,height,width)
data[i][1]=img


  if(i%1000 ==0) then
  print(i)
  collectgarbage()
  end
end

print('Occupancy grid has size ' .. width .. 'x' .. height)
M = math.floor(#data / params.N) -- total number of training sequences
print('Number of sequences ' .. M)

torch.setdefaulttensortype(DEFAULT_TENSOR_TYPE)

data_true = data

----load training data (tiny set of images)
--print('Loading training data from file ' .. params.data)
--data = torch.load("tinydata.t7")
--print('Occupancy grid has size ' .. width .. 'x' .. height)
--M = math.floor(#data / params.N) -- total number of training sequences
--print('Number of sequences ' .. M)

----load training data (full set)
--print('Loading training data from file ' .. params.data)

--data = torch.load("/home/justin94lewis/Documents/DeepTracking/Linear_Tracking/trainingdata.t7")
----M = math.floor((#data)[1] / params.N) -- total number of training sequences
--M = math.floor(#data / params.N) -- total number of training sequences
--print('Occupancy grid has size ' .. width .. 'x' .. height)

--data_true = torch.load("/home/justin94lewis/Documents/DeepTracking/Linear_Tracking/trainingdata_true.t7")


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

function getTrueSequence(i)
	local input = {}
	for j = 1,params.N do
		input[j] = data_true[(i-1) * params.N + j]:type(DEFAULT_TENSOR_TYPE)
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
    
		image.save('video_' .. params.model .. '/input' .. i .. '.png',  input[i])
		image.save('video_' .. params.model .. '/output' .. i .. '.png', output[i])
	end
	torch.setdefaulttensortype(DEFAULT_TENSOR_TYPE)
end

-- blanks part of the sequence for predictive training
function dropoutInput(target)
	local input = {}
	for i=1,#target do
		input[i] = target[i]:clone()
		if (i-1) % train_modulus >= math.ceil(train_modulus/2) then
		    input[i]:zero()
		end
	end
	return input
end

function getWeights(iteration)
  local m = (3/4-1/2)/(params.iter/2.0-1.0)
  local b = 1/2-m
  local alpha = m*iteration+b
  if(alpha >3/4) then
    alpha = 3/4
  end
  
  local start = 1/8
  local m2 = (2/train_modulus-start)/(3.0*params.iter/4.0-1.0)
  local b2 =start-m2
  local factor = m2*iteration+b2
  if(factor < 2/train_modulus) then
    alpha = 2/train_modulus
  end


  local beta = alpha*factor
  local mfinal = (alpha-train_modulus/2.0*beta)/modulus_sum

  local weights_a = torch.Tensor(train_modulus/2):fill((1-alpha)*2/train_modulus)
  local weights = torch.cat(weights_a, torch.Tensor(train_modulus/2))
  for i = 0,train_modulus/2-1 do
   weights[i+1+train_modulus/2] = mfinal*i+beta
  end
  return weights
end

-- evaluates model on a random input
function trainModel(weights)
	-- input and target
  local randint = torch.IntTensor().random(M)
	local input = getSequence(randint)
	input  = dropoutInput(input)
  local target = getTrueSequence(randint)
	table.insert(input, h0)
	-- forward pass
	w:copy(weights)
	local output = model:forward(input)
	local cost   = criterion:forward(output, target)
	-- backward pass
	dw:zero()
	model:backward(input, criterion:backward(output, target) )
	-- return cost and weight gradients
	return {cost}, dw
end

-- create directory to save weights and videos
lfs.mkdir('weights_' .. params.model)
lfs.mkdir('video_'   .. params.model)

local total_cost, config, state = 0, { learningRate = params.learningRate }, {}
collectgarbage()

-- primary training loop

for k = 1,params.iter do

	xlua.progress(k, params.iter)
--  costweights = getWeights(k)
  
--  --update training criterion
--  criterion = nn.ParallelCriterion()
--  for i=1,params.N do
--    mod_index = (i-1)%train_modulus
--    criterion:add(WeightedBCECriterion(), costweights[mod_index+1])
--  end
  
  --apply SGD
	local _, cost = optim.adagrad(trainModel, w, config, state)
	total_cost = total_cost + cost[1][1]
	-- save the training progress
	if k % params.saveiter == 0 then
		print('Iteration ' .. k .. ', cost: ' .. total_cost / params.saveiter)
		total_cost = 0
		-- save weights
		torch.save('weights_' .. params.model .. '/' .. k .. '.dat', w:type('torch.FloatTensor'))
		-- visualise performance
		evalModel(w)
	end
	-- not to run out of memory
	collectgarbage()
end
