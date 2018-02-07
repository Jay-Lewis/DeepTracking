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
cmd:option('-iter', 10, 'the number of training iterations')
cmd:option('-N', 100, 'training sequence length')
cmd:option('-model', 'model', 'neural network model')
cmd:option('-data', 'data.t7', 'training data')
cmd:option('-learningRate', 0.01, 'learning rate')
cmd:option('-weights',"2000.dat", 'initial weights')

params = cmd:parse(arg)

if params.gpu > 0 then
	print('Using GPU ' .. params.gpu)
	require('cunn')
	require('cutorch')
	cutorch.setDevice(params.gpu)
	DEFAULT_TENSOR_TYPE = 'torch.CudaTensor'
end


-- load training data (new images)
data = {}
width = 51
height = 51
print('Loading training data from file')
torch.setdefaulttensortype('torch.FloatTensor')
for i=1,params.iter*params.N do

img = image.load("/home/justin94lewis/Documents/DeepTracking/New_Tracking/TestData_png/input"..i..".png",1)

data[i] = torch.CudaTensor(2,width,height)
data[i][1] = torch.CudaTensor():resize(img:size()):copy(img)
data[i][2] = torch.CudaTensor():resize(img:size()):fill(1.0)

end
print('Occupancy grid has size ' .. width .. 'x' .. height)
M = math.floor(#data / params.N) -- total number of training sequences
print('Number of sequences ' .. M)

torch.setdefaulttensortype(DEFAULT_TENSOR_TYPE)

----Load training data-----------------------------------

--DEFAULT_TENSOR_TYPE = 'torch.FloatTensor'
--torch.setdefaulttensortype(DEFAULT_TENSOR_TYPE)

--print('Loading training data from file ' .. params.data)

----Load full data set
--if false then
----data = torch.load(params.data) -- load pre-processed 2D grid sensor input
--data = LoadSensorData(params.data, params)
--width  = (#data)[4] -- occupancy 2D grid width
--height = (#data)[3] -- occupancy 2D grid height
--print('Occupancy grid has size ' .. width .. 'x' .. height)
--M = math.floor((#data)[1] / params.N) -- total number of training sequences
--print('Number of sequences ' .. M)

--else

----Load smaller data set
--data = torch.load("tinydata.t7")
--end

--Load Network ------------------------------


-- load neural network model
require(params.model)
-- initial hidden state
h0 = getInitialState(width, height)
-- one step of RNN
step = getStepModule(width, height)

-- network weights + gradients
w, dw = step:getParameters()
print('Model has ' .. w:numel() .. ' parameters')

wdirectory = "/home/justin94lewis/Documents/DeepTracking/New_Tracking/weights_model1/"
wname = wdirectory..params.weights
print('Loading weights' .. wname )
w:copy(torch.load(wname))


-- chain N steps into a recurrent neural network
model = Recurrent(step, params.N)




--Run Network on sample stream------------------------------

-- create directory to save videos
directory = 'samplevideo_'
lfs.mkdir(directory   .. params.model)

-- return i-th training sequence
function getSequence(i)
	local input = {}
	for j = 1,params.N do
		input[j] = data[(i-1) * params.N + j]:type(DEFAULT_TENSOR_TYPE)
	end
	return input
end

-- filter and save model performance on a sample sequence
function evalModel(weights,index)
	input = getSequence(index)
	table.insert(input, h0)
	w:copy(weights)
	local output = model:forward(input)
	-- temporarily switch to FloatTensor as image does not work otherwise.
	torch.setdefaulttensortype('torch.FloatTensor')
	for i = 1,#input-1 do
		image.save(directory .. params.model .. '/input' .. i .. '.png',  input[i][1])
		image.save(directory .. params.model .. '/output' .. i .. '.png', output[i])
	end
	torch.setdefaulttensortype(DEFAULT_TENSOR_TYPE)
end

-- blanks part of the sequence for predictive training
function dropoutInput(target)
	local input = {}
	for i=1,#target do
		input[i] = target[i]:clone()
		if (i-1) % 20 >= 9 then
		    input[i]:zero()
		end
	end
	return input
end

function evalModel_dropout(weights,index)
	input = getSequence(index)
  input = dropoutInput(input)
	table.insert(input, h0)
	w:copy(weights)
	local output = model:forward(input)
	-- temporarily switch to FloatTensor as image does not work otherwise.
	torch.setdefaulttensortype('torch.FloatTensor')
	for i = 1,#input-1 do
		image.save(directory .. params.model .. '/input' .. i .. '.png',  input[i][1])
		image.save(directory .. params.model .. '/output' .. i .. '.png', output[i])
	end
	torch.setdefaulttensortype(DEFAULT_TENSOR_TYPE)
end


evalModel_dropout(w,5)

--Save / Display network activaitons------------------------------
--input = getSequence(1)
--output = model:fullview_forward(input)
