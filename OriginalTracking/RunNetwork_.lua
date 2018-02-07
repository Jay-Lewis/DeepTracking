require('torch')
cmd = torch.CmdLine()

require('lfs')
require('nngraph')
require('optim')
require('image')
require('SensorData')
require('WeightedBCECriterion')
require('Recurrent')

cmd:option('-gpu', 0, 'use GPU')
cmd:option('-iter', 100000, 'the number of training iterations')
cmd:option('-N', 100, 'training sequence length')
cmd:option('-model', 'model', 'neural network model')
cmd:option('-data', 'data.t7', 'training data')
cmd:option('-learningRate', 0.01, 'learning rate')
cmd:option('-initweights', '', 'initial weights')

params = cmd:parse(arg)


--Load training data-----------------------------------

DEFAULT_TENSOR_TYPE = 'torch.FloatTensor'
torch.setdefaulttensortype(DEFAULT_TENSOR_TYPE)

print('Loading training data from file ' .. params.data)
--data = torch.load(params.data) -- load pre-processed 2D grid sensor input
data = LoadSensorData(params.data, params)
width  = (#data)[4] -- occupancy 2D grid width
height = (#data)[3] -- occupancy 2D grid height
print('Occupancy grid has size ' .. width .. 'x' .. height)
M = math.floor((#data)[1] / params.N) -- total number of training sequences
print('Number of sequences ' .. M)


--Load Network ------------------------------

width = 51
height = 51

-- load neural network model
require(params.model)
-- initial hidden state
h0 = getInitialState(width, height)
-- one step of RNN
step = getStepModule(width, height)

-- network weights + gradients
w, dw = step:getParameters()
print('Model has ' .. w:numel() .. ' parameters')

initweights = '50000.dat'
print('Loading weights' .. initweights )
w:copy(torch.load(initweights))


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
		image.save(directory .. params.model .. '/input' .. i .. '.png',  input[i][2] / 2 + input[i][1])
		image.save(directory .. params.model .. '/output' .. i .. '.png', input[i][2] / 2 + output[i])
	end
	torch.setdefaulttensortype(DEFAULT_TENSOR_TYPE)
end


--evalModel(w,1)

--Save / Display network activaitons------------------------------
input = getSequence(1)
--output = model:fullview_forward(input)
