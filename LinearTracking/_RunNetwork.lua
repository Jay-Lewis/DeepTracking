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
cmd:option('-weights','43000.dat', 'initial weights')

params = cmd:parse(arg)

if params.gpu > 0 then
	print('Using GPU ' .. params.gpu)
	require('cunn')
	require('cutorch')
	cutorch.setDevice(params.gpu)
	DEFAULT_TENSOR_TYPE = 'torch.CudaTensor'
else
  DEFAULT_TENSOR_TYPE = 'torch.FloatTensor'
end


-- load training data (new images)
data = {}
data_true = {}
width = 51
height = 51
print('Loading training data from file')
torch.setdefaulttensortype('torch.FloatTensor')
for i=1,params.iter*params.N do

name = "/home/justin94lewis/sketchbook/DeepTracking/linear_filtering/TestData/input"..i..".png"
img = image.load(name,1)

data[i] = torch.Tensor(1,height,width)
data[i][1]=img

end
print('Occupancy grid has size ' .. width .. 'x' .. height)
M = math.floor(#data / params.N) -- total number of training sequences
print('Number of sequences ' .. M)


print('Loading training data from file')
torch.setdefaulttensortype('torch.FloatTensor')
for i=1,params.iter*params.N do

name = "/home/justin94lewis/sketchbook/DeepTracking/gen_ground_truth/TestDataTrue/input"..i..".png"
img = image.load(name,1)

data_true[i] = torch.Tensor(1,height,width)
data_true[i][1]=img

end

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

wdirectory = "/home/justin94lewis/Documents/DeepTracking/Linear_Tracking/weights_model/"
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

function getTrueSequence(i)
	local input = {}
	for j = 1,params.N do
		input[j] = data_true[(i-1) * params.N + j]:type(DEFAULT_TENSOR_TYPE)
	end
	return input
end


function colorDisplay(ground,output,color)
  local imgsize = ground:size()
  local display = torch.Tensor(3,imgsize[1],imgsize[2])
  print(torch.max(output))
  
  for i = 1,imgsize[1] do
    for j = 1,imgsize[2] do
      for k = 1,3 do
        if( ground[i][j] > 0.1) then
        display[k][i][j] = color[k]
        else
        display[k][i][j] = math.pow(output[i][j],3)*255
        end
      end
    end
  end

  return display
end

-- filter and save model performance on a sample sequence
function evalModel(weights,index)
	input = getSequence(index)
	table.insert(input, h0)
	w:copy(weights)
	local output = model:forward(input)
  local display = torch.FloatTensor(3,height,width)
  local ground_truth = getTrueSequence(index)
  local color = {255,0,0}
	-- temporarily switch to FloatTensor as image does not work otherwise.
	torch.setdefaulttensortype('torch.FloatTensor')
	for i = 1,#input-1 do
    display = colorDisplay(ground_truth[i],output[i],color)
		image.save(directory .. params.model .. '/display' .. i .. '.png', display)
	end
	torch.setdefaulttensortype(DEFAULT_TENSOR_TYPE)
end

-- blanks part of the sequence for predictive training
function dropoutInput(target)
	local input = {}
	for i=1,#target do
		input[i] = target[i]:clone()
		if (i-1) % 50 >= 20 then
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
--  local display = torch.FloatTensor(3,height,width)
--  local ground_truth = getTrueSequence(index)
--  local color = {255,0,0}
	-- temporarily switch to FloatTensor as image does not work otherwise.
	torch.setdefaulttensortype('torch.FloatTensor')
	for i = 1,#input-1 do
--    display = colorDisplay(ground_truth[i][1],output[i][1],color)
--		image.save(directory .. params.model .. '/display' .. i .. '.png', display)
    image.save(directory .. params.model .. '/output' .. i .. '.png', output[i])
	end
	torch.setdefaulttensortype(DEFAULT_TENSOR_TYPE)
end


  

evalModel_dropout(w,1)

--Save / Display network activaitons------------------------------
--input = getSequence(1)
--output = model:fullview_forward(input)
