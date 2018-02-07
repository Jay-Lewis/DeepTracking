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
cmd:option('-iter', 20000/100, 'the number of training iterations')
cmd:option('-N', 100, 'training sequence length')
cmd:option('-model', 'model', 'neural network model')
cmd:option('-data', 'data.t7', 'training data')
cmd:option('-learningRate', 0.01, 'learning rate')
cmd:option('-initweights', '', 'initial weights')

params = cmd:parse(arg)

if params.gpu > 0 then
	print('Using GPU ' .. params.gpu)
	require('cunn')
	require('cutorch')
	cutorch.setDevice(params.gpu)
	DEFAULT_TENSOR_TYPE = 'torch.CudaTensor'
end

width = 51
height = 51

---- load training data (new images)
--data = {}

--print('Loading training data from file')
--torch.setdefaulttensortype('torch.FloatTensor')
--for i=1,params.iter*params.N do

--img = image.load("/home/justin94lewis/Documents/DeepTracking/New_Tracking/TestData2/input"..i..".jpg",1)

--data[i] = torch.CudaTensor(2,width,height)
--data[i][1] = torch.CudaTensor():resize(img:size()):copy(img)
--data[i][2] = torch.CudaTensor():resize(img:size()):copy(img)

--end
--print('Occupancy grid has size ' .. width .. 'x' .. height)
--M = math.floor(#data / params.N) -- total number of training sequences
--print('Number of sequences ' .. M)

--torch.setdefaulttensortype(DEFAULT_TENSOR_TYPE)

----Load training data-----------------------------------

DEFAULT_TENSOR_TYPE = 'torch.FloatTensor'
torch.setdefaulttensortype(DEFAULT_TENSOR_TYPE)

print('Loading training data from file ')

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
end


-- Necessary Functions for file

-- return i-th training sequence
function getSequence(i)
	local input = {}
	for j = 1,params.N do
		input[j] = data[(i-1) * params.N + j]:type(DEFAULT_TENSOR_TYPE)
	end
	return input
end

-- filter and save model performance on a sample sequence
function evalModel(weights,index,directory)
	input = getSequence(index)
	table.insert(input, h0)
	w:copy(weights)
	local output = model:forward(input)
	-- temporarily switch to FloatTensor as image does not work otherwise.
	torch.setdefaulttensortype('torch.FloatTensor')
	for i = 1,#input-1 do
--    image.save(directory .. params.model .. '/input' .. i .. '.png',  input[i][2] / 2 + input[i][1])
		image.save(directory .. params.model .. '/output' .. i .. '.png', output[i])
	end
	torch.setdefaulttensortype(DEFAULT_TENSOR_TYPE)
end

function evalModel_dropout(weights,index,directory)
	input = getSequence(index)
  for i = math.ceil(params.N/2),params.N do
  input[i] = torch.Tensor(2,width,height)
  end
	table.insert(input, h0)
	w:copy(weights)
	local output = model:forward(input)
	-- temporarily switch to FloatTensor as image does not work otherwise.
	torch.setdefaulttensortype('torch.FloatTensor')
	for i = 1,#input-1 do
--		image.save(directory .. params.model .. '/input' .. i .. '.png',  input[i][2] / 2 + input[i][1])
		image.save(directory .. '/output' .. i .. '.png', output[i])
	end
	torch.setdefaulttensortype(DEFAULT_TENSOR_TYPE)
end


-- create directory to save videos
dir = 'BeliefImg'
lfs.mkdir(dir)

----Primary Loop ------------------------------

--for k = 1,1 do
  
----Load Network ------------------------------
---- load neural network model
--require(params.model)
---- initial hidden state
--h0 = getInitialState(width, height)
---- one step of RNN
--step = getStepModule(width, height)

---- network weights + gradients
--w, dw = step:getParameters()
--print('Model has ' .. w:numel() .. ' parameters')

--wdirectory = "weights_model/"
--wnum = k*1000
--initweights = wdirectory..wnum..".dat"
--print('Loading weights ' .. wnum..".dat" )
--w:copy(torch.load(initweights))


---- chain N steps into a recurrent neural network
--model = Recurrent(step, params.N)
  
  
---- create directory to save videos
--dir = 'BeliefImg/'..wnum.."/"
--lfs.mkdir(dir)

----Run Network on sample stream------------------------------
--dir2 = "/home/justin94lewis/Documents/DeepTracking/"..dir
--evalModel_dropout(w,1,dir2)
  
--end

-- create directory to save videos
dir = 'Belief3D'
lfs.mkdir(dir)

--Primary Loop ------------------------------
weight_nums = {1000,5000,35000}
kernel = torch.Tensor({
{0.0471,  0.2353,  0.5098, 0.4863,  0.1961,  0.0392},
{0.5137,  0.9686,  0.9922, 0.9922,  0.9255,  0.3098},
{0.8863,  0.9961,  0.9961, 0.9961,  0.9922,  0.6549},
{0.8824,  0.9961,  0.9961, 0.9961,  0.9961,  0.5922},
{0.4431,  0.9922,  0.9961, 0.9961,  0.9686,  0.1882},
{0.0235,  0.4627,  0.8627, 0.7922,  0.2196,  0.0118}})

for k = 1,#weight_nums do
  


wnum = weight_nums[k]
  
  
-- create directory to save videos
dir = 'Belief3D/'..wnum.."/"
lfs.mkdir(dir)
dir2 = "/home/justin94lewis/Documents/DeepTracking/"..dir
dir3 =  "/home/justin94lewis/Documents/DeepTracking/BeliefImg/"..wnum.."/"
print(wnum)

--Convolve Each Image with kernel------------------------------
for i = 1,100 do
  img = image.load(dir3.."output"..i..".png")
  img2 =image.convolve(img,kernel,'valid')
  max = torch.max(img2)
  img2 = torch.mul(img2,1/max)
  image.save(dir2.."Belief"..i..".png",img2)
end
  
end
