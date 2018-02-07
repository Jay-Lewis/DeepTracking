require('torch')
require('lfs')
require('nngraph')
require('optim')
require('image')
require('cunn')
require('cutorch')
  
data = {}
print('Loading training data from file')
torch.setdefaulttensortype('torch.FloatTensor')
width = 51
height = 51

test = torch.load("/home/justin94lewis/Documents/DeepTracking/tinydata.t7")

--Create New Dataset----------------------------------------------

for i=50,100 do

name = "/home/justin94lewis/Documents/DeepTracking/New_Tracking/TestData_png/input"..i..".png"
local f=io.open(name,"r")

if f~=nil then
io.close(f)
img = image.load(name,1)
image.save("/home/justin94lewis/Documents/DeepTracking/New_Tracking/Test/".."input"..i..".png",img)
img = torch.add(img,-0.5)
img = torch.ceil(img)

image.save("/home/justin94lewis/Documents/DeepTracking/New_Tracking/Test/".."new"..i..".png",img)

--data[i] = torch.Tensor(2,width,height)
--data[i][1] = torch.Tensor():resize(img:size()):copy(img)
--data[i][2] = torch.Tensor():resize(img:size())
--print(torch.max(img))
--if(i%1000 == 0) then
--print(i)
--collectgarbage()
--end

end


end
