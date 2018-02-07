require('torch')
require('lfs')
require('nngraph')
require('optim')
require('image')
require('cunn')
require('cutorch')

print('Loading training data from file')
torch.setdefaulttensortype('torch.FloatTensor')
width = 51
height = 51
imagenum = 200
data = {}

--Create Plots for Paper----------------------------------------------

for i=1,imagenum do

name = "/home/justin94lewis/Documents/DeepTracking/New_Tracking/TestData_png/input"..i..".png"

local f=io.open(name,"r")

if f~=nil then
io.close(f)
img = image.load(name,1)


image.save("/home/justin94lewis/Documents/DeepTracking/New_Tracking/ReportPlots/input"..i..".png",img)

end
end




--Create New Dataset----------------------------------------------

--for i=1,imagenum do

--name = "/home/justin94lewis/Documents/DeepTracking/New_Tracking/TestData_png/input"..i..".png"
--local f=io.open(name,"r")

--if f~=nil then
--io.close(f)
--img = image.load(name,1)
--img = torch.ceil(img)

--data[i] = torch.CudaTensor(2,width,height)
--data[i][1] = torch.CudaTensor():resize(img:size()):copy(img)
--data[i][2] = torch.CudaTensor():resize(img:size())

--if(i%1000 == 0) then
--print(i)

--end
--collectgarbage()
--end


--end

--torch.save('/home/justin94lewis/Documents/DeepTracking/New_Tracking/tinydata2.t7',data)



---- Test Image Transform---------------------------------------------------

--height = 51
--width = 51

--for i=1,10 do

--name = "/home/justin94lewis/Documents/DeepTracking/New_Tracking/TestData2/input"..i..".jpg"
--local f=io.open(name,"r")

--if f~=nil then
--io.close(f)
--img = image.load(name,1)

--for j = 1,width do
--    for k = 1,height do
--      if(img[j][k] >= 0.06) then
--        img[j][k] = 1
--      else
--        img[j][k] = 0
--      end
--    end
--end

--image.save("/home/justin94lewis/Documents/DeepTracking/New_Tracking/Test/input"..i..".png",img)

--end
--	collectgarbage()

--end

--torch.save('/home/justin94lewis/Documents/DeepTracking/New_Tracking/trainingdata2.t7',data)