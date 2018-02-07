
require('torch')
require('gnuplot')

--y = torch.load('lin_costs0safe.t7')
--y = torch.Tensor(y)/100
--start_i = 1
--end_i = 60
--x=torch.linspace(start_i,end_i,end_i)*1000
--gnuplot.plot(x,y)
--gnuplot.xlabel('Training Iteration')
--gnuplot.ylabel('Average Training Cost')



--y2 = torch.load('lin_costs1safe.t7')
--y2 = torch.Tensor(y)/100
--start_i = 1
--end_i = 30
--x2=torch.linspace(start_i,end_i,end_i)*1000
--gnuplot.plot(x,y)
--gnuplot.xlabel('Training Iteration')
--gnuplot.ylabel('Average Training Cost')


y = torch.load('lin_costs0safe.t7')
y = torch.Tensor(y)/100
start_i = 1
end_i = 60
x=torch.linspace(start_i,end_i,end_i)*1000

y2 = torch.load('lin_costs1safe.t7')
y2 = torch.Tensor(y2)/100
start_i = 1
end_i = 30
x2=torch.linspace(start_i,end_i,end_i)*1000

gnuplot.plot({'Original',x,y,'-'},{'New',x2,y2,'-'})
gnuplot.xlabel('Training Iteration')
gnuplot.ylabel('Average Training Cost')


--y = torch.load('lin_costs1safe.t7')
--y = torch.Tensor(y)/100
--start_i = 1
--end_i = 30
--x=torch.linspace(start_i,end_i,end_i)*1000

--x2 = torch.linspace(end_i+1,60,end_i)*1000
--y2 = torch.ones(end_i)*0.017
--x=torch.cat(x,x2)
--y=torch.cat(y,y2)

--gnuplot.plot(x,y)
--gnuplot.xlabel('Training Iteration')
--gnuplot.ylabel('Average Training Cost')