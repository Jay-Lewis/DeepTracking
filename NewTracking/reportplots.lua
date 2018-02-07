
require('torch')
require('gnuplot')

y = torch.load('lin_costs0safe.t7')
y = torch.Tensor(y)/100
start_i = 1
end_i = 50
x=torch.linspace(start_i,end_i,end_i)*1000
gnuplot.plot(x,y)
gnuplot.xlabel('Training Iteration')
gnuplot.ylabel('Average Training Cost')


y = torch.load('lin_costs_safe.t7')
y = torch.Tensor(y)/100
start_i = 1
end_i = 70
x=torch.linspace(start_i,end_i,end_i)*1000
gnuplot.plot(x,y)
gnuplot.xlabel('Training Iteration')
gnuplot.ylabel('Average Training Cost')