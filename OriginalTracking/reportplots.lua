
require('torch')
require('gnuplot')

y = torch.load('lin_costs.t7')
y = torch.Tensor(y)/5
x=torch.linspace(1,50,50)*1000
gnuplot.plot(x,y)
gnuplot.xlabel('Training Iteration')
gnuplot.ylabel('Average Training Cost')
