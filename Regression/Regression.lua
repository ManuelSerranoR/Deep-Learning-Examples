--Lua code

require 'nn' --load library
require 'gnuplot'

--Funtion definitions
function f(input)

	local result = torch.Tensor(input:size()[1])

	for i = 1, input:size()[1] do
		result[i] = input[i][1]^2 + (input[i][1]*input[i][2]) + input[i][2]^2
	end

	return result
end

--Generate dataSet
dataSet = torch.Tensor(5000,2)

for i = 1, dataSet:size()[1] do
	random_X = torch.uniform(-10,10)
	random_Y = torch.uniform(-10,10)
	dataSet[i][1] = random_X
	dataSet[i][2] = random_Y
end

label = torch.Tensor(dataSet:size()[1])
label = f(dataSet)

train_DataSet = torch.Tensor(dataSet:size()[1]*0.9,dataSet:size()[2])
train_Label = torch.Tensor(label:size()[1]*0.9)

test_DataSet = torch.Tensor(dataSet:size()[1]*0.1,dataSet:size()[2])
test_Label = torch.Tensor(label:size()[1]*0.1)

for i = 1, dataSet:size()[1] do 
	if i <= train_DataSet:size()[1] then
		train_DataSet[i] = dataSet[i]
		train_Label[i] = label[i]
	else
		test_DataSet[i-train_DataSet:size()[1]] = dataSet[i]
		test_Label[i-train_DataSet:size()[1]] = label[i]
	end
end	

--Creates architecture
local model = nn.Sequential() --Creates a model that is sequential. Model is a function now.
model:add(nn.Linear(2,20))
model:add(nn.ReLU(true))
model:add(nn.Linear(20,1))

local criterion = nn.MSECriterion()

number_of_iterations = 2000


train_DataSet1 = torch.Tensor(dataSet:size()[1]*0.9,dataSet:size()[2])
train_Label1 = torch.Tensor(label:size()[1]*0.9)


local iter_num = torch.Tensor(number_of_iterations)
local loss_plot_train = torch.Tensor(number_of_iterations)
local loss_plot_test = torch.Tensor(number_of_iterations)


for i = 1, number_of_iterations do

	train_random_permutation = torch.randperm(train_DataSet:size()[1])

	for j = 1, train_DataSet:size()[1] do 
			train_DataSet1[j] = train_DataSet[train_random_permutation[j]]
			train_Label1[j] = train_Label[train_random_permutation[j]]
	end
    --print(train_DataSet1:size())
    --print("train_DataSet1train_DataSet1train_DataSet1train_DataSet1train_DataSet1train_DataSet1train_DataSet1train_DataSet1train_DataSet1train_DataSet1")
	output = model:forward(train_DataSet1)
	loss = criterion:forward(output,train_Label1)
	loss_plot_train[i] = loss
	iter_num[i] = i
	model:zeroGradParameters()

	criterion_Grad = criterion:backward(output, train_Label1)
	model:backward(train_DataSet1,criterion_Grad)

	model:updateParameters(0.0001)
	
	output_test = model:forward(test_DataSet)
	loss_test = criterion:forward(output_test,test_Label)
	loss_plot_test[i] = loss_test


end


gnuplot.pngfigure('plot_train.png')
gnuplot.plot({'Loss',  iter_num,  loss_plot_train,  '-'})
gnuplot.xlabel('Iterations')
gnuplot.ylabel('Loss')
gnuplot.plotflush()

gnuplot.pngfigure('plot_test.png')
gnuplot.plot({'Loss',  iter_num,  loss_plot_test,  '-'})
gnuplot.xlabel('Iterations')
gnuplot.ylabel('Loss')
gnuplot.plotflush()
