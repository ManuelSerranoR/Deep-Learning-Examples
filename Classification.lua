--Lua code

require 'nn' --load library
require 'gnuplot'

dataSet = torch.load('dataset.t7b')

--print(data.label)

--Creamos una lista de 4000 elementos con numeros aleatorios del 1 al 4000 sin repetirse, para usarlos de index
data_permuted = torch.randperm(dataSet.label:size()[1]) --Colon to access function. We permute.

train_DataSet = torch.Tensor(dataSet.data:size()[1]*0.9,dataSet.data:size()[2])
test_DataSet = torch.Tensor(dataSet.data:size()[1]*0.1,dataSet.data:size()[2])
train_Label = torch.Tensor(dataSet.label:size()[1]*0.9)
test_Label = torch.Tensor(dataSet.label:size()[1]*0.1)


for i = 1, dataSet.data:size()[1] do 
	if i <= train_DataSet:size()[1] then
		train_DataSet[i] = dataSet.data[data_permuted[i]]
		train_Label[i] = dataSet.label[data_permuted[i]]
	else
		test_DataSet[i-train_DataSet:size()[1]] = dataSet.data[data_permuted[i]]
		test_Label[i-train_DataSet:size()[1]] = dataSet.label[data_permuted[i]]
	end
end	


local model = nn.Sequential() --Creates a model that is sequential. Model is a function now.
model:add(nn.Linear(2,20))
model:add(nn.ReLU(true))
model:add(nn.Linear(20,4))
model:add(nn.LogSoftMax())

local criterion = nn.ClassNLLCriterion()


number_of_iterations = 500


local iter_num = torch.Tensor(number_of_iterations)
local loss_plot_train = torch.Tensor(number_of_iterations)
local loss_plot_test = torch.Tensor(number_of_iterations)

local accuracy_plot_train = torch.Tensor(number_of_iterations)
local accuracy_plot_test = torch.Tensor(number_of_iterations)


train_DataSet1 = torch.Tensor(dataSet.data:size()[1]*0.9,dataSet.data:size()[2])
train_Label1 = torch.Tensor(dataSet.label:size()[1]*0.9)


for i = 1, number_of_iterations do

	train_random_permutation = torch.randperm(train_DataSet:size()[1])

	for j = 1, train_DataSet:size()[1] do 
			train_DataSet1[j] = train_DataSet[train_random_permutation[j]]
			train_Label1[j] = train_Label[train_random_permutation[j]]
	end

	output = model:forward(train_DataSet1)
	loss = criterion:forward(output,train_Label1)
	loss_plot_train[i] = loss
	iter_num[i] = i
	model:zeroGradParameters()


	accuracy_train = torch.Tensor(4)
	train_Label1_aux=train_Label1:long()
	no_need1, accuracy_train = torch.max(output, 2)
	guessed_right_train = accuracy_train:eq(train_Label1_aux):sum()
	accuracy_plot_train[i] = guessed_right_train

	criterion_Grad = criterion:backward(output, train_Label1)
	model:backward(train_DataSet1,criterion_Grad)

	model:updateParameters(0.2)
	
	output_test = model:forward(test_DataSet)
	loss_test = criterion:forward(output_test,test_Label)
	loss_plot_test[i] = loss_test


	accuracy_test = torch.Tensor(4)
	test_Label_aux=test_Label:long()
	no_need, accuracy_test = torch.max(output_test, 2)
	guessed_right_test = accuracy_test:eq(test_Label_aux):sum()
	accuracy_plot_test[i] = guessed_right_test
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

gnuplot.pngfigure('accuracy_plot_train.png')
gnuplot.plot({'Accuracy',  iter_num,  accuracy_plot_train,  '-'})
gnuplot.xlabel('Iterations')
gnuplot.ylabel('Accuracy')
gnuplot.plotflush()

gnuplot.pngfigure('accuracy_plot_test.png')
gnuplot.plot({'Accuracy',  iter_num,  accuracy_plot_test,  '-'})
gnuplot.xlabel('Iterations')
gnuplot.ylabel('Accuracy')
gnuplot.plotflush()
