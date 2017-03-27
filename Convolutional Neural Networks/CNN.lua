--Lua code

require 'nn' --load library
require 'gnuplot'

trainData = torch.load('mnist.t7/train_32x32.t7','ascii')
testData = torch.load('mnist.t7/test_32x32.t7','ascii')

print 'Loading dataset...'


train_dataSet = torch.Tensor(0.1*trainData.data:size()[1],trainData.data:size()[2],trainData.data:size()[3],trainData.data:size()[4])
train_label = torch.Tensor(0.1*trainData.data:size()[1])
for i = 1, 0.1*trainData.data:size()[1] do
	train_dataSet[i] = trainData.data[torch.random(1,trainData.data:size()[1])]
	train_label[i] = trainData.labels[torch.random(1,trainData.data:size()[1])]

end

test_dataSet = torch.Tensor(0.1*testData.data:size()[1],testData.data:size()[2],testData.data:size()[3],testData.data:size()[4])
test_label = torch.Tensor(0.1*testData.data:size()[1])

for i = 1, 0.1*testData.data:size()[1] do
	test_dataSet[i] = testData.data[torch.random(1,testData.data:size()[1])]
	test_label[i] = testData.labels[torch.random(1,testData.data:size()[1])]
end
batch_size=256

model = nn.Sequential()
model:add(nn.SpatialConvolution(1, 20, 5, 5))
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
model:add(nn.SpatialConvolution(20, 50, 5, 5))
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

model:add(nn.View(50*5*5)) --reshapes to view data at 50x4x4
model:add(nn.Linear(50*5*5, 500))
model:add(nn.ReLU())
model:add(nn.Linear(500, 10))
--encoding layer - go from 10 class out to 2-dimensional encoding
model:add(nn.Linear(10, 2))


--The siamese model
siamese_model = nn.ParallelTable()
siamese_model:add(model)
siamese_model:add(model:clone('weight','bias', 'gradWeight','gradBias')) 

final_model= nn.Sequential()
final_model:add(nn.Reshape(batch_size,1,2,train_dataSet:size()[3],train_dataSet:size()[4]))
final_model:add(nn.SplitTable(3))
final_model:add(siamese_model)
final_model:add(nn.PairwiseDistance(2))


margin = 1
criterion = nn.HingeEmbeddingCriterion(margin)

number_of_iterations=80

local iter_num = torch.Tensor(number_of_iterations)
local loss_plot_train = torch.Tensor(number_of_iterations)
local loss_plot_test = torch.Tensor(number_of_iterations)

for i = 1, number_of_iterations do
	data=torch.Tensor(batch_size,2,train_dataSet:size()[3],train_dataSet:size()[4])
	label=torch.Tensor(batch_size)
	for j=1,batch_size do
		data1_rand=torch.random(1,train_dataSet:size()[1])
		data2_rand=torch.random(1,train_dataSet:size()[1])
		data[j]=torch.cat(train_dataSet[data1_rand],train_dataSet[data2_rand],1)
		if train_label[data1_rand]==train_label[data2_rand] then
			label[j]=1
		else
			label[j]=-1
		end
	end

	output=final_model:forward(data)
	loss = criterion:forward(output,label)
	--print(loss)
	loss_plot_train[i] = loss
	iter_num[i] = i
	model:zeroGradParameters()

	criterion_Grad = criterion:backward(output, label)
	final_model:backward(data,criterion_Grad)

	final_model:updateParameters(0.01)
end

gnuplot.pngfigure('plot_train.png')
gnuplot.plot({'Loss',  iter_num,  loss_plot_train,  '-'})
gnuplot.xlabel('Iterations')
gnuplot.ylabel('Loss')
gnuplot.plotflush()




