require 'torch'
require 'nn'

local MLP = {}

function MLP.create_network(x_size, y_size, hidden_size)
   -- the mlp network
   local model = nn.Sequential()
   model:add(nn.Linear(x_size, hidden_size))
   model:add(nn.ReLU(true))
   model:add(nn.Linear(hidden_size, y_size))
   model:add(nn.Sigmoid(true))
   return model
end
   
return MLP
