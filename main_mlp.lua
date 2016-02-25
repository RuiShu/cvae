require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
nngraph.setDebug(false)
-- load custom modules
require 'load'
require 'KLDCriterion'
require 'Sampler'
local MLP = require 'MLP'
local bce = nn.BCECriterion()
bce.sizeAverage = false
-- get data
local data = loadmnist()
local train = data.train
local masked_train = train:clone()
masked_train[{{},{1,392}}] = 0
-- settings
local batch_size = 200
local x_size = 784
local y_size = 784
local hidden_size = 400
-- create network
local model = MLP.create_network(x_size, y_size, hidden_size)
-- retain parameters and gradients
local parameters, gradients = model:getParameters()
-- optimization function
local config = {learningRate = 0.01}
local state = {}
local opfunc = function(parameters_input, x_input, y_input)
   -- uses the following outside of encapsulation:
   -- model, bce, kde, parameters, gradients
   if parameters_input ~= parameters then
      print("does this ever happen?")
      parameters:copy(parameters_input)
   end
   -- forward
   model:zeroGradParameters()
   local recon = model:forward(x_input)
   local bce_err = bce:forward(recon, y_input)
   local drecon = bce:backward(recon, y_input)
   model:backward(x_input, drecon)
   return bce_err, gradients
end
-- training
local epoch = 0
local bce_status = 0
while epoch < 10 do
   -- set up status
   local tic = torch.tic()
   epoch = epoch + 1
   -- create batches
   local indices = torch.randperm(train:size(1)):long():split(batch_size)
   indices[#indices] = nil
   local N = #indices * batch_size
   -- update learning rate
   if epoch % 5 == 0 then
      config.learningRate = config.learningRate/10
      print("New learning rate: " .. config.learningRate)
   end
   -- loop through minibatch
   for t, v in ipairs(indices) do
      xlua.progress(t, #indices)
      local x_input = masked_train:index(1, v)
      local y_input = train:index(1, v)
      local innerfunc = function(parameters_input)
         local bce_err, gradients = opfunc(parameters_input,
                                           x_input, y_input)
         -- shamelessly break encapsulation (again)
         -- accumulate bce, kld, and lowerbound statistics in moving average
         bce_status = 0.99*bce_status + 0.01*bce_err
         return bce_err, gradients
      end
      -- pass in parameters to do in-place update
      optim.adam(innerfunc, parameters, config, state)
   end
   -- print my progress
   print("Epoch: " .. epoch)
   print("Running Average:")
   print(".. Bernoulli cross entropy: " .. bce_status/batch_size)
   -- save
   torch.save('save/MLP.t7',
              {state=state,
               config=config,
               model=model})
end
