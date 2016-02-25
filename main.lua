require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
nngraph.setDebug(false)
-- load custom modules
require 'load'
require 'KLDCriterion'
require 'Sampler'
local CVAE = require 'CVAE'
local kld = nn.KLDCriterion()
local bce = nn.BCECriterion()
bce.sizeAverage = false
local use_dataset = "flipshift"
local data, train, masked_train, batch_size, x_size, y_size, z_size, hidden_size
if use_dataset == "mnist" then
   -- get data
   data = loadmnist()
   train = data.train
   masked_train = train:clone()
   masked_train[{{},{1,392}}] = 0
   -- settings
   batch_size = 200
   x_size = 784
   y_size = 784
   z_size = 2
   hidden_size = 400
else
   -- get data
   data = loadflipshift()
   train = data.train
   masked_train = train:clone()
   masked_train[{{},{1,2048}}] = 0
   -- settings
   batch_size = 200
   x_size = 4096
   y_size = 4096
   z_size = 2
   hidden_size = 400
end
local prior = CVAE.create_prior_network(x_size, z_size, hidden_size)
local encoder = CVAE.create_encoder_network(x_size, y_size, z_size, hidden_size)
local decoder = CVAE.create_decoder_network(x_size, y_size, z_size, hidden_size)
local sampler = nn.Sampler()
-- construct graph
local x_input = nn.Identity()()
local y_input = nn.Identity()()
local pmu, plogv = prior(x_input):split(2)
local mu, logv = encoder({x_input, y_input}):split(2)
local code = sampler({mu, logv})
local recon = decoder({x_input, code})
local model = nn.gModule({x_input, y_input}, {pmu, plogv, mu, logv, recon})
-- retain parameters and gradients
local parameters, gradients = model:getParameters()
-- optimization function
local config = {learningRate = 0.001}
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
   local pmu, plogv, mu, logv, recon = unpack(model:forward({x_input, y_input}))
   local bce_err = bce:forward(recon, y_input)
   local drecon = bce:backward(recon, y_input)
   -- backprop
   local kld_err = kld:forward({pmu, plogv}, {mu, logv})
   local dpmu, dplogv, dmu, dlogv = unpack(
      kld:backward({pmu, plogv}, {mu, logv})
   )
   local error_grads = {dpmu, dplogv, dmu, dlogv, drecon}
   model:backward({x_input, y_input}, error_grads)
   return bce_err, kld_err, gradients
end
-- training
local epoch = 0
local lowerbound_status = 0
local bce_status = 0
local kld_status = 0
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
         local bce_err, kld_err, gradients = opfunc(parameters_input,
                                                    x_input, y_input)
         -- shamelessly break encapsulation (again)
         -- accumulate bce, kld, and lowerbound statistics in moving average
         local neg_lowerbound = bce_err + kld_err
         lowerbound_status = 0.99*lowerbound_status - 0.01*neg_lowerbound
         bce_status = 0.99*bce_status + 0.01*bce_err
         kld_status = 0.99*kld_status + 0.01*kld_err
         return neg_lowerbound, gradients
      end
      -- pass in parameters to do in-place update
      optim.adam(innerfunc, parameters, config, state)
   end
   -- print my progress
   print("Epoch: " .. epoch)
   print("Running Averages:")
   print(".. Lowerbound: " .. lowerbound_status/batch_size)
   print(".. Bernoulli cross entropy: " .. bce_status/batch_size)
   print(".. Gaussian KL divergence: " .. kld_status/batch_size)
   -- save
   torch.save('save/' .. use_dataset .. '_CVAE_z' .. z_size .. '.t7',
              {state=state,
               config=config,
               model=model,
               prior=prior,
               encoder=encoder,
               decoder=decoder})
end
