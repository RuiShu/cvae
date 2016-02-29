require 'torch'
require 'nn'

local VAE = {}

function VAE.create_encoder_network(x_size, y_size, z_size, hidden_size)
   -- the encoder network
   local encoder = nn.Sequential()
   encoder:add(nn.JoinTable(1,1))
   encoder:add(nn.Linear(x_size + y_size, hidden_size))
   encoder:add(nn.ReLU(true))

   -- construct mu and log variance in parallel
   local mu_logv = nn.ConcatTable()
   mu_logv:add(nn.Linear(hidden_size, z_size))
   mu_logv:add(nn.Linear(hidden_size, z_size))
   encoder:add(mu_logv)
   return encoder
end

function VAE.create_prior_network(x_size, z_size, hidden_size)
   -- the prior network
   local prior = nn.Sequential()
   prior:add(nn.Linear(x_size, hidden_size))
   prior:add(nn.ReLU(true))

   -- construct mu and log variance in parallel
   local mu_logv = nn.ConcatTable()
   mu_logv:add(nn.Linear(hidden_size, z_size))
   mu_logv:add(nn.Linear(hidden_size, z_size))
   prior:add(mu_logv)
   return prior
end

function VAE.create_decoder_network(x_size, y_size, z_size, hidden_size)
   -- the decoder network
   local decoder = nn.Sequential()
   decoder:add(nn.JoinTable(1,1))
   decoder:add(nn.Linear(x_size + z_size, hidden_size))
   decoder:add(nn.ReLU(true))
   decoder:add(nn.Linear(hidden_size, y_size))
   decoder:add(nn.Sigmoid(true))

   return decoder
end
   
return VAE
