require 'hdf5'
require 'torch'
require 'nn'
require 'nngraph'
require 'cunn'
require 'cutorch'
nngraph.setDebug(false)
-- load custom modules
require 'load'
require 'Sampler'
require 'image'
-- load init data
f = hdf5.open('animate/ghost.h5', 'r')
init = f:read("init"):all():double():cuda()
-- cvae
sampler = nn.Sampler():cuda()
saved = torch.load('save/ghost_CVAE_z1.t7')
model = saved.model
prior = saved.prior
encoder = saved.encoder
decoder = saved.decoder
-- mlp
saved = torch.load('save/ghost_MLP.t7')
mlp = saved.model
-- setup
x_input = torch.Tensor(8192):cuda()
x_input[{{1,4096}}] = init[1]
x_input[{{4097,8192}}] = init[2]
N = 2000
win = nil
-- for i = 1,N do
--    pmu, plogv = unpack(prior:forward(x_input))
--    code = sampler:forward({pmu, plogv})
--    recon = decoder:forward({x_input, code})
--    x_input[{{1,4096}}] = x_input[{{4097,8192}}]
--    x_input[{{4097,8192}}] = recon
--    win = image.display({image=recon:reshape(64,64), win=win, zoom=4})
-- end
for i = 1,N do
   recon = mlp:forward(x_input)
   x_input[{{1,4096}}] = x_input[{{4097,8192}}]
   x_input[{{4097,8192}}] = recon
   win = image.display({image=recon:reshape(64,64), win=win, zoom=4})
end
