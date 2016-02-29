require 'nn'

local GMMKLDCriterion, parent = torch.class('nn.GMMKLDCriterion', 'nn.Criterion')

function GMMKLDCriterion:__init(weight)
    parent.__init(self)
    self.weight = weight
end

function GMMKLDCriterion:_kld(input, target)
   -- KL(target || input)
   local mu1 = target[1]:clone()
   local logv1 = target[2]:clone()
   local mu2 = input[1]:clone()
   local logv2 = input[2]:clone()

   local v1 = torch.exp(logv1)
   local v2 = torch.exp(logv2)
   
   output = (torch.add(logv2, -logv1):add(-1):addcdiv(v1, v2):
                     addcdiv((mu2 - mu1):pow(2), v2))

   return output:sum(input[1]:size():size()) * 0.5
end

function GMMKLDCriterion:_kld_backward(input, target)
   -- KL(target || input)
   local mu1 = target[1]:clone()
   local logv1 = target[2]:clone()
   local mu2 = input[1]:clone()
   local logv2 = input[2]:clone()
   
   local v1 = torch.exp(logv1)
   local v2 = torch.exp(logv2)

   local diff12 = mu1:add(-mu2)
   local dmu1 = torch.cdiv(diff12, v2)
   local dmu2 = torch.cdiv(-diff12, v2)
   local div12 = torch.cdiv(v1, v2)
   local dlogv1 = div12:clone():add(-1):div(2)
   -- be careful: use of inplace
   local dlogv2 = div12:mul(-1):add(1):add(-diff12:pow(2):cdiv(v2)):div(2)

   -- return grad w.r.t. input first
   gradInput = {dmu2, dlogv2, dmu1, dlogv1}
   return gradInput
end

function GMMKLDCriterion:_KLD(input, target)
   -- KL(target || input)
   -- special structure: target is Gaussian, input is GMM
   -- input: {mu1, logv1, mu2, logv2, ..., mu3, logv3, pi}
   -- target: {mu, logv}
   local K = (#input-1)/2
   local N = input[1]:size(1)
   local KLD = torch.zeros(N, K):cuda()
   local pi = input[#input]
   for i=1,K do
      local kld = self:_kld({input[i], input[i+K]}, target)
      KLD[{{},i}] = kld
   end
   return KLD
end

function GMMKLDCriterion:updateOutput(input, target)
   self.KLD = self:_KLD(input, target)
   self.exp_nKLD = self.KLD:clone():mul(-1):exp()
   local pi = input[#input]
   self.pi_exp_nKLD = self.exp_nKLD:clone():cmul(pi)
   self.denom = self.pi_exp_nKLD:clone():sum(2)
   self.output = self.denom:clone():log():mul(-1):sum()*self.weight
   return self.output
end

function GMMKLDCriterion:updateGradInput(input, target)
   if self.KLD == nil then
      self:updateOutput(input, target)
   end
   local KLD = self.KLD
   local exp_nKLD = self.exp_nKLD
   local pi_exp_nKLD = self.pi_exp_nKLD
   local denom = self.denom
   -- dpi
   local dpi = exp_nKLD:mul(-1):cdiv(denom:expand(#exp_nKLD))
   self.gradInput = {}
   self.gradInput[#input] = dpi
   -- -- the remaining derivatives
   local dKLD = pi_exp_nKLD:cdiv(denom:expand(#pi_exp_nKLD))
   local K = (#input-1)/2
   local N = input[1]:size(1)
   local D = input[1]:size(2)
   -- I need to return: {dpmu1, dpmu2, ..., dplogv1, dplogv2,... dpi, dmu, dlogv}
   self.gradInput[#input+1] = torch.zeros(#target[1]):cuda()
   self.gradInput[#input+2] = torch.zeros(#target[1]):cuda()
   for i=1,K do
      local dpmu, dplogv, dmu, dlogv = unpack(self:_kld_backward({input[i], input[i+K]}, target))
      -- -- dpmui and dplogvi
      local section = dKLD[{{},i}]:reshape(N,1):expand(N,D)
      self.gradInput[i] = dpmu:cmul(section)
      self.gradInput[i+K] = dplogv:cmul(section)
      -- -- -- dmu and dlogv
      self.gradInput[#input+1]:add(dmu:cmul(section))
      self.gradInput[#input+2]:add(dlogv:cmul(section))
   end
   for i=1,#self.gradInput do
      self.gradInput[i]:mul(self.weight)
   end
   -- flush cache
   self.KLD = nil
   return self.gradInput
end

