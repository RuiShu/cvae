require 'nn'

local KLDCriterion, parent = torch.class('nn.KLDCriterion', 'nn.Criterion')

function KLDCriterion:updateOutput(input, target)
   -- KL(target || input)
   local mu1 = target[1]:clone()
   local logv1 = target[2]:clone()
   local mu2 = input[1]:clone()
   local logv2 = input[2]:clone()

   local v1 = torch.exp(logv1)
   local v2 = torch.exp(logv2)
   
   self.output = (torch.csub(logv2, logv1):csub(1):addcdiv(v1, v2):
                     addcdiv((mu2 - mu1):pow(2), v2))

   return self.output:sum() * 0.5
end

function KLDCriterion:updateGradInput(input, target)
   -- KL(target || input)
   local mu1 = target[1]:clone()
   local logv1 = target[2]:clone()
   local mu2 = input[1]:clone()
   local logv2 = input[2]:clone()
   
   local v1 = torch.exp(logv1)
   local v2 = torch.exp(logv2)

   local diff12 = mu1:csub(mu2)
   local dmu1 = torch.cdiv(diff12, v2)
   local dmu2 = torch.cdiv(-diff12, v2)
   local div12 = torch.cdiv(v1, v2)
   local dlogv1 = div12:clone():csub(1):div(2)
   -- be careful: use of inplace
   local dlogv2 = div12:mul(-1):add(1):csub(diff12:pow(2):cdiv(v2)):div(2)

   -- return grad w.r.t. input first
   self.gradInput = {dmu2, dlogv2, dmu1, dlogv1}
   return self.gradInput
end

