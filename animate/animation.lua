require 'torch'
require 'image'
require 'hdf5'
local win = nil
local background = torch.zeros(84, 84)
-- the special locations are:
-- 11, 32, 53
local step = function(loc, dir)
   local coin
   if loc[1] == 11 and loc[2] == 11 then
      coin = torch.random(0, 1)
      if coin == 0 then
         dir = 'd'
      else
         dir = 'r'
      end
   end
   if loc[1] == 11 and loc[2] == 32 then
      coin = torch.random(0, 2)
      if coin == 0 then
         dir = 'l'
      elseif coin == 1 then
         dir = 'd'
      elseif coin == 2 then
         dir = 'r'
      end
   end
   if loc[1] == 11 and loc[2] == 53 then
      coin = torch.random(0, 1)
      if coin == 0 then
         dir = 'd'
      else
         dir = 'l'
      end
   end
   if loc[1] == 32 and loc[2] == 11 then
      coin = torch.random(0, 2)
      if coin == 0 then
         dir = 'u'
      elseif coin == 1 then
         dir = 'd'
      elseif coin == 2 then
         dir = 'r'
      end
   end
   if loc[1] == 32 and loc[2] == 32 then
      coin = torch.random(0, 3)
      if coin == 0 then
         dir = 'u'
      elseif coin == 1 then
         dir = 'd'
      elseif coin == 2 then
         dir = 'r'
      elseif coin == 3 then
         dir = 'l'
      end
   end
   if loc[1] == 32 and loc[2] == 53 then
      coin = torch.random(0, 2)
      if coin == 0 then
         dir = 'u'
      elseif coin == 1 then
         dir = 'd'
      elseif coin == 2 then
         dir = 'l'
      end
   end
   if loc[1] == 53 and loc[2] == 11 then
      coin = torch.random(0, 1)
      if coin == 0 then
         dir = 'u'
      else
         dir = 'r'
      end
   end
   if loc[1] == 53 and loc[2] == 32 then
      coin = torch.random(0, 2)
      if coin == 0 then
         dir = 'u'
      elseif coin == 1 then
         dir = 'l'
      elseif coin == 2 then
         dir = 'r'
      end
   end
   if loc[1] == 53 and loc[2] == 53 then
      coin = torch.random(0, 1)
      if coin == 0 then
         dir = 'u'
      else
         dir = 'l'
      end
   end
   -- update location
   if dir == 'u' then
      loc[1] = loc[1] - 1
   elseif dir == 'd' then
      loc[1] = loc[1] + 1
   elseif dir == 'l' then
      loc[2] = loc[2] - 1
   elseif dir == 'r' then
      loc[2] = loc[2] + 1
   end
   return loc, dir
end

local paint = function(loc)
   local x = torch.zeros(64, 64):int()
   local i = loc[1]
   local j = loc[2]
   local h = 3
   x[{{i-h, i+h}, {j-h, j+h}}] = 1
   return x
end

function sleep(n)
  os.execute("sleep " .. tonumber(n))
end

loc = {32, 32}
dir = 'u'
win = nil
N = 1000
buf = torch.Tensor(N, 4096)
for i = 1,N do
   loc, dir = step(loc, dir)
   x = paint(loc)
   buf[i] = x:reshape(4096)
   win = image.display({image=x, win=win, zoom=4})
   -- sleep(.01)
end
-- print("Compressing file")
-- local myFile = hdf5.open('ghost.h5', 'w')
-- local options = hdf5.DataSetOptions()
-- options:setChunked(512, 512)
-- options:setDeflate()
-- myFile:write('dataset', buf, options)
-- myFile:close()
