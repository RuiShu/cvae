function TableConcat(T)
   t = {}
   idx = 0
   for i=1,#T do
      inner_t = T[i]
      for j=1,#inner_t do
	 idx = idx + 1
	 t[idx] = inner_t[j]
      end
   end
   return t
end
