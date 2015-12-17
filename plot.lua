require 'nn'
require 'json'
local m = require 'manifold'
local emb = torch.load("embedding.th7")
local s = emb.weight:size()
local ndims = s[1]
local nheroes = s[2]
local p = m.embedding.tsne(emb.weight:t(), {dim=2})
--print(p)
local heroes = json.load("heroes.json")
for i=1, p:size(1) do
    print(p[i][1], p[i][2], heroes[tostring(i-1)])
end
--print(heroes)
