require 'csvigo'
require 'json'
require 'nn'

local function tlength(t)
    local n = 0
    for _, _ in pairs(t) do
        n = n + 1
    end
    return n
end

local heroes = json.load("heroes.json")
local nheroes = tlength(heroes) / 2
local eye = torch.eye(nheroes)
local x = csvigo.load({path="gosugamers_x.csv", mode="large"})
local y = csvigo.load({path="gosugamers_y.csv", mode="large"})
local teams1 = torch.Tensor(#y/2, 5, nheroes)
local teams2 = torch.Tensor(#y/2, 5, nheroes)
local ys = torch.Tensor(#y/2, 1)
for i=1, #y, 2 do
    local team1 = x[i]
    local team2 = x[i+1]
    local winner = y[i] == 0 and 1 or 0
    for j, h in pairs(team1) do
        teams1[math.ceil(i/2)][j] = eye[tonumber(h)+1]
    end
    for j, h in pairs(team2) do
        teams2[math.ceil(i/2)][j] = eye[tonumber(h)+1]
    end
    ys[math.ceil(i/2)] = winner
end
print(teams1:size())
print(teams2:size())
print(ys:size())

local emb_size = 20
local emb1 = nn.Linear(nheroes, emb_size)
local emb2 = emb1:clone('weight', 'bias')
local model = nn.Sequential()
model:add(nn.ParallelTable():add(emb1):add(emb2))
model:add(nn.JoinTable(1))
model:add(nn.Linear(2*emb_size, 1))
model:add(nn.Sigmoid())
print(model)

local crit = nn.BCECriterion()
local lr = 0.01

local smoothing_size = 42
local err_buf_size = 1000
local err_smoothing = torch.ones(smoothing_size)
local err_base_size = err_buf_size+1-smoothing_size
local err_base = torch.ones(err_base_size, 2)
err_base[{{}, {1}}] = torch.linspace(1, err_base_size, err_base_size)
local err_buf = torch.zeros(err_buf_size)
local function check_progress(ind, err, lr)
    err_buf[(ind-1) % err_buf:size(1) + 1] = err
    if ((ind-1) % err_buf:size(1)) == 0 then -- every 1000
        local tmp = torch.conv2(err_buf:view(-1, 1),
                                err_smoothing:view(-1, 1))
        local coeff = torch.gels(tmp, err_base)[1][1]
        if coeff > 0 or coeff < 1E-8 then
            print("halving the learning rate, avg error is:", err_buf:mean())
            return lr/2
        end
    end
    return lr
end

local nepoch = 100
for epoch=1, nepoch do
    for i=1, ys:size(1) do
        local x1 = teams1[i]:sum(1):squeeze()
        local x2 = teams2[i]:sum(1):squeeze()
        local y = ys[i]
        local err = crit:forward(model:forward({x1, x2}), y)
        if epoch > 1 then
            lr = check_progress(i, err, lr)
        end
        local grad = crit:backward(model.output, y)
        model:zeroGradParameters()
        model:backward({x1, x2}, grad)
        model:updateParameters(lr)
        --print(err)
    end
end

