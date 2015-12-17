require 'csvigo'
require 'json'
require 'nn'
require 'optim'


-- preprocessing / data loading
local function tlength(t)
    local n = 0
    for _, _ in pairs(t) do
        n = n + 1
    end
    return n
end

local heroes = json.load("heroes.json")
local nheroes = tlength(heroes) / 2
print("nheroes", nheroes)
local eye = torch.eye(nheroes)
local x = csvigo.load({path="gosugamers_x.csv", mode="large"})
local y = csvigo.load({path="gosugamers_y.csv", mode="large"})
local teams1 = torch.Tensor(#y/2, 5, nheroes)
local teams2 = torch.Tensor(#y/2, 5, nheroes)
local ys = torch.Tensor(#y/2, 1)
for i=1, #y, 2 do
    local team1 = x[i]
    local team2 = x[i+1]
    local winner = tonumber(y[i][1]) == 0 and 1 or 0
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


-- model
local emb_size = 20
local emb1 = nn.Linear(nheroes, emb_size)
--local emb2 = emb1:clone('weight', 'bias')
local emb2 = nn.Linear(nheroes, emb_size)
local model = nn.Sequential()
model:add(nn.ParallelTable():add(emb1):add(emb2))
model:add(nn.JoinTable(1))
model:add(nn.Sigmoid())
model:add(nn.Linear(2*emb_size, 1))
model:add(nn.Sigmoid())
print(model)

local crit = nn.BCECriterion()


-- heuristic for halving the learning rate
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


-- configure training 
local params, dparams = model:getParameters()
local adagrad = false
local adam = false
--local adaconfig = {weightDecay=0.01, nesterov=true, momentum=0.9, lr=0.1, lrd=0.0001, dampening=0}
local adaconfig = {}
local adastate = {}
local nepoch = 20000
local lr = 0.1
local prob = torch.ones(ys:size(1))
-- extract a validation dataset
local nvalid = 500
local ntrain = ys:size(1) - nvalid
local valid_inds = torch.multinomial(prob, nvalid, false)
local valid_mask = torch.zeros(ys:size(1)):byte()
for i=1, valid_inds:size(1) do
    valid_mask[valid_inds[i]] = 1
end
local valid_ys = ys[valid_mask]:view(-1, 1)
local train_mask = valid_mask:le(0.5)
local train_ys = ys[train_mask]:view(-1, 1)
valid_mask = valid_mask:view(valid_mask:size(1), 1, 1):expand(teams1:size())
train_mask = valid_mask:le(0.5)
local valid_teams1 = teams1[valid_mask]:view(nvalid, teams1:size(2), teams1:size(3))
local valid_teams2 = teams2[valid_mask]:view(nvalid, teams2:size(2), teams2:size(3))
local train_teams1 = teams1[train_mask]:view(ntrain, teams1:size(2), teams1:size(3))
local train_teams2 = teams2[train_mask]:view(ntrain, teams2:size(2), teams2:size(3))
print("train", train_ys:size())
print("valid", valid_ys:size())
print("bias in wins", valid_ys:mean())
-- training
for epoch=1, nepoch do
    local avgtrerr = 0
    for i=1, train_ys:size(1) do
        local x1 = train_teams1[i]:sum(1):squeeze()
        local x2 = train_teams2[i]:sum(1):squeeze()
        local y = train_ys[i]
        local err
        local function feval(par)
            assert(par == params)
            local err = crit:forward(model:forward({x1, x2}), y)
            --if epoch > 1 then
            --    lr = check_progress(i, err, lr)
            --end
            local grad = crit:backward(model.output, y)
            model:zeroGradParameters()
            model:backward({x1, x2}, grad)
            return err, dparams
        end
        if adagrad then
            _, err = optim.adagrad(feval, params, adaconfig, adastate)
            err = err[1]
        elseif adam then
            _, err = optim.adam(feval, params, adaconfig, adastate)
            err = err[1]
        else
            --err, _ = feval(params)
            --model:updateParameters(lr)
            _, err = optim.sgd(feval, params, adaconfig, adastate)
            err = err[1]
        end
        avgtrerr = avgtrerr + err
    end
    local avgvalerr = 0
    local ncorrects = 0
    for i=1, valid_ys:size(1) do
        local x1 = valid_teams1[i]:sum(1):squeeze()
        local x2 = valid_teams2[i]:sum(1):squeeze()
        local y = valid_ys[i]
        avgvalerr = avgvalerr + crit:forward(model:forward({x1, x2}), y)
        if y == torch.round(model.output[1]) then
            ncorrects = ncorrects + 1
        end
    end
    print("average train error this epoch", avgtrerr/train_ys:size(1))
    print("average valid error this epoch", avgvalerr/valid_ys:size(1))
    print("average valid acc this epoch", ncorrects/valid_ys:size(1))
end

torch.save("model.th7", model)
torch.save("embedding.th7", emb1)
