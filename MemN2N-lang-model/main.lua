-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

require('xlua')
require('paths')
local tds = require('tds')
paths.dofile('data.lua')
paths.dofile('model.lua')
torch.setdefaulttensortype('torch.FloatTensor')

--------------------------------------------------------------------
--------------------------------------------------------------------
-- model params:
local cmd = torch.CmdLine()
cmd:option('--gpu', 0, 'GPU id to use - 1 indexed in lua')
cmd:option('--edim', 150, 'internal state dimension')
cmd:option('--lindim', 75, 'linear part of the state')
cmd:option('--init_std', 0.05, 'weight initialization std')
cmd:option('--init_hid', 0.1, 'initial internal state value')
cmd:option('--sdt', 0.01, 'initial learning rate')
cmd:option('--maxgradnorm', 7, 'maximum gradient norm')
cmd:option('--memsize', 100, 'memory size')
cmd:option('--nhop', 6, 'number of hops')
cmd:option('--batchsize', 128)
cmd:option('--show', true, 'print progress')
cmd:option('--load', '', 'model file to load')
cmd:option('--save', '', 'path to save model')
cmd:option('--epochs', 100)
cmd:option('--test', false, 'enable testing')
opt = cmd:parse(arg or {})
print(opt)

if opt.gpu > 0 then
	function transfer_data(x)
		return x:cuda()
	end
	cutorch.setDevice(opt.gpu)
else
	function transfer_data(x)
		return x
	end
end

local function train(words)
    local N = math.ceil(words:size(1) / opt.batchsize)
    local cost = 0
    local y = torch.ones(1)
    local input = transfer_data(torch.Tensor(opt.batchsize, opt.edim))
    local target = transfer_data(torch.Tensor(opt.batchsize))
    local context = transfer_data(torch.Tensor(opt.batchsize, opt.memsize))
    local time = transfer_data(torch.Tensor(opt.batchsize, opt.memsize))

    input:fill(opt.init_hid)
    for t = 1, opt.memsize do
        time:select(2, t):fill(t)
    end
    for n = 1, N do
        if opt.show then xlua.progress(n, N) end
        for b = 1, opt.batchsize do
            local m = math.random(opt.memsize + 1, words:size(1)-1)
            target[b] = words[m+1]
            context[b]:copy(
                words:narrow(1, m - opt.memsize + 1, opt.memsize))
        end
        local x = {input, target, context, time}
        local out = g_model:forward(x)
        cost = cost + out[1]
        g_paramdx:zero()
        g_model:backward(x, y)
        local gn = g_paramdx:norm()
        if gn > opt.maxgradnorm then
            g_paramdx:mul(opt.maxgradnorm / gn)
        end
        g_paramx:add(g_paramdx:mul(-opt.dt))
    end
    return cost/N/opt.batchsize
end

local function test(words)
    local N = math.ceil(words:size(1) / opt.batchsize)
    local cost = 0
    local input = transfer_data(torch.Tensor(opt.batchsize, opt.edim))
    local target = transfer_data(torch.Tensor(opt.batchsize))
    local context = transfer_data(torch.Tensor(opt.batchsize, opt.memsize))
    local time = transfer_data(torch.Tensor(opt.batchsize, opt.memsize))
    input:fill(opt.init_hid)
    for t = 1, opt.memsize do
        time:select(2, t):fill(t)
    end
    local m = opt.memsize + 1
    for n = 1, N do
        if opt.show then xlua.progress(n, N) end
        for b = 1, opt.batchsize do
            target[b] = words[m+1]
            context[b]:copy(
                words:narrow(1, m - opt.memsize + 1, opt.memsize))
            m = m + 1
            if m > words:size(1)-1 then
                m = opt.memsize + 1
            end
        end
        local x = {input, target, context, time}
        local out = g_model:forward(x)
        cost = cost + out[1]
    end
    return cost/N/opt.batchsize
end

local function run(epochs)
    for i = 1, epochs do
        local c, ct
        c = train(g_words_train)
        ct = test(g_words_valid)

        -- Logging
        local m = #g_log_cost+1
        g_log_cost[m] = {m, c, ct}
        g_log_perp[m] = {m, math.exp(c), math.exp(ct)}
        local stat = {perplexity = math.exp(c) , epoch = m,
                valid_perplexity = math.exp(ct), LR = opt.dt}
        if opt.test then
            local ctt = test(g_words_test)
            table.insert(g_log_cost[m], ctt)
            table.insert(g_log_perp[m], math.exp(ctt))
            stat['test_perplexity'] = math.exp(ctt)
        end
        print(stat)

        -- Learning rate annealing
        if m > 1 and g_log_cost[m][3] > g_log_cost[m-1][3] * 0.9999 then
            opt.dt = opt.dt / 1.5
            if opt.dt < 1e-5 then break end
        end
    end
end

local function save(path)
    local d = {}
    d.params = opt
    d.paramx = g_paramx:float()
    d.log_cost = g_log_cost
    d.log_perp = g_log_perp
    torch.save(path, d)
end


g_vocab =  tds.hash()
g_ivocab =  tds.hash()
g_ivocab[#g_vocab+1] = '<eos>'
g_vocab['<eos>'] = #g_vocab+1

g_words_train = g_read_words('data/train.txt', g_vocab, g_ivocab)
g_words_valid = g_read_words('data/valid.txt', g_vocab, g_ivocab)
g_words_test = g_read_words('data/test.txt', g_vocab, g_ivocab)
opt.nwords = #g_vocab
print('vocabulary size ' .. #g_vocab)

g_model = g_build_model(opt)
g_paramx, g_paramdx = g_model:getParameters()
g_paramx:normal(0, opt.init_std)
if opt.load ~= '' then
    local f = torch.load(opt.load)
    g_paramx:copy(f.paramx)
end

g_log_cost = {}
g_log_perp = {}
opt.dt = opt.sdt

print('starting to run....')
run(opt.epochs)

if opt.save ~= '' then
    save(opt.save)
end
