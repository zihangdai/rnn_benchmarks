--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
----
local ok,cunn = pcall(require, 'fbcunn')
if not ok then
    ok,cunn = pcall(require,'cunn')
    if ok then
        print("warning: fbcunn not found. Falling back to cunn") 
        LookupTable = nn.LookupTable
    else
        print("Could not find cunn or fbcunn. Either is required")
        os.exit()
    end
else
    deviceParams = cutorch.getDeviceProperties(1)
    cudaComputeCapability = deviceParams.major + deviceParams.minor/10
    LookupTable = nn.LookupTable
end
require('cudnn')
require('nngraph')
require('base')
local ptb = require('data')

-- Train 1 day and gives 82 perplexity.
--[[
local params = {batch_size=20,
                seq_length=35,
                layers=2,
                decay=1.15,
                rnn_size=1500,
                dropout=0.65,
                init_weight=0.04,
                lr=1,
                vocab_size=10000,
                max_epoch=14,
                max_max_epoch=55,
                max_grad_norm=10}
               ]]--

-- Trains 1h and gives test 115 perplexity.
--[[
local params = {batch_size=20,
                seq_length=40,
                layers=1,
                decay=2,
                rnn_size=650,
                dropout=0.5,
                init_weight=0.1,
                lr=1,
                vocab_size=10000,
                max_epoch=4,
                max_max_epoch=6,
                max_grad_norm=5}
--]]
--[[ 
local params = {batch_size=20,
                seq_length=50,
                layers=2,
                decay=2,
                rnn_size=650,
                dropout=0,
                init_weight=0.1,
                lr=1,
                vocab_size=10000,
                max_epoch=4,
                max_max_epoch=6,
                max_grad_norm=5}
--]]

local params = {batch_size=20,
                seq_length=20,
                layers=1,
                decay=2,
                rnn_size=200,
                dropout=0.5,
                init_weight=0.1,
                lr=1,
                vocab_size=10000,
                max_epoch=4,
                max_max_epoch=6,
                max_grad_norm=5}
local function transfer_data(x)
  return x:cuda()
end

local state_train, state_valid, state_test
local model = {}

local paramx, paramdx

local function set_lstm_init(lstm, hidden, cell)
  assert(lstm)
  if (not lstm.hiddenInput) or (not lstm.cellInput) then
    lstm.hiddenInput = transfer_data(torch.zeros(params.layers, params.batch_size, params.rnn_size))
    lstm.cellInput   = transfer_data(torch.zeros(params.layers, params.batch_size, params.rnn_size))
  end
  if hidden and cell then
    lstm.hiddenInput:copy(hidden)
    lstm.cellInput:copy(cell)
  else
    lstm.hiddenInput:zero()
    lstm.cellInput:zero()
  end
end

local function LSTM(input_size, hidden_size, num_layer, dropout)
  local lstm = cudnn.LSTM(input_size, hidden_size, num_layer)
  if dropout > 0 then
    lstm.dropout = dropout
    lstm:reset()
  end

  return lstm
end

local lstm = LSTM(params.rnn_size, params.rnn_size, params.layers, params.dropout)

local function create_network()
  local rnns = nn.Sequential()
  rnns:add(nn.LookupTable(params.vocab_size, params.rnn_size))
  rnns:add(nn.Dropout(params.dropout))
  rnns:add(lstm)
  rnns:add(nn.Dropout(params.dropout))
  rnns:add(nn.View(-1, params.rnn_size))
  rnns:add(nn.Linear(params.rnn_size, params.vocab_size))
  rnns:add(cudnn.LogSoftMax())

  return transfer_data(rnns)
end

model.rnns = create_network()
model.rnns:getParameters():uniform(-params.init_weight, params.init_weight)

local criterion = transfer_data(nn.ClassNLLCriterion())

local function reset_state(state)
  set_lstm_init(lstm)
  state.pos = 1
end

local function setup()
  print("Creating a RNN LSTM network.")
  paramx, paramdx = model.rnns:getParameters()
  model.norm_dw = 0
end

local function fp(state)

  if state.pos + params.seq_length + 1 > state.data:size(1) then
    reset_state(state)
  end

  local x = state.data[{{state.pos, state.pos+params.seq_length-1}}]
  local y = state.data[{{state.pos+1, state.pos+params.seq_length}}]
  
  model.pred = model.rnns:forward(x)
  model.err = criterion:forward(model.pred, y:view(-1))
  set_lstm_init(lstm, lstm.hiddenOutput, lstm.cellOutput)

  return model.err
end

local function bp(state)
  paramdx:zero()

  local x = state.data[{{state.pos, state.pos+params.seq_length-1}}]
  local y = state.data[{{state.pos+1, state.pos+params.seq_length}}]
  
  local d_pred = criterion:backward(model.pred, y:view(-1))
  model.rnns:backward(x, d_pred)
  cutorch.synchronize()

  state.pos = state.pos + params.seq_length
  model.norm_dw = paramdx:norm()
  if model.norm_dw > params.max_grad_norm then
    local shrink_factor = params.max_grad_norm / model.norm_dw
    paramdx:mul(shrink_factor)
  end
  paramx:add(paramdx:mul(-params.lr))
end

local function run_valid()
  reset_state(state_valid)
  g_disable_dropout(model.rnns)
  local len = (state_valid.data:size(1) - 1) / (params.seq_length)
  local perp = 0
  for i = 1, len do
    perp = perp + fp(state_valid)
  end
  print("Validation set perplexity : " .. g_f3(torch.exp(perp / len)))
  g_enable_dropout(model.rnns)
end

local function run_test()
  reset_state(state_test)
  g_disable_dropout(model.rnns)
  local perp = 0
  local len = state_test.data:size(1)
  g_replace_table(model.s[0], model.start_s)
  for i = 1, (len - 1) do
    local x = state_test.data[i]
    local y = state_test.data[i + 1]
    perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
    perp = perp + perp_tmp[1]
    g_replace_table(model.s[0], model.s[1])
  end
  print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
  g_enable_dropout(model.rnns)
end

local function main()
  g_init_gpu(arg)
  state_train = {data=transfer_data(ptb.traindataset(params.batch_size))}
  state_valid = {data=transfer_data(ptb.validdataset(params.batch_size))}
  state_test  = {data=transfer_data(ptb.testdataset(params.batch_size))}
  print("Network parameters:")
  print(params)
  local states = {state_train, state_valid, state_test}
  for _, state in pairs(states) do
    reset_state(state)
  end
  setup()
  local step = 0
  local epoch = 0
  local total_cases = 0
  local beginning_time = torch.tic()
  local start_time = torch.tic() 
  local nstart_time = torch.tic()

  print("Starting training.")
  local words_per_step = params.seq_length * params.batch_size
  local epoch_size = torch.floor(state_train.data:size(1) / params.seq_length)
  local data_len = state_train.data:size(1) * params.batch_size
  local perps
  local wpss
  local nepochs = 1

  while epoch < params.max_max_epoch do
    local perp = fp(state_train)
    if perps == nil then
      perps = torch.zeros(epoch_size):add(perp)
      wpss = torch.zeros(params.max_max_epoch)
    end

    perps[step % epoch_size + 1] = perp
    step = step + 1
    bp(state_train)
    total_cases = total_cases + params.seq_length * params.batch_size
    epoch = step / epoch_size

    if step % torch.round(epoch_size / 10) == 10 then
      local wps = torch.floor(total_cases / torch.toc(start_time))
      local since_beginning = g_d(torch.toc(beginning_time) / 60)

      print('epoch = ' .. g_f3(epoch) ..
            ', train perp. = ' .. g_f3(torch.exp(perps:mean())) ..
            ', wps = ' .. wps ..
            ', dw:norm() = ' .. g_f3(model.norm_dw) ..
            ', lr = ' ..  g_f3(params.lr) ..
            ', since beginning = ' .. since_beginning .. ' mins.')
    end

    if step % epoch_size == 0 then

      if step ~= 0 then
	      local wps = torch.floor(data_len / torch.toc(nstart_time))
	      wpss[nepochs] = wps
	      nepochs = nepochs + 1
      end

      run_valid()
      if epoch > params.max_epoch then
          params.lr = params.lr / params.decay
      end
      nstart_time = torch.tic()
    end
    if step % 33 == 0 then
      cutorch.synchronize()
      collectgarbage()
    end
  end
  -- run_test()
  print(" Ave wps = " .. wpss:mean())
  print(" Std wps = " .. wpss:std())

  print("Training is over.")
end

main()
