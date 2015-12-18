-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

require('paths')

local stringx = require('pl.stringx')
local file = require('pl.file')
local utf8 = require 'lua-utf8'

function g_read_words(fname, vocab, ivocab)
    local data = file.read(fname)
    local lines = stringx.splitlines(data)
    --local words = torch.Tensor(c, 1)
		local words = {}
    for _, line in ipairs(lines) do
			for _, char_code in utf8.next, line do
				local char = utf8.char(char_code)
				if not vocab[char] then
						ivocab[#vocab+1] = char
						vocab[char] = #vocab+1
				end
				table.insert(words,vocab[char])
			end
    end
    print('Read ' .. #words .. ' words from ' .. fname)
		words = torch.Tensor{words}
		words:reshape(words:size(1),1)
    return words
end
