local socket = require("socket")
local json = require("Mods.BalatroBot.src.json")

-- [CONFIG]
local LOG_FILE_PATH = "Mods/BalatroBot/lua_debug.log"
local ENABLE_FILE_LOGGING = true 

local Bot = {
    client = nil,
    connected = false,
    retry_timer = 0,
    waiting_for_action = false,
    processing_hand = false,
    auto_timer = 0,
    action_lock = false,
    receive_attempts = 0  -- Debug counter
}

function Bot.log(msg)
    print("[BOT] " .. msg)
    if ENABLE_FILE_LOGGING then
        local f = io.open(LOG_FILE_PATH, "a")
        if f then f:write(os.date("%X") .. ": " .. msg .. "\n"); f:close() end
    end
end

-- [SAFE SEARCH] Find button by ID
function Bot.find_button_by_id(root, target_id)
    if not root then return nil end
    if root.config and root.config.id == target_id then return root end
    if root.children then
        for _, child in ipairs(root.children) do
            local found = Bot.find_button_by_id(child, target_id)
            if found then return found end
        end
    end
    if root.UIRoot then return Bot.find_button_by_id(root.UIRoot, target_id) end
    if root.nodes then 
        for _, node in ipairs(root.nodes) do
            local found = Bot.find_button_by_id(node, target_id)
            if found then return found end
        end
    end
    return nil
end

-- [SAFE MOCK] Only mocks UI functions
function Bot.create_safe_uibox_mock()
    local mock_node = {
        config = { object = { pop_out = function() end, pop_delay = 0 } }
    }
    
    local mock_uibox = {
        get_UIE_by_ID = function(self, id) return mock_node end,
        recalculate = function() end
    }
    return mock_uibox
end

function Bot.connect()
    Bot.client = socket.tcp()
    Bot.client:setoption("tcp-nodelay", true)
    Bot.client:settimeout(0.01)
    
    local success, err = Bot.client:connect("127.0.0.1", 5000)
    if success then
        Bot.connected = true
        Bot.client:settimeout(0)
        
        -- [CRITICAL FIX] Reset waiting flag on new connection
        -- This breaks the deadlock if Python restarted
        Bot.waiting_for_action = false 
        
        Bot.log("CONNECTED TO PYTHON")
    else
        Bot.connected = false
    end
end

function Bot.update(dt)
    if not Bot.connected then
        Bot.retry_timer = Bot.retry_timer + dt
        if Bot.retry_timer > 2 then Bot.connect(); Bot.retry_timer = 0 end
        return
    end

    -- [REMOVED HEARTBEAT CHECK - it was eating messages!]
    -- Connection errors will be detected in the actual receive calls below

    if not Bot.auto_timer then Bot.auto_timer = 0 end
    Bot.auto_timer = Bot.auto_timer + dt

    -- [CRITICAL FIX] Reset flags when state changes
    if Bot.last_state_enum ~= G.STATE then
        Bot.last_state_enum = G.STATE
        Bot.action_lock = false
        Bot.processing_hand = false
        Bot.waiting_for_action = false -- Ensure we send the new state immediately
        Bot.auto_timer = 0 
    end

    -- A. GAME OVER
    if G.STATE == G.STATES.GAME_OVER then
        if not Bot.action_lock and Bot.auto_timer > 1.5 then
            Bot.log("Restarting...")
            Bot.action_lock = true
            G.FUNCS.start_run(nil, { saved_run = nil, challenge = nil })
        end
        return
    end

    -- B. BLIND SELECT
    if G.STATE == G.STATES.BLIND_SELECT then
        if not Bot.action_lock and Bot.auto_timer > 1.0 then
            local target_key = 'Small'
            if G.GAME.round_resets.blind_states.Big == 'Select' then target_key = 'Big' end
            if G.GAME.round_resets.blind_states.Boss == 'Select' then target_key = 'Boss' end

            -- 1. Try finding REAL button first
            if G.blind_select then
                local btn = Bot.find_button_by_id(G.blind_select, 'select_blind_button')
                if btn and btn.config and btn.config.ref_table then
                    local blind_key = G.GAME.round_resets.blind_choices[target_key]
                    Bot.log("Clicking Real Blind Button")
                    Bot.action_lock = true
                    if not btn.UIBox then btn.UIBox = G.blind_select end
                    G.FUNCS.select_blind(btn)
                    Bot.auto_timer = 0
                    return
                end
            end

            -- 2. Fallback: Safe Mock Event
            if Bot.auto_timer > 3.0 then
                Bot.log("Button missing. Using Safe Mock for: " .. target_key)
                Bot.action_lock = true
                
                local blind_key = G.GAME.round_resets.blind_choices[target_key]
                local blind_data = G.P_BLINDS[blind_key]
                
                if not blind_data then return end

                local fake_event = {
                    config = {
                        ref_table = blind_data,
                        id = 'select_blind_button'
                    },
                    UIBox = Bot.create_safe_uibox_mock()
                }
                
                local ok, err = pcall(function() G.FUNCS.select_blind(fake_event) end)
                if not ok then Bot.log("Select Blind Error: " .. tostring(err)) end
                
                Bot.auto_timer = 0
            end
        end
        return
    end

    -- C. SHOP
    if G.STATE == G.STATES.SHOP then
        if not Bot.action_lock and Bot.auto_timer > 1.0 then
            local btn = nil
            if G.shop then btn = Bot.find_button_by_id(G.shop, 'next_round_button') end

            if btn then
                Bot.log("Clicking Next Round...")
                Bot.action_lock = true
                if not btn.UIBox then btn.UIBox = G.shop end
                if btn.config.button == 'toggle_shop' then G.FUNCS.toggle_shop(btn)
                else G.FUNCS.next_round(btn) end
                Bot.auto_timer = 0
            else
                if Bot.auto_timer > 3.0 then
                    Bot.log("Forcing Shop Skip...")
                    Bot.action_lock = true
                    local fake_event = { 
                        config = { id = 'next_round_button' }, 
                        UIBox = Bot.create_safe_uibox_mock() 
                    }
                    pcall(function() G.FUNCS.next_round(fake_event) end)
                    Bot.auto_timer = 0
                end
            end
        end
        return
    end

    -- D. GAMEPLAY
    if G.STATE == G.STATES.SELECTING_HAND then
        local status, err = pcall(function()
            -- Try multiple times per frame to catch messages
            -- With non-blocking sockets, we might miss messages if we only check once
            local max_attempts = 10
            for i = 1, max_attempts do
                Bot.client:settimeout(0)
                local response, receive_err = Bot.client:receive("*l")

                Bot.receive_attempts = Bot.receive_attempts + 1

                -- Removed periodic logging to reduce spam
                -- (Was: Log every 60 attempts to show we're checking)

                if receive_err and receive_err ~= "timeout" and receive_err ~= "wantread" then
                    Bot.log("Receive error: " .. tostring(receive_err))
                    if receive_err == "closed" then
                        Bot.log("Connection closed during gameplay!")
                        Bot.connected = false
                        Bot.waiting_for_action = false
                        return
                    end
                end

                if response then
                    Bot.log("Raw response (attempt " .. i .. "): " .. response)
                    local action = json.decode(response)
                    Bot.log("Received action: " .. action.type)

                    -- Reset the waiting flag BEFORE executing (in case action changes state)
                    Bot.waiting_for_action = false

                    Bot.execute(action)

                    -- [CRITICAL FIX] Immediately send state back after executing action
                    local state = Bot.collect_state()
                    if state then
                        -- Detect NaN values (infinite chips bug)
                        if state.chips ~= state.chips then state.chips = 0 end

                        Bot.client:send(json.encode(state) .. "\n")
                        Bot.waiting_for_action = true
                        Bot.log("Sent state back (chips: " .. state.chips .. "), waiting for next action...")
                    else
                        Bot.log("ERROR: Could not collect state after action!")
                    end
                    break  -- Got a message, stop checking
                end
            end

            -- Send initial state if we're not waiting
            if not Bot.waiting_for_action then
                local state = Bot.collect_state()
                if state then
                    if state.chips ~= state.chips then state.chips = 0 end
                    Bot.client:send(json.encode(state) .. "\n")
                    Bot.waiting_for_action = true
                    Bot.receive_attempts = 0  -- Reset counter
                    Bot.log("Sent initial state (chips: " .. state.chips .. "), waiting for first action...")
                end
            end
        end)

        -- Log pcall errors
        if not status then
            Bot.log("GAMEPLAY ERROR: " .. tostring(err))
        end
    end
end

-- (Keep collect_state and execute the same)
function Bot.collect_state()
    if not G.GAME or not G.hand or not G.deck then return nil end
    local deck_stats = { Spades=0, Hearts=0, Clubs=0, Diamonds=0, Stone=0 }
    if G.deck.cards then
        for _, card in ipairs(G.deck.cards) do
            if card.base and card.base.suit and deck_stats[card.base.suit] then
                deck_stats[card.base.suit] = deck_stats[card.base.suit] + 1
            elseif card.base.name == "Stone Card" then
                deck_stats.Stone = deck_stats.Stone + 1
            end
        end
    end
    local poker_hand_text = "High Card"
    if G.hand.highlighted and #G.hand.highlighted > 0 then
        local text = G.FUNCS.get_poker_hand_info(G.hand.highlighted)
        poker_hand_text = text
    end
    local hand_cards = {}
    if G.hand.cards then
        for _, card in ipairs(G.hand.cards) do
            table.insert(hand_cards, {
                rank = card.base.value,
                suit = card.base.suit,
                selected = (card.highlighted == true)
            })
        end
    end
    return {
        money = G.GAME.dollars or 0,
        chips = G.GAME.chips or 0,
        blind_target = G.GAME.blind.chips or 1,
        hands_left = G.GAME.current_round.hands_left or 0,
        discards_left = G.GAME.current_round.discards_left or 0,
        ante = G.GAME.round_resets.ante or 1,
        is_boss = G.GAME.blind.boss or false,
        deck_stats = deck_stats,
        poker_hand = poker_hand_text,
        hand = hand_cards,
        game_over = (G.STATE == G.STATES.GAME_OVER)
    }
end

function Bot.execute(action)
    if not action then return end
    if action.type == "ping" then 
        Bot.waiting_for_action = false -- [FIX] Ping breaks the wait loop
        return 
    end 
    
    if action.type == "toggle" then
        local idx = action.index + 1
        if G.hand and G.hand.cards and G.hand.cards[idx] then
            G.hand.cards[idx]:click()
        end
    elseif action.type == "play" then
        if Bot.processing_hand then return end 
        if G.hand and G.hand.highlighted and #G.hand.highlighted > 0 then
            Bot.processing_hand = true
            G.FUNCS.play_cards_from_highlighted()
        end
    elseif action.type == "discard" then
        if G.hand and G.hand.highlighted and #G.hand.highlighted > 0 then
            G.FUNCS.discard_cards_from_highlighted()
        end
    end
end

return Bot