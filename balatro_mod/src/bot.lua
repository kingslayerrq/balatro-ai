local socket = require("socket")
local json = require("Mods.BalatroBot.src.json")

-- [CONFIG]
local LOG_FILE_PATH = "Mods/BalatroBot/lua_debug.log"
local ENABLE_FILE_LOGGING = false 

local Bot = {
    client = nil,
    connected = false,
    retry_timer = 0,
    waiting_for_action = false,
    processing_hand = false,
    auto_timer = 0,
    action_lock = false,
    blind_init_delay = 0  -- Frames to wait after blind selection for initialization
}

function Bot.log(msg)
    print("[BOT] " .. msg)
    if ENABLE_FILE_LOGGING then
        local f = io.open(LOG_FILE_PATH, "a")
        if f then f:write(os.date("%X") .. ": " .. msg .. "\n"); f:close() end
    end
end

function Bot.inspect_node_details(node, depth, max_depth)
    if not node or depth > max_depth then return "" end
    local indent = string.rep("  ", depth)
    local info = ""
    
    if type(node) == "table" then
        -- Log basic info
        info = info .. indent .. "Type: " .. type(node) .. "\n"
        
        -- Check for config
        if node.config then
            info = info .. indent .. "config found:\n"
            if node.config.button then
                info = info .. indent .. "  button: " .. tostring(node.config.button) .. "\n"
            end
            if node.config.id then
                info = info .. indent .. "  id: " .. tostring(node.config.id) .. "\n"
            end
            if node.config.ref_table then
                info = info .. indent .. "  has ref_table: true\n"
            end
            if node.config.blind then
                info = info .. indent .. "  has blind: " .. tostring(node.config.blind.key or "?") .. "\n"
            end
        end
        
        -- Check for important fields
        if node.children and #node.children > 0 then
            info = info .. indent .. "children: " .. #node.children .. " items\n"
        end
        if node.nodes and #node.nodes > 0 then
            info = info .. indent .. "nodes: " .. #node.nodes .. " items\n"
        end
    end
    
    return info
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

function Bot.connect()
    Bot.client = socket.tcp()
    Bot.client:setoption("tcp-nodelay", true)
    Bot.client:settimeout(0.01)
    
    local success, err = Bot.client:connect("127.0.0.1", 5000)
    if success then
        Bot.connected = true
        Bot.client:settimeout(0)
        Bot.waiting_for_action = false
        Bot.log("CONNECTED TO PYTHON")
    else
        Bot.connected = false
    end
end

-- =============================================================================
-- MAIN LOOP
-- =============================================================================

function Bot.update(dt)
    -- 1. Connection Handling
    if not Bot.connected then
        Bot.retry_timer = Bot.retry_timer + dt
        if Bot.retry_timer > 2 then Bot.connect(); Bot.retry_timer = 0 end
        return
    end

    -- [REMOVED HEARTBEAT CHECK - it eats messages from Python!]
    -- Connection errors will be detected in the actual receive calls below

    if not Bot.auto_timer then Bot.auto_timer = 0 end
    Bot.auto_timer = Bot.auto_timer + dt

    -- 2. State Change Detection
    if Bot.last_state_enum ~= G.STATE then
        Bot.log("STATE CHANGE: " .. tostring(Bot.last_state_enum) .. " -> " .. tostring(G.STATE))

        -- Extra debugging when entering ROUND_EVAL (state 8)
        if G.STATE == G.STATES.ROUND_EVAL then
            Bot.log("=== ENTERED ROUND_EVAL STATE ===")
            if G.GAME then
                local chips = G.GAME.chips or 0
                local target = (G.GAME.blind and G.GAME.blind.chips) or 0
                Bot.log("  Chips: " .. chips .. "/" .. target)
                Bot.log("  Won: " .. tostring(chips >= target))
                Bot.log("  blind_won flag: " .. tostring(G.GAME.blind_won))

                if G.GAME.round_resets then
                    for key, value in pairs(G.GAME.round_resets) do
                        if type(value) == "boolean" or type(value) == "number" or type(value) == "string" then
                            Bot.log("  round_resets." .. key .. " = " .. tostring(value))
                        end
                    end
                end
            end
        end

        -- Extra debugging when entering mega pack state (999)
        if G.STATE == 999 then
            Bot.log("=== ENTERED MEGA PACK STATE (999) ===")

            -- Search for ALL G tables that might contain pack cards
            local found_cards = false
            for key, value in pairs(G) do
                if type(value) == "table" and value.cards and type(value.cards) == "table" and #value.cards > 0 then
                    -- Found a table with cards!
                    local lower_key = string.lower(key)
                    if string.find(lower_key, "pack") or string.find(lower_key, "booster") or
                       string.find(lower_key, "card") or string.find(lower_key, "hand") then
                        Bot.log("  Found cards in G." .. key .. ": " .. #value.cards .. " cards")
                        found_cards = true

                        -- Check if it has config.choose (picks remaining)
                        if value.config and value.config.choose then
                            Bot.log("    config.choose: " .. tostring(value.config.choose))
                        end
                    end
                end
            end

            if not found_cards then
                Bot.log("  No pack cards found yet - will check again later")
            end
        end

        Bot.last_state_enum = G.STATE
        Bot.action_lock = false
        Bot.processing_hand = false
        Bot.waiting_for_action = false
        Bot.auto_timer = 0
        Bot.pack_skip_attempted = false  -- Reset pack skip flag on state change

        -- Reset state-specific flags
        if G.STATE ~= G.STATES.ROUND_EVAL then
            Bot.round_eval_timer = nil
            Bot.cash_out_waiting = false
            Bot.cash_out_ui_waiting = false
            Bot.cash_out_attempted = false
        end
        if G.STATE ~= G.STATES.SHOP then
            Bot.next_round_waiting = false
            Bot.next_round_attempted = false
        end
        if G.STATE ~= 999 then
            Bot.mega_pack_skip_warned = false
        end
    end

    -- 3. Game Over Auto-Restart (Still handled by Lua for speed)
    if G.STATE == G.STATES.GAME_OVER then
        if not Bot.action_lock and Bot.auto_timer > 2.0 then
            Bot.log("Game Over. Restarting...")
            Bot.action_lock = true
            G.FUNCS.start_run(nil, { saved_run = nil, challenge = nil })
        end
        return
    end

    -- 4. CONTROL LOOP (Applies to Gameplay, Shop, Blind Select, Packs, Blind Complete)
    -- We only run this if we are in a stable state where inputs are allowed

    local active_states = {
        [G.STATES.SELECTING_HAND] = true,
        [G.STATES.SHOP] = true,
        [G.STATES.BLIND_SELECT] = true,
        [G.STATES.TAROT_PACK] = true,
        [G.STATES.PLANET_PACK] = true,
        [G.STATES.SPECTRAL_PACK] = true,
        [G.STATES.STANDARD_PACK] = true,
        [G.STATES.BUFFOON_PACK] = true,
        [G.STATES.ROUND_EVAL] = true,  -- State 8 - blind complete/cash out screen
        [999] = true,  -- State 999 - Mega packs (choose multiple cards, e.g., "Arcana Pack - Choose 2")
        -- NOTE: DRAW_TO_HAND (state 3) is NOT active - it's a transition/animation state
        -- Taking actions during card draw causes negative hand counts
    }

    if active_states[G.STATE] then
        local status, err = pcall(function()
            -- A. Check for incoming commands
            local response, receive_err = Bot.client:receive("*l")

            -- Check for connection errors
            if receive_err and receive_err ~= "timeout" and receive_err ~= "wantread" then
                if receive_err == "closed" then
                    Bot.log("Connection closed during gameplay")
                    Bot.connected = false
                    Bot.waiting_for_action = false
                    return
                else
                    Bot.log("Receive error: " .. tostring(receive_err))
                end
            end

            if response then
                local action = json.decode(response)

                -- Log actions with details (skip ping and toggle which happen frequently)
                local action_type = action.type or "unknown"
                if action_type ~= "ping" and action_type ~= "toggle" then
                    local details = ""
                    if action.index ~= nil then details = details .. " index=" .. action.index end
                    if action.slot ~= nil then details = details .. " slot=" .. action.slot end
                    Bot.log("Action: " .. action_type .. details)
                end

                Bot.execute(action)
                Bot.waiting_for_action = false
            end
            
            -- B. Send state if idle
            if not Bot.waiting_for_action then
                -- Handle blind initialization delay
                if Bot.blind_init_delay > 0 then
                    Bot.blind_init_delay = Bot.blind_init_delay - 1
                    if Bot.blind_init_delay == 0 then
                        Bot.log("Blind initialization complete, resuming AI control")
                    end
                else
                    -- CRITICAL: Wait for animations to complete before sending state
                    -- This prevents actions from executing during card flipping/scoring
                    local queued = 0
                    if G.E_MANAGER and G.E_MANAGER.queues and G.E_MANAGER.queues.base then
                        queued = #G.E_MANAGER.queues.base
                    end

                    -- If animations are running, wait for them (prevents timing issues)
                    if queued > 0 and Bot.processing_hand then
                        -- Still processing previous hand, don't send state yet
                        -- Timeout after 5 seconds to prevent permanent stuck
                        if Bot.auto_timer > 5.0 then
                            Bot.log("WARNING: Animation timeout - forcing state update")
                            Bot.processing_hand = false
                        else
                            return
                        end
                    end

                    -- If animations finished, reset processing flag
                    if queued == 0 then
                        Bot.processing_hand = false
                    end

                    local state = Bot.collect_full_state() -- Uses the rich data collector
                    if state then
                        if state.chips ~= state.chips then state.chips = 0 end -- NaN fix

                        Bot.client:send(json.encode(state) .. "\n")
                        Bot.waiting_for_action = true
                    end
                end
            end
        end)

        if not status then
            Bot.log("Error in Update Loop: " .. tostring(err))
            Bot.waiting_for_action = false -- Reset on error so we retry
        end
    end
end

-- =============================================================================
-- ACTION EXECUTION
-- =============================================================================

function Bot.execute(action)
    if not action then return end
    if action.type == "ping" then 
        Bot.waiting_for_action = false 
        return 
    end 
    -- This prevents instant wins when target is 0
    if (action.type == "play" or action.type == "discard") then
        local target = (G.GAME.blind and G.GAME.blind.chips) or 0
        if target == 0 then
            Bot.log("IGNORING " .. action.type .. ": Blind target is 0 (initializing)")
            return
        end
    end
    
    -- GAMEPLAY ACTIONS
    if action.type == "toggle" then
        local idx = action.index + 1
        if G.hand and G.hand.cards and G.hand.cards[idx] then
            G.hand.cards[idx]:click()
        end
    elseif action.type == "play" then
        if G.hand and G.hand.highlighted and #G.hand.highlighted > 0 then
            Bot.processing_hand = true  -- Block state updates until animations complete
            G.FUNCS.play_cards_from_highlighted()
        else
            Bot.log("Cannot play: no cards selected")
        end
    elseif action.type == "discard" then
        if G.hand and G.hand.highlighted and #G.hand.highlighted > 0 then
            Bot.processing_hand = true  -- Block state updates until animations complete
            G.FUNCS.discard_cards_from_highlighted()
        else
            Bot.log("Cannot discard: no cards selected")
        end

    -- SHOP ACTIONS
    elseif action.type == "buy" then
        local slot = action.slot + 1
        local money = G.GAME.dollars or 0
        local card = nil

        -- Find the card to buy
        if G.shop_jokers and G.shop_jokers.cards and G.shop_jokers.cards[slot] then
            card = G.shop_jokers.cards[slot]
        elseif G.shop_booster and G.shop_booster.cards and G.shop_booster.cards[slot] then
            card = G.shop_booster.cards[slot]
        elseif G.shop_vouchers and G.shop_vouchers.cards and G.shop_vouchers.cards[slot] then
            card = G.shop_vouchers.cards[slot]
        end

        if not card then
            Bot.log("Cannot buy: no card in slot " .. (slot-1))
            return
        end

        -- Check if we can afford it (use .cost for buying, NOT .sell_cost!)
        local cost = card.cost or 0
        if money < cost then
            Bot.log("Cannot buy: need $" .. cost .. " but only have $" .. money)
            return
        end

        Bot.log("Buying from slot " .. (slot-1) .. " for $" .. cost)

        -- Execute the buy
        if G.shop_vouchers and G.shop_vouchers.cards and G.shop_vouchers.cards[slot] then
            G.FUNCS.redeem_from_shop({config = {ref_table = card}})
        else
            G.FUNCS.buy_from_shop({config = {ref_table = card}})
        end

    elseif action.type == "sell_joker" then
        local idx = action.index + 1
        if G.jokers and G.jokers.cards and G.jokers.cards[idx] then
            Bot.log("Selling joker: " .. tostring(G.jokers.cards[idx].ability and G.jokers.cards[idx].ability.name or "unknown"))
            G.FUNCS.sell_card({config = {ref_table = G.jokers.cards[idx]}})
        else
            Bot.log("Cannot sell: no joker at index " .. (idx-1))
        end

    elseif action.type == "use_consumable" then
        local idx = action.index + 1
        if not G.consumeables or not G.consumeables.cards or not G.consumeables.cards[idx] then
            Bot.log("Cannot use consumable: no consumable at index " .. (idx-1))
            return
        end

        local card = G.consumeables.cards[idx]
        local card_name = card.ability and card.ability.name or "unknown"
        local card_type = card.ability and card.ability.set or "unknown"
        Bot.log("Using consumable: " .. card_name .. " (type: " .. card_type .. ")")

        local success, err = pcall(function()
            G.FUNCS.use_card({config = {ref_table = card}})
        end)

        if not success then
            Bot.log("  ERROR using consumable: " .. tostring(err))
        else
            Bot.log("  Successfully used " .. card_name)
        end

    elseif action.type == "reroll" then
        -- Check if we have enough money for reroll
        local reroll_cost = G.GAME.round_resets.reroll_cost or 5
        local money = G.GAME.dollars or 0

        if money < reroll_cost then
            Bot.log("Cannot reroll: need $" .. reroll_cost .. " but only have $" .. money)
            return
        end

        Bot.log("Rerolling shop for $" .. reroll_cost)
        G.FUNCS.reroll_shop()

    elseif action.type == "next_round" then
        -- Wait for shop to be fully initialized
        if not G.shop then
            Bot.log("Waiting for shop to initialize...")
            return
        end

        if Bot.next_round_attempted then
            return  -- Silently wait for state change
        end

        -- Wait for any animations to complete with 3s timeout
        local queued = 0
        if G.E_MANAGER and G.E_MANAGER.queues and G.E_MANAGER.queues.base then
            queued = #G.E_MANAGER.queues.base
        end

        if queued > 0 and Bot.auto_timer < 3.0 then
            if not Bot.next_round_waiting then
                Bot.log("Waiting for " .. queued .. " animations before leaving shop (max 3s)...")
                Bot.next_round_waiting = true
            end
            return
        end

        Bot.next_round_attempted = true
        Bot.log("Attempting next_round...")

        local btn = Bot.find_button_by_id(G.shop, 'next_round_button')
        if btn then
            Bot.log("  Found button: " .. tostring(btn.config.button))
            if btn.config.button == 'toggle_shop' then
                G.FUNCS.toggle_shop(btn)
            else
                G.FUNCS.next_round(btn)
            end
        else
            Bot.log("  Button not found, using fallback toggle_shop()")
            G.FUNCS.toggle_shop()
        end

    
    -- BLIND ACTIONS
    elseif action.type == "select_blind" then
        if G.STATE ~= G.STATES.BLIND_SELECT then
             Bot.log("IGNORING select_blind: Invalid state " .. tostring(G.STATE))
             return
        end
        local target = action.blind -- "small", "big", "boss"
        local key_map = {small = 'Small', big = 'Big', boss = 'Boss'}
        local target_key = key_map[target]

        Bot.log("Selecting blind: " .. target_key)

        -- [FIX] Use the WORKING method from the old code
        -- Get the blind data from G.P_BLINDS (the source of truth)
        local blind_data_key = G.GAME.round_resets.blind_choices[target_key]
        local desired_blind_data = G.P_BLINDS[blind_data_key]

        if not desired_blind_data then
            Bot.log("ERROR: Could not find blind data for " .. target_key)
            return
        end

        Bot.log("Blind data key: " .. blind_data_key .. ", chips: " .. tostring(desired_blind_data.chips))

        -- Construct proper event object with blind data
        local e = {
            config = {
                button = 'select_blind',
                blind = desired_blind_data,
                ref_table = desired_blind_data
            },
            UIBox = G.blind_select
        }

        -- Call select_blind with the properly structured event
        local success, err = pcall(function()
            G.FUNCS.select_blind(e)
        end)

        if success then
            Bot.log("Successfully selected " .. target_key .. " blind!")
            return
        else
            Bot.log("ERROR: Failed to select blind: " .. tostring(err))
        end

        -- Old fallback code below (only executes if above fails)
        if G.blind_select_opts and G.blind_select_opts[target] then
            local btn = G.blind_select_opts[target]

            -- Debug: Dump button config in detail
            Bot.log("Button config for " .. target .. ":")
            if btn.config then
                for k, v in pairs(btn.config) do
                    local val_str = tostring(v)
                    if type(v) == "function" then
                        val_str = "function"
                    elseif type(v) == "table" then
                        val_str = "table"
                    end
                    Bot.log("  config." .. k .. " = " .. val_str)
                end
            else
                Bot.log("  config is nil!")
            end

            -- Check if there's a config.button field that tells us which function to call
            if btn.config and btn.config.button then
                Bot.log("Button function: " .. tostring(btn.config.button))
                local button_func = G.FUNCS[btn.config.button]
                if button_func then
                    local ok, err = pcall(function()
                        button_func(btn)
                    end)
                    if ok then
                        Bot.log("config.button function executed")
                        -- Check if it worked by seeing if state changed
                        if G.STATE ~= G.STATES.BLIND_SELECT then
                            Bot.log("State changed! Selection successful")
                            return
                        end
                        Bot.log("State didn't change, trying other methods...")
                    else
                        Bot.log("config.button function failed: " .. tostring(err))
                    end
                end
            else
                Bot.log("No config.button field found")
            end

            -- Try clicking the button
            if btn.click then
                local ok, err = pcall(function()
                    btn:click()
                end)
                if ok then
                    Bot.log("btn:click() executed (but may not have worked)")
                    -- Don't return - fall through to keyboard method as click often doesn't work
                else
                    Bot.log("btn:click() failed: " .. tostring(err))
                end
            end

            -- Fallback: Original method with UIBox
            if not btn.UIBox then
                btn.UIBox = G.blind_select
            end

            local ok, err = pcall(function() G.FUNCS.select_blind(btn) end)
            if ok then
                Bot.log("G.FUNCS.select_blind executed (but may not have worked)")
                -- Don't return - fall through to keyboard method as this often doesn't work either
            else
                Bot.log("G.FUNCS.select_blind failed: " .. tostring(err))
            end
        end

        -- If we get here, all button methods failed or didn't work
        -- Fall back to keyboard navigation which is most reliable

        -- Try 2: Keyboard navigation (most reliable - simulates user input)
        if G.CONTROLLER and G.CONTROLLER.key_press then
            -- Navigate to the correct blind using arrow keys
            -- Small = leftmost, Big = middle, Boss = rightmost
            -- Press left twice to ensure we're at the start, then navigate right as needed
            Bot.log("Using keyboard navigation to select " .. target)

            -- Reset to leftmost position (small blind)
            G.CONTROLLER:key_press('left')
            G.CONTROLLER:key_press('left')

            -- Navigate to target blind
            if target == 'big' then
                G.CONTROLLER:key_press('right')  -- Move from small to big
            elseif target == 'boss' then
                G.CONTROLLER:key_press('right')  -- Move from small to big
                G.CONTROLLER:key_press('right')  -- Move from big to boss
            end
            -- For 'small', we're already there after going left

            -- Select the blind
            G.CONTROLLER:key_press('return')
            Bot.log("Selected via keyboard navigation")
            return
        end

        Bot.log("WARNING: All blind selection methods failed for " .. target_key)
    

    elseif action.type == "skip_blind" then
        if G.STATE ~= G.STATES.BLIND_SELECT then
             Bot.log("IGNORING skip_blind: Invalid state " .. tostring(G.STATE))
             return
        end

        local target_key = G.GAME.blind_on_deck  -- "Small", "Big", or "Boss"
        Bot.log("Attempting to skip blind: " .. target_key)

        -- [FIX] Use the same working method as select_blind
        -- Get the blind data from G.P_BLINDS (the source of truth)
        local blind_data_key = G.GAME.round_resets.blind_choices[target_key]
        local desired_blind_data = G.P_BLINDS[blind_data_key]

        if not desired_blind_data then
            Bot.log("ERROR: Could not find blind data for " .. target_key)
            return
        end

        Bot.log("Skipping blind: " .. blind_data_key)

        -- Construct proper event object with blind data
        local e = {
            config = {
                button = 'skip_blind',
                blind = desired_blind_data,
                ref_table = desired_blind_data
            },
            UIBox = G.blind_select
        }

        -- Call skip_blind with the properly structured event
        local success, err = pcall(function()
            G.FUNCS.skip_blind(e)
        end)

        if success then
            Bot.log("Successfully skipped " .. target_key .. " blind!")
            return
        else
            Bot.log("ERROR: Failed to skip blind: " .. tostring(err))
        end

    -- BLIND COMPLETE / CASH OUT
    elseif action.type == "cash_out" then
        -- Only process cash_out in ROUND_EVAL state
        if G.STATE ~= G.STATES.ROUND_EVAL then
            return  -- Wrong state, ignore
        end

        -- If we already attempted cash_out, just wait for state change
        if Bot.cash_out_attempted then
            return  -- Silently wait
        end

        -- Check if animations are running
        local queued = 0
        if G.E_MANAGER and G.E_MANAGER.queues and G.E_MANAGER.queues.base then
            queued = #G.E_MANAGER.queues.base
        end

        -- Wait for animations, but with 3 second timeout
        -- (animations might be stuck waiting for user input)
        if queued > 0 and Bot.auto_timer < 3.0 then
            if not Bot.cash_out_waiting then
                Bot.log("Waiting for " .. queued .. " animations (max 3s)...")
                Bot.cash_out_waiting = true
            end
            return
        end

        -- If animations still queued after 3s, proceed anyway
        if queued > 0 then
            Bot.log("Timeout: " .. queued .. " animations still queued, proceeding...")
        end

        -- Only attempt once
        Bot.cash_out_attempted = true
        Bot.log("Attempting cash_out...")

        -- Debug: Log what state we're in
        Bot.log("  G.STATE=" .. tostring(G.STATE))
        Bot.log("  chips=" .. tostring(G.GAME.chips) .. "/" .. tostring(G.GAME.blind and G.GAME.blind.chips or 0))

        -- Try G.FUNCS.cash_out with constructed event
        if G.FUNCS.cash_out then
            local e = {
                config = { button = 'cash_out' }
            }
            local success, err = pcall(function()
                G.FUNCS.cash_out(e)
            end)
            if success then
                Bot.log("  cash_out() called successfully")
            else
                Bot.log("  cash_out() error: " .. tostring(err))
            end
        else
            Bot.log("  ERROR: G.FUNCS.cash_out not found!")
        end

        -- Log what functions exist for debugging (only once)
        Bot.log("Available FUNCS with 'cash/continue/eval':")
        for k, v in pairs(G.FUNCS) do
            if type(v) == "function" then
                local lower_k = string.lower(k)
                if string.find(lower_k, "cash") or string.find(lower_k, "continue") or string.find(lower_k, "eval") then
                    Bot.log("  G.FUNCS." .. k)
                end
            end
        end

    -- PACK ACTIONS
    elseif action.type == "pack_select" then
        local idx = action.index + 1
        Bot.log("Selecting pack card at index " .. tostring(idx-1))

        if not G.pack_cards then
            Bot.log("ERROR: G.pack_cards is nil")
            return
        end
        if not G.pack_cards.cards then
            Bot.log("ERROR: G.pack_cards.cards is nil")
            return
        end
        if not G.pack_cards.cards[idx] then
            Bot.log("ERROR: No card at index " .. idx .. " (total cards: " .. #G.pack_cards.cards .. ")")
            return
        end

        -- Check picks remaining (mega packs allow multiple selections)
        local picks_remaining = G.pack_cards.config and G.pack_cards.config.choose or 1
        if picks_remaining <= 0 then
            Bot.log("ERROR: No picks remaining (already selected maximum cards for this pack)")
            Bot.log("  You must skip the pack now or wait for it to auto-close")
            return
        end

        local card = G.pack_cards.cards[idx]
        local card_name = tostring(card.ability and card.ability.name or "unknown")
        Bot.log("Selecting card: " .. card_name .. " (picks remaining: " .. picks_remaining .. ")")

        local success, err = pcall(function()
            G.FUNCS.use_card({config = {ref_table = card}})
        end)

        if not success then
            Bot.log("ERROR selecting pack card: " .. tostring(err))
        else
            Bot.log("Successfully selected pack card")
        end

    elseif action.type == "pack_skip" then
        -- CRITICAL: Check if booster_pack AND its cards exist before trying to skip
        -- During cleanup or initialization, these might not exist yet
        if not G.booster_pack or not G.booster_pack.cards then
            -- Booster pack not ready yet or being destroyed, wait for it to load
            -- Don't log spam - this is normal during pack loading
            return
        end

        -- Check if cards array is empty (all cards selected, pack auto-closing)
        if #G.booster_pack.cards == 0 then
            -- Pack is empty and closing, wait for state change
            return
        end

        -- Prevent repeated skip attempts once we know pack is ready
        if Bot.pack_skip_attempted then
            return  -- Silently wait for state change
        end
        Bot.pack_skip_attempted = true

        Bot.log("Skipping pack (state=" .. tostring(G.STATE) .. ")...")

        -- Call skip_booster - works for all pack types including mega packs
        local success, err = pcall(function()
            G.FUNCS.skip_booster()
        end)

        if not success then
            Bot.log("  ERROR skipping pack: " .. tostring(err))
        else
            Bot.log("  Successfully skipped pack")
        end
    end
end

-- =============================================================================
-- RICH STATE COLLECTION (Adapted from lua_state_reference.lua)
-- =============================================================================

function Bot.collect_full_state()
    if not G.GAME then return nil end

    local current_phase = Bot.get_current_phase()
    
    -- [FIX] Try multiple sources for blind chip requirement
    local target = 0
    if G.GAME.blind then
        -- Try 1: Direct chips field
        target = G.GAME.blind.chips or 0

        -- Try 2: Config nested field
        if target == 0 and G.GAME.blind.config and G.GAME.blind.config.blind then
            target = G.GAME.blind.config.blind.chips or 0
            if target > 0 then
                Bot.log("Found blind target in config.blind.chips: " .. target)
            end
        end

        -- Try 3: chip_text field (might be a string)
        if target == 0 and G.GAME.blind.chip_text then
            local chip_num = tonumber(G.GAME.blind.chip_text)
            if chip_num and chip_num > 0 then
                target = chip_num
                Bot.log("Found blind target in chip_text: " .. target)
            end
        end

        -- Debug: If still 0, log blind state
        if target == 0 and current_phase == "blind" then
            Bot.log("blind_set = " .. tostring(G.GAME.blind.blind_set or "nil"))
            if G.GAME.blind.config and G.GAME.blind.config.blind then
                Bot.log("config.blind exists, checking fields...")
                for k, v in pairs(G.GAME.blind.config.blind) do
                    Bot.log("  config.blind." .. k .. " = " .. tostring(v))
                end
            end
        end
    else
        if current_phase == "blind" then
            Bot.log("WARNING: G.GAME.blind is nil in blind phase!")
        end
    end

    -- Try 4: Check round-level chip requirement
    if target == 0 and G.GAME.current_round then
        if G.GAME.current_round.chips_req then
            target = G.GAME.current_round.chips_req
            Bot.log("Found blind target in current_round.chips_req: " .. target)
        elseif G.GAME.current_round.chips_to_beat then
            target = G.GAME.current_round.chips_to_beat
            Bot.log("Found blind target in current_round.chips_to_beat: " .. target)
        end
    end

    -- Try 5: Use G.P_BLINDS with blind_on_deck (MOST RELIABLE)
    -- This is the source of truth for blind chip requirements
    if target == 0 and G.GAME.blind_on_deck and G.GAME.round_resets and G.GAME.round_resets.blind_choices then
        local blind_key = G.GAME.blind_on_deck  -- "Small", "Big", or "Boss"
        local blind_id = G.GAME.round_resets.blind_choices[blind_key]
        if blind_id and G.P_BLINDS and G.P_BLINDS[blind_id] then
            target = G.P_BLINDS[blind_id].chips or 0
            if target > 0 then
                Bot.log("Found blind target in G.P_BLINDS[" .. blind_id .. "].chips: " .. target)
            end
        end
    end
    
    local chips = G.GAME.chips or 0
    local hands = G.GAME.current_round.hands_left or 0

    -- Only log state when something meaningful changes (reduce spam)
    local state_key = string.format("%s|%s|%d|%d|%d", tostring(G.STATE), current_phase, chips, target, hands)
    if Bot.last_logged_state ~= state_key then
        Bot.log(string.format("State: G.STATE=%s, phase=%s, chips=%d/%d, hands=%d",
            tostring(G.STATE), current_phase, chips, target, hands))
        Bot.last_logged_state = state_key
    end

    -- [FIX] If we just entered blind phase with target=0, start the initialization delay
    if current_phase == "blind" and target == 0 and Bot.blind_init_delay == 0 then
        Bot.log("Blind phase entered with target=0, starting initialization delay")
        Bot.blind_init_delay = 60  -- Wait 60 frames (1 second) for blind to fully initialize
        return nil
    end

    -- [FIX] If blind still not ready after delay, keep waiting
    if current_phase == "blind" and target == 0 then
        Bot.log("WARNING: Blind not initialized yet (target=0), waiting...")
        return nil
    end

    -- Get game seed
    local seed = "unknown"
    if G.GAME and G.GAME.pseudorandom and G.GAME.pseudorandom.seed then
        seed = tostring(G.GAME.pseudorandom.seed)
    end

    local state = {
        phase = current_phase,
        ante = G.GAME.round_resets.ante or 1,
        round = G.GAME.round or 1,
        money = G.GAME.dollars or 0,
        reroll_cost = G.GAME.round_resets.reroll_cost or 5,
        chips = G.GAME.chips or 0,

        blind_target = target, -- Can be 0

        hands_left = G.GAME.current_round.hands_left or 0,
        discards_left = G.GAME.current_round.discards_left or 0,
        is_boss = G.GAME.blind and G.GAME.blind.boss,

        seed = seed,  -- Game seed for replay

        deck_stats = Bot.get_deck_stats(),
        hand = Bot.get_hand_cards(),
        hand_levels = Bot.get_hand_levels(),  -- Level/chips/mult for each hand type
        jokers = Bot.get_jokers(),
        consumables = Bot.get_consumables(),
        shop = Bot.get_shop_items(),
        blind_select = Bot.get_blind_select_info(),
        blinds_available = Bot.get_available_blinds(),
        can_skip = Bot.can_skip_blind(),
        pack_cards = Bot.get_pack_cards(),
        pack_picks_remaining = G.pack_cards and G.pack_cards.config and G.pack_cards.config.choose or 0,
        game_over = G.STATE == G.STATES.GAME_OVER,
        game_won = G.GAME.won
    }
    return state
end

function Bot.get_current_phase()
    if G.STATE == G.STATES.SELECTING_HAND then return "blind" end
    if G.STATE == G.STATES.SHOP then return "shop" end
    if G.STATE == G.STATES.BLIND_SELECT then return "blind_select" end

    -- ROUND_EVAL state - This is where the cash out button appears after beating/failing a blind
    if G.STATE == G.STATES.ROUND_EVAL then return "blind_complete" end

    -- DRAW_TO_HAND state - Transition state when drawing cards (not actionable)
    if G.STATE == G.STATES.DRAW_TO_HAND then return "drawing" end

    -- Mega pack state (choose multiple cards)
    if G.STATE == 999 then return "pack_opening" end

    -- Check all pack states explicitly
    if G.STATE == G.STATES.TAROT_PACK then return "pack_opening" end
    if G.STATE == G.STATES.PLANET_PACK then return "pack_opening" end
    if G.STATE == G.STATES.SPECTRAL_PACK then return "pack_opening" end
    if G.STATE == G.STATES.STANDARD_PACK then return "pack_opening" end
    if G.STATE == G.STATES.BUFFOON_PACK then return "pack_opening" end

    -- Fallback: check if state is in pack range
    if G.STATES.TAROT_PACK and G.STATES.BUFFOON_PACK then
        if G.STATE >= G.STATES.TAROT_PACK and G.STATE <= G.STATES.BUFFOON_PACK then
            return "pack_opening"
        end
    end

    -- Log unknown states for debugging
    Bot.log("UNKNOWN STATE: " .. tostring(G.STATE))
    if G.STATES.TAROT_PACK then
        Bot.log("  TAROT_PACK=" .. tostring(G.STATES.TAROT_PACK) ..
                ", PLANET_PACK=" .. tostring(G.STATES.PLANET_PACK) ..
                ", SPECTRAL_PACK=" .. tostring(G.STATES.SPECTRAL_PACK) ..
                ", STANDARD_PACK=" .. tostring(G.STATES.STANDARD_PACK) ..
                ", BUFFOON_PACK=" .. tostring(G.STATES.BUFFOON_PACK))
    end

    return "unknown"
end

function Bot.get_hand_cards()
    local cards = {}
    if G.hand and G.hand.cards then
        for i, card in ipairs(G.hand.cards) do
            table.insert(cards, {
                rank = card.base.value,
                suit = card.base.suit,
                selected = card.highlighted or false,
                enhancement = card.ability and card.ability.name or "None",
                edition = (card.edition and (card.edition.foil and "Foil" or card.edition.holo and "Holographic" or card.edition.polychrome and "Polychrome" or "None")) or "None",
                debuffed = card.debuff
            })
        end
    end
    return cards
end

function Bot.get_jokers()
    local jokers = {}
    if G.jokers and G.jokers.cards then
        for i, card in ipairs(G.jokers.cards) do
            table.insert(jokers, {
                name = card.ability.name,
                sell_value = card.sell_cost,
                edition = (card.edition and (card.edition.foil and "Foil" or card.edition.holo and "Holographic" or card.edition.polychrome and "Polychrome")) or "None"
            })
        end
    end
    return jokers
end

function Bot.get_consumables()
    local cons = {}
    if G.consumeables and G.consumeables.cards then
        for i, card in ipairs(G.consumeables.cards) do
            table.insert(cons, {
                name = card.ability.name,
                type = card.ability.set
            })
        end
    end
    return cons
end

function Bot.get_shop_items()
    local items = {}
    if G.shop_jokers and G.shop_jokers.cards then
        for _, card in ipairs(G.shop_jokers.cards) do
            table.insert(items, {type = "joker", name = card.ability.name, cost = card.cost})
        end
    end
    if G.shop_booster and G.shop_booster.cards then
        for _, card in ipairs(G.shop_booster.cards) do
            table.insert(items, {type = "pack", name = card.ability.name, cost = card.cost})
        end
    end
    if G.shop_vouchers and G.shop_vouchers.cards then
        for _, card in ipairs(G.shop_vouchers.cards) do
            table.insert(items, {type = "voucher", name = card.ability.name, cost = card.cost})
        end
    end
    return items
end

function Bot.get_deck_stats()
    local stats = {Spades=0, Hearts=0, Clubs=0, Diamonds=0, Stone=0, total=0}
    if G.deck and G.deck.cards then
        stats.total = #G.deck.cards
        for _, card in ipairs(G.deck.cards) do
            if card.base.suit then stats[card.base.suit] = (stats[card.base.suit] or 0) + 1 end
            if card.ability.name == "Stone Card" then stats.Stone = stats.Stone + 1 end
        end
    end
    return stats
end

function Bot.get_hand_levels()
    -- Collect current level, chips, and mult for each hand type
    -- This helps the AI know which hands are leveled up and should be prioritized
    local hands = {}
    if G.GAME and G.GAME.hands then
        for hand_name, hand_data in pairs(G.GAME.hands) do
            hands[hand_name] = {
                level = hand_data.level or 1,
                chips = hand_data.chips or 0,
                mult = hand_data.mult or 1,
                played = hand_data.played or 0,  -- Times played this run
                visible = hand_data.visible or false  -- If discovered/visible
            }
        end
    end
    return hands
end

function Bot.get_blind_select_info()
    return {
        small_target = G.GAME.round_resets.blind_choices.Small and G.P_BLINDS[G.GAME.round_resets.blind_choices.Small].chips or 0,
        big_target = G.GAME.round_resets.blind_choices.Big and G.P_BLINDS[G.GAME.round_resets.blind_choices.Big].chips or 0,
        boss_target = G.GAME.round_resets.blind_choices.Boss and G.P_BLINDS[G.GAME.round_resets.blind_choices.Boss].chips or 0
    }
end

function Bot.get_available_blinds()
    local avail = {}
    local states = G.GAME.round_resets.blind_states

    -- Small blind: Available if not defeated/skipped
    local small_done = states.Small == 'Defeated' or states.Small == 'Skipped'
    if not small_done then
        table.insert(avail, "small")
    end

    -- Big blind: Available only if small is done AND big is not done
    local big_done = states.Big == 'Defeated' or states.Big == 'Skipped'
    if small_done and not big_done then
        table.insert(avail, "big")
    end

    -- Boss blind: Available only if both small and big are done
    local boss_done = states.Boss == 'Defeated' or states.Boss == 'Skipped'
    if small_done and big_done and not boss_done then
        table.insert(avail, "boss")
    end

    return avail
end

function Bot.can_skip_blind()
    -- Safety check
    if not G.GAME or not G.GAME.blind_on_deck then return false end
    
    
    if G.GAME.blind_on_deck == 'Small' or G.GAME.blind_on_deck == 'Big' then
        return true
    end
    
    return false
end

function Bot.get_pack_cards()
    local cards = {}

    -- For mega packs (state 999), G.pack_cards might not be populated immediately
    -- or might be in a different location
    if G.STATE == 999 then
        -- Check multiple possible locations for mega pack cards
        if G.booster_pack and G.booster_pack.cards then
            for _, card in ipairs(G.booster_pack.cards) do
                if card and card.ability then
                    table.insert(cards, {name = card.ability.name})
                end
            end
            return cards
        end

        -- Check if cards are in a UI element
        if G.booster_pack_ui and G.booster_pack_ui.cards then
            for _, card in ipairs(G.booster_pack_ui.cards) do
                if card and card.ability then
                    table.insert(cards, {name = card.ability.name})
                end
            end
            return cards
        end
    end

    -- Standard pack handling
    if G.pack_cards and G.pack_cards.cards then
        for _, card in ipairs(G.pack_cards.cards) do
            if card and card.ability then
                table.insert(cards, {name = card.ability.name})
            end
        end
    end

    return cards
end

return Bot