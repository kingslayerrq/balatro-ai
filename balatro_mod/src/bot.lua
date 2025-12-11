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
    action_lock = false
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
        Bot.last_state_enum = G.STATE
        Bot.action_lock = false
        Bot.processing_hand = false
        Bot.waiting_for_action = false
        Bot.auto_timer = 0
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
        -- NOTE: NEW_ROUND and ROUND_EVAL removed - they occur at wrong times
        -- and cause premature phase transitions. Let game handle these automatically.
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
                Bot.log("Action: " .. (action.type or "unknown"))
                Bot.execute(action)
                Bot.waiting_for_action = false
            end
            
            -- B. Send state if idle
            if not Bot.waiting_for_action then
                local state = Bot.collect_full_state() -- Uses the rich data collector
                if state then
                    if state.chips ~= state.chips then state.chips = 0 end -- NaN fix
                    
                    Bot.client:send(json.encode(state) .. "\n")
                    Bot.waiting_for_action = true
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
    
    -- GAMEPLAY ACTIONS
    if action.type == "toggle" then
        local idx = action.index + 1
        if G.hand and G.hand.cards and G.hand.cards[idx] then
            G.hand.cards[idx]:click()
        end
    elseif action.type == "play" then
        if G.hand and G.hand.highlighted and #G.hand.highlighted > 0 then
            G.FUNCS.play_cards_from_highlighted()
        end
    elseif action.type == "discard" then
        G.FUNCS.discard_cards_from_highlighted()

    -- SHOP ACTIONS
    elseif action.type == "buy" then
        local slot = action.slot + 1
        if G.shop_jokers and G.shop_jokers.cards and G.shop_jokers.cards[slot] then
            G.FUNCS.buy_from_shop({config = {ref_table = G.shop_jokers.cards[slot]}})
        elseif G.shop_booster and G.shop_booster.cards and G.shop_booster.cards[slot] then
            G.FUNCS.buy_from_shop({config = {ref_table = G.shop_booster.cards[slot]}})
        elseif G.shop_vouchers and G.shop_vouchers.cards and G.shop_vouchers.cards[slot] then
             G.FUNCS.redeem_from_shop({config = {ref_table = G.shop_vouchers.cards[slot]}})
        end

    elseif action.type == "sell_joker" then
        local idx = action.index + 1
        if G.jokers and G.jokers.cards and G.jokers.cards[idx] then
            G.FUNCS.sell_card({config = {ref_table = G.jokers.cards[idx]}})
        end

    elseif action.type == "reroll" then
        G.FUNCS.reroll_shop()

    elseif action.type == "next_round" then
        if G.shop then 
            local btn = Bot.find_button_by_id(G.shop, 'next_round_button')
            if btn then 
                if btn.config.button == 'toggle_shop' then G.FUNCS.toggle_shop(btn)
                else G.FUNCS.next_round(btn) end
            else
                -- Fallback if button search fails (rare)
                G.FUNCS.toggle_shop() 
            end
        end

    -- BLIND ACTIONS
    elseif action.type == "select_blind" then
        local target = action.blind -- "small", "big", "boss"
        local key_map = {small = 'Small', big = 'Big', boss = 'Boss'}
        local target_key = key_map[target]

        Bot.log("Selecting blind: " .. target_key)

        -- Try 1: Use the game's blind_select_opts if available
        -- This is the most reliable method as it uses the actual UI buttons
        if G.blind_select_opts and G.blind_select_opts[target] then
            local btn = G.blind_select_opts[target]

            -- [CRITICAL FIX] Ensure UIBox is properly set before calling
            if not btn.UIBox then
                btn.UIBox = G.blind_select
            end

            local ok = pcall(function() G.FUNCS.select_blind(btn) end)
            if ok then
                Bot.log("Selected via blind_select_opts")
                return
            else
                Bot.log("blind_select_opts failed, trying fallback")
            end
        end

        -- Try 2: Keyboard input (simpler and less error-prone)
        if G.CONTROLLER and G.CONTROLLER.key_press then
            G.CONTROLLER:key_press('return')
            Bot.log("Selected via keyboard input")
            return
        end

        Bot.log("WARNING: All blind selection methods failed for " .. target_key)
    

    elseif action.type == "skip_blind" then
        -- We must find the REAL skip button to click it safely
        local target = string.lower(G.GAME.blind_on_deck or '')

        Bot.log("Attempting to skip blind: " .. target)

        -- G.blind_select_opts contains the UI nodes for 'small', 'big', 'boss'
        local root = G.blind_select_opts and G.blind_select_opts[target]

        if root then
            -- [FIX] Search both children AND nodes arrays (like find_button_by_id)
            local function find_skip_btn(node)
                if not node then return nil end
                if node.config and node.config.button == 'skip_blind' then return node end

                -- Search children
                if node.children then
                    for _, child in ipairs(node.children) do
                        local found = find_skip_btn(child)
                        if found then return found end
                    end
                end

                -- [FIX] Also search nodes array!
                if node.nodes then
                    for _, subnode in ipairs(node.nodes) do
                        local found = find_skip_btn(subnode)
                        if found then return found end
                    end
                end

                return nil
            end

            local btn = find_skip_btn(root)
            if btn then
                Bot.log("Found skip button, clicking for " .. target)
                -- Ensure UIBox context exists
                if not btn.UIBox then btn.UIBox = G.blind_select end
                G.FUNCS.skip_blind(btn)
            else
                Bot.log("Skip button not found for " .. target .. " (tag may not be available)")
            end
        else
            Bot.log("No blind_select_opts found for target: " .. target)
        end

    -- BLIND COMPLETE / CASH OUT
    elseif action.type == "cash_out" then
        Bot.log("Clicking cash out button")

        -- Try to find and click the cash out / continue button
        -- Method 1: Try the eval_quit button (used after completing a blind)
        if G.E_MANAGER and G.E_MANAGER.queued_messages and #G.E_MANAGER.queued_messages == 0 then
            -- No animations queued, safe to continue
            if G.FUNCS.cash_out then
                G.FUNCS.cash_out()
                Bot.log("Used G.FUNCS.cash_out()")
                return
            end
        end

        -- Method 2: Look for continue button in UI
        if G.buttons then
            local continue_btn = Bot.find_button_by_id(G.buttons, 'eval_quit') or
                                 Bot.find_button_by_id(G.buttons, 'cash_out')
            if continue_btn and continue_btn.config and continue_btn.config.button then
                G.FUNCS[continue_btn.config.button](continue_btn)
                Bot.log("Clicked continue button via UI search")
                return
            end
        end

        -- Method 3: Keyboard shortcut (Enter usually works)
        if G.CONTROLLER and G.CONTROLLER.key_press then
            G.CONTROLLER:key_press('return')
            Bot.log("Used keyboard shortcut for cash out")
            return
        end

        Bot.log("WARNING: Could not find cash out button")

    -- PACK ACTIONS
    elseif action.type == "pack_select" then
        local idx = action.index + 1
        if G.pack_cards and G.pack_cards.cards and G.pack_cards.cards[idx] then
            G.FUNCS.use_card({config = {ref_table = G.pack_cards.cards[idx]}})
        end
    elseif action.type == "pack_skip" then
        G.FUNCS.skip_booster()
    end
end

-- =============================================================================
-- RICH STATE COLLECTION (Adapted from lua_state_reference.lua)
-- =============================================================================

function Bot.collect_full_state()
    if not G.GAME then return nil end

    local current_phase = Bot.get_current_phase()
    local chips = G.GAME.chips or 0
    local target = (G.GAME.blind and G.GAME.blind.chips) or 0
    local hands = G.GAME.current_round.hands_left or 0
    Bot.log(string.format("State: G.STATE=%s, phase=%s, chips=%d/%d, hands=%d",
        tostring(G.STATE), current_phase, chips, target, hands))

    -- [FIX] Don't send state if blind target is 0 (not initialized yet)
    -- This prevents the AI from acting during the brief window when the blind
    -- hasn't been properly set up, which was causing premature blind completion
    if current_phase == "blind" and target == 0 then
        Bot.log("WARNING: Blind not initialized yet (target=0), waiting...")
        return nil
    end

    local state = {
        -- Global
        phase = current_phase,
        ante = G.GAME.round_resets.ante or 1,
        round = G.GAME.round or 1,
        money = G.GAME.dollars or 0,
        chips = G.GAME.chips or 0,
        blind_target = G.GAME.blind and G.GAME.blind.chips or 1,
        hands_left = G.GAME.current_round.hands_left or 0,
        discards_left = G.GAME.current_round.discards_left or 0,
        is_boss = G.GAME.blind and G.GAME.blind.boss,
        
        -- Collections
        deck_stats = Bot.get_deck_stats(),
        hand = Bot.get_hand_cards(),
        jokers = Bot.get_jokers(),
        consumables = Bot.get_consumables(),
        shop = Bot.get_shop_items(),
        
        -- Blind Info
        blind_select = Bot.get_blind_select_info(),
        blinds_available = Bot.get_available_blinds(),
        can_skip = Bot.can_skip_blind(),
        
        -- Pack Info
        pack_cards = Bot.get_pack_cards(),
        pack_picks_remaining = G.pack_cards and G.pack_cards.config and G.pack_cards.config.choose or 0,
        
        -- Flags
        game_over = G.STATE == G.STATES.GAME_OVER,
        game_won = G.GAME.won
    }
    return state
end

function Bot.get_current_phase()
    if G.STATE == G.STATES.SELECTING_HAND then return "blind" end
    if G.STATE == G.STATES.SHOP then return "shop" end
    if G.STATE == G.STATES.BLIND_SELECT then return "blind_select" end
    if G.STATE >= G.STATES.TAROT_PACK and G.STATE <= G.STATES.BUFFOON_PACK then return "pack_opening" end
    -- NOTE: NEW_ROUND/ROUND_EVAL removed - they transition automatically
    -- The game handles blind completion internally
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

function Bot.get_blind_select_info()
    return {
        small_target = G.GAME.round_resets.blind_choices.Small and G.P_BLINDS[G.GAME.round_resets.blind_choices.Small].chips or 0,
        big_target = G.GAME.round_resets.blind_choices.Big and G.P_BLINDS[G.GAME.round_resets.blind_choices.Big].chips or 0,
        boss_target = G.GAME.round_resets.blind_choices.Boss and G.P_BLINDS[G.GAME.round_resets.blind_choices.Boss].chips or 0
    }
end

function Bot.get_available_blinds()
    local avail = {}
    if G.GAME.round_resets.blind_states.Small ~= 'Defeated' and G.GAME.round_resets.blind_states.Small ~= 'Skipped' then table.insert(avail, "small") end
    if G.GAME.round_resets.blind_states.Big ~= 'Defeated' and G.GAME.round_resets.blind_states.Big ~= 'Skipped' then table.insert(avail, "big") end
    table.insert(avail, "boss")
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
    if G.pack_cards and G.pack_cards.cards then
        for _, card in ipairs(G.pack_cards.cards) do
            table.insert(cards, {name = card.ability.name})
        end
    end
    return cards
end

return Bot