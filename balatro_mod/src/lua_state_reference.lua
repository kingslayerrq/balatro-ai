--[[
================================================================================
BALATRO RL - LUA STATE REFERENCE
================================================================================

This file documents the JSON state structure that the Lua mod should send
to the Python RL environment. The Python side expects these fields to build
observations and calculate rewards.

Copy the relevant functions to your Balatro mod and call send_state() after
every game state change.

================================================================================
]]--

-- TCP Connection Setup (you likely already have this)
local socket = require("socket")
local json = require("json") -- Or your JSON library

local client = nil
local SERVER_HOST = "127.0.0.1"
local SERVER_PORT = 5000

function connect_to_server()
    client = socket.tcp()
    client:settimeout(1)
    local success, err = client:connect(SERVER_HOST, SERVER_PORT)
    if not success then
        print("Failed to connect: " .. tostring(err))
        return false
    end
    client:setoption('tcp-nodelay', true)
    print("Connected to RL server")
    return true
end

--[[
================================================================================
MAIN STATE STRUCTURE
================================================================================

This is the complete JSON state the Python environment expects.
Not all fields are required for all phases, but including them helps
the agent learn better.

]]--

function build_game_state()
    local state = {
        -- ==================== GLOBAL STATE ====================
        -- Current game phase (CRITICAL for action masking)
        phase = get_current_phase(),  -- "blind", "shop", "blind_select", "pack_opening"
        
        -- Game progression
        ante = G.GAME.round_resets.ante or 1,
        round = G.GAME.round or 1,
        
        -- Resources
        money = G.GAME.dollars or 0,
        
        -- Blind info (during blind phase)
        chips = G.GAME.chips or 0,
        blind_target = get_blind_chips() or 300,
        hands_left = G.GAME.current_round.hands_left or 0,
        discards_left = G.GAME.current_round.discards_left or 0,
        is_boss = is_boss_blind(),
        
        -- Deck info
        deck_remaining = #G.deck.cards or 52,
        deck_stats = get_deck_stats(),
        
        -- Hand cards (during blind phase)
        hand = get_hand_cards(),
        
        -- Current poker hand evaluation
        poker_hand = get_current_poker_hand(),
        
        -- Hand levels (for each poker hand type)
        hand_levels = get_hand_levels(),
        
        -- Jokers
        jokers = get_jokers(),
        
        -- Consumables (tarots, planets, spectrals)
        consumables = get_consumables(),
        
        -- Shop items (during shop phase)
        shop = get_shop_items(),
        reroll_cost = G.GAME.current_round.reroll_cost or 5,
        
        -- Blind selection info (during blind_select phase)
        blind_select = get_blind_select_info(),
        can_skip = can_skip_blind(),
        blinds_available = get_available_blinds(),
        
        -- Pack opening info (during pack phase)
        pack_cards = get_pack_cards(),
        pack_picks_remaining = get_pack_picks_remaining(),
        
        -- Game state flags
        game_over = G.STATE == G.STATES.GAME_OVER,
        game_won = G.GAME.round_resets.ante >= 9,  -- Beat ante 8
    }
    
    return state
end

--[[
================================================================================
PHASE DETECTION
================================================================================
]]--

function get_current_phase()
    -- Adjust these to match your game's state names
    if G.STATE == G.STATES.SELECTING_HAND or G.STATE == G.STATES.HAND_PLAYED then
        return "blind"
    elseif G.STATE == G.STATES.SHOP then
        return "shop"
    elseif G.STATE == G.STATES.BLIND_SELECT then
        return "blind_select"
    elseif G.STATE == G.STATES.PLANET_PACK or G.STATE == G.STATES.TAROT_PACK 
           or G.STATE == G.STATES.SPECTRAL_PACK or G.STATE == G.STATES.STANDARD_PACK then
        return "pack_opening"
    else
        return "blind"  -- Default
    end
end

--[[
================================================================================
CARD DATA
================================================================================
]]--

function get_hand_cards()
    local cards = {}
    
    if not G.hand or not G.hand.cards then
        return cards
    end
    
    for i, card in ipairs(G.hand.cards) do
        table.insert(cards, {
            -- Basic card info
            rank = get_rank_name(card),  -- "2"-"10", "Jack", "Queen", "King", "Ace"
            suit = get_suit_name(card),  -- "Spades", "Hearts", "Clubs", "Diamonds"
            
            -- Selection state (CRITICAL for action masking)
            selected = card.highlighted or false,
            
            -- Enhancements
            enhancement = get_enhancement_name(card),  -- "None", "Bonus", "Mult", "Wild", etc.
            edition = get_edition_name(card),          -- "None", "Foil", "Holographic", "Polychrome"
            seal = get_seal_name(card),                -- "None", "Gold", "Red", "Blue", "Purple"
            
            -- Status
            scoring = card.ability and card.ability.scoring or false,  -- Will this card score?
            debuffed = card.debuff or false,  -- Is the card debuffed?
        })
    end
    
    return cards
end

function get_rank_name(card)
    local rank = card.base.value
    local rank_map = {
        ["2"] = "2", ["3"] = "3", ["4"] = "4", ["5"] = "5",
        ["6"] = "6", ["7"] = "7", ["8"] = "8", ["9"] = "9", ["10"] = "10",
        ["Jack"] = "Jack", ["Queen"] = "Queen", ["King"] = "King", ["Ace"] = "Ace"
    }
    return rank_map[rank] or "2"
end

function get_suit_name(card)
    local suit = card.base.suit
    local suit_map = {
        ["Spades"] = "Spades",
        ["Hearts"] = "Hearts", 
        ["Clubs"] = "Clubs",
        ["Diamonds"] = "Diamonds"
    }
    return suit_map[suit] or "Spades"
end

function get_enhancement_name(card)
    if not card.ability then return "None" end
    local enhancements = {
        "m_bonus", "m_mult", "m_wild", "m_glass", 
        "m_steel", "m_stone", "m_gold", "m_lucky"
    }
    local names = {
        ["m_bonus"] = "Bonus", ["m_mult"] = "Mult", ["m_wild"] = "Wild",
        ["m_glass"] = "Glass", ["m_steel"] = "Steel", ["m_stone"] = "Stone",
        ["m_gold"] = "Gold", ["m_lucky"] = "Lucky"
    }
    for _, e in ipairs(enhancements) do
        if card.ability[e] then return names[e] end
    end
    return "None"
end

function get_edition_name(card)
    if not card.edition then return "None" end
    if card.edition.foil then return "Foil" end
    if card.edition.holo then return "Holographic" end
    if card.edition.polychrome then return "Polychrome" end
    if card.edition.negative then return "Negative" end
    return "None"
end

function get_seal_name(card)
    if not card.seal then return "None" end
    local seal_map = {
        ["Gold"] = "Gold", ["Red"] = "Red", 
        ["Blue"] = "Blue", ["Purple"] = "Purple"
    }
    return seal_map[card.seal] or "None"
end

--[[
================================================================================
POKER HAND EVALUATION
================================================================================
]]--

function get_current_poker_hand()
    -- Get the current evaluated poker hand from selected cards
    if G.GAME.current_round and G.GAME.current_round.current_hand then
        local hand = G.GAME.current_round.current_hand.handname
        return hand or "High Card"
    end
    return "High Card"
end

function get_hand_levels()
    -- Returns the level of each poker hand type
    local levels = {}
    local hand_types = {
        "High Card", "Pair", "Two Pair", "Three of a Kind", "Straight",
        "Flush", "Full House", "Four of a Kind", "Straight Flush",
        "Five of a Kind", "Flush House", "Flush Five"
    }
    
    for _, hand_type in ipairs(hand_types) do
        local hand_data = G.GAME.hands[hand_type]
        levels[hand_type] = hand_data and hand_data.level or 1
    end
    
    return levels
end

--[[
================================================================================
JOKERS
================================================================================
]]--

function get_jokers()
    local jokers = {}
    
    if not G.jokers or not G.jokers.cards then
        return jokers
    end
    
    for i, joker in ipairs(G.jokers.cards) do
        table.insert(jokers, {
            name = joker.ability.name or "Unknown",
            
            -- Edition
            edition = get_joker_edition(joker),
            
            -- Value
            sell_value = joker.sell_cost or 0,
            
            -- Rarity
            rarity = get_joker_rarity(joker),
            
            -- For scaling jokers
            has_counter = joker.ability.extra and type(joker.ability.extra) == "table",
            counter = get_joker_counter(joker),
            
            -- Active state
            active = joker.ability.active or false,
        })
    end
    
    return jokers
end

function get_joker_edition(joker)
    if not joker.edition then return "None" end
    if joker.edition.foil then return "Foil" end
    if joker.edition.holo then return "Holographic" end
    if joker.edition.polychrome then return "Polychrome" end
    if joker.edition.negative then return "Negative" end
    return "None"
end

function get_joker_rarity(joker)
    local rarity = joker.config and joker.config.center and joker.config.center.rarity
    local rarity_map = {
        [1] = "Common",
        [2] = "Uncommon", 
        [3] = "Rare",
        [4] = "Legendary"
    }
    return rarity_map[rarity] or "Common"
end

function get_joker_counter(joker)
    -- Get scaling counter for jokers like Ride the Bus, Ice Cream, etc.
    if joker.ability.extra and type(joker.ability.extra) == "table" then
        return joker.ability.extra.count or joker.ability.extra.chips or 0
    end
    return 0
end

--[[
================================================================================
CONSUMABLES
================================================================================
]]--

function get_consumables()
    local consumables = {}
    
    if not G.consumeables or not G.consumeables.cards then
        return consumables
    end
    
    for i, card in ipairs(G.consumeables.cards) do
        table.insert(consumables, {
            name = card.ability.name or "Unknown",
            
            -- Type
            type = get_consumable_type(card),
            
            -- Value
            sell_value = card.sell_cost or 0,
            
            -- Can use now?
            usable = card.ability.consumeable and card.ability.consumeable.can_use or true,
        })
    end
    
    return consumables
end

function get_consumable_type(card)
    if card.ability.set == "Tarot" then return "tarot" end
    if card.ability.set == "Planet" then return "planet" end
    if card.ability.set == "Spectral" then return "spectral" end
    return "tarot"
end

--[[
================================================================================
SHOP
================================================================================
]]--

function get_shop_items()
    local items = {}
    
    -- Main shop items
    if G.shop and G.shop.cards then
        for i, card in ipairs(G.shop.cards) do
            table.insert(items, {
                name = card.ability and card.ability.name or "Unknown",
                type = get_shop_item_type(card),
                cost = card.cost or 0,
                rarity = get_shop_item_rarity(card),
                sold = card.sold or false,
            })
        end
    end
    
    -- Booster packs
    if G.shop_booster and G.shop_booster.cards then
        for i, card in ipairs(G.shop_booster.cards) do
            table.insert(items, {
                name = card.ability and card.ability.name or "Pack",
                type = "pack",
                cost = card.cost or 0,
                rarity = "Common",
                sold = card.sold or false,
            })
        end
    end
    
    -- Vouchers
    if G.shop_vouchers and G.shop_vouchers.cards then
        for i, card in ipairs(G.shop_vouchers.cards) do
            table.insert(items, {
                name = card.ability and card.ability.name or "Voucher",
                type = "voucher",
                cost = card.cost or 0,
                rarity = "Rare",
                sold = card.sold or false,
            })
        end
    end
    
    return items
end

function get_shop_item_type(card)
    if not card.ability then return "joker" end
    if card.ability.set == "Joker" then return "joker" end
    if card.ability.set == "Tarot" then return "tarot" end
    if card.ability.set == "Planet" then return "planet" end
    return "joker"
end

function get_shop_item_rarity(card)
    local rarity = card.config and card.config.center and card.config.center.rarity
    local rarity_map = {
        [1] = "Common",
        [2] = "Uncommon",
        [3] = "Rare", 
        [4] = "Legendary"
    }
    return rarity_map[rarity] or "Common"
end

--[[
================================================================================
BLIND SELECTION
================================================================================
]]--

function get_blind_select_info()
    return {
        small_target = G.GAME.blind and G.GAME.blind.chips or 300,
        big_target = G.GAME.blind and G.GAME.blind.chips * 1.5 or 450,
        boss_target = G.GAME.blind and G.GAME.blind.chips * 2 or 600,
        current = get_current_blind_index(),
    }
end

function get_current_blind_index()
    -- 0 = small, 1 = big, 2 = boss
    if G.GAME.blind and G.GAME.blind.name then
        if string.find(G.GAME.blind.name, "Small") then return 0 end
        if string.find(G.GAME.blind.name, "Big") then return 1 end
    end
    return 2  -- Boss
end

function get_available_blinds()
    -- Which blinds can be selected right now
    local blinds = {}
    
    if G.GAME.round_resets then
        local blind_on = G.GAME.round_resets.blind_on or "Small"
        
        if blind_on == "Small" then
            table.insert(blinds, "small")
        elseif blind_on == "Big" then
            table.insert(blinds, "big")
        else
            table.insert(blinds, "boss")
        end
    end
    
    return blinds
end

function can_skip_blind()
    -- Check if player has a skip tag
    if G.GAME.tags then
        for _, tag in ipairs(G.GAME.tags) do
            if tag.ability and tag.ability.skip_blind then
                return true
            end
        end
    end
    return false
end

function is_boss_blind()
    if G.GAME.blind and G.GAME.blind.boss then
        return true
    end
    return false
end

--[[
================================================================================
PACK OPENING
================================================================================
]]--

function get_pack_cards()
    local cards = {}
    
    if G.pack_cards and G.pack_cards.cards then
        for i, card in ipairs(G.pack_cards.cards) do
            table.insert(cards, {
                name = card.ability and card.ability.name or "Card",
                picked = card.picked or false,
            })
        end
    end
    
    return cards
end

function get_pack_picks_remaining()
    if G.pack_cards then
        return G.pack_cards.config and G.pack_cards.config.choose or 1
    end
    return 0
end

--[[
================================================================================
DECK STATS
================================================================================
]]--

function get_deck_stats()
    local stats = {
        Spades = 0,
        Hearts = 0,
        Clubs = 0,
        Diamonds = 0,
        Stone = 0,
        total = 0,
    }
    
    if G.deck and G.deck.cards then
        for _, card in ipairs(G.deck.cards) do
            local suit = get_suit_name(card)
            if stats[suit] then
                stats[suit] = stats[suit] + 1
            end
            
            if card.ability and card.ability.m_stone then
                stats.Stone = stats.Stone + 1
            end
            
            stats.total = stats.total + 1
        end
    end
    
    return stats
end

function get_blind_chips()
    if G.GAME.blind and G.GAME.blind.chips then
        return G.GAME.blind.chips
    end
    return 300
end

--[[
================================================================================
COMMAND HANDLING
================================================================================

Handle commands from the Python environment.
]]--

function handle_command(cmd)
    if cmd.type == "ping" then
        -- Just send current state
        send_state()
        
    elseif cmd.type == "toggle" then
        -- Toggle card selection
        local index = cmd.index + 1  -- Lua is 1-indexed
        if G.hand and G.hand.cards and G.hand.cards[index] then
            local card = G.hand.cards[index]
            if card.highlighted then
                G.hand:remove_from_highlighted(card)
            else
                G.hand:add_to_highlighted(card)
            end
        end
        send_state()
        
    elseif cmd.type == "play" then
        -- Play selected cards
        G.FUNCS.play_cards_from_highlighted()
        -- State will be sent after animation
        
    elseif cmd.type == "discard" then
        -- Discard selected cards
        G.FUNCS.discard_cards_from_highlighted()
        -- State will be sent after animation
        
    elseif cmd.type == "buy" then
        -- Buy item from shop slot
        local slot = cmd.slot + 1
        if G.shop and G.shop.cards and G.shop.cards[slot] then
            G.FUNCS.buy_from_shop(G.shop.cards[slot])
        end
        send_state()
        
    elseif cmd.type == "sell_joker" then
        -- Sell a joker
        local index = cmd.index + 1
        if G.jokers and G.jokers.cards and G.jokers.cards[index] then
            G.FUNCS.sell_card(G.jokers.cards[index])
        end
        send_state()
        
    elseif cmd.type == "use_consumable" then
        -- Use a consumable
        local index = cmd.index + 1
        if G.consumeables and G.consumeables.cards and G.consumeables.cards[index] then
            G.FUNCS.use_card(G.consumeables.cards[index])
        end
        send_state()
        
    elseif cmd.type == "reroll" then
        -- Reroll shop
        G.FUNCS.reroll_shop()
        send_state()
        
    elseif cmd.type == "next_round" then
        -- Leave shop, go to next blind
        G.FUNCS.skip_blind_select()  -- or appropriate function
        send_state()
        
    elseif cmd.type == "select_blind" then
        -- Select a blind to play
        local blind = cmd.blind  -- "small", "big", or "boss"
        if blind == "small" then
            G.FUNCS.select_blind(G.blind_select_opts.small)
        elseif blind == "big" then
            G.FUNCS.select_blind(G.blind_select_opts.big)
        else
            G.FUNCS.select_blind(G.blind_select_opts.boss)
        end
        send_state()
        
    elseif cmd.type == "skip_blind" then
        -- Skip the current blind (if tag available)
        G.FUNCS.skip_blind()
        send_state()
        
    elseif cmd.type == "pack_select" then
        -- Select a card from pack
        local index = cmd.index + 1
        if G.pack_cards and G.pack_cards.cards and G.pack_cards.cards[index] then
            G.FUNCS.select_pack_card(G.pack_cards.cards[index])
        end
        send_state()
        
    elseif cmd.type == "pack_skip" then
        -- Skip remaining pack picks
        G.FUNCS.skip_pack()
        send_state()
    end
end

--[[
================================================================================
STATE SENDING
================================================================================
]]--

function send_state()
    if not client then
        if not connect_to_server() then
            return
        end
    end
    
    local state = build_game_state()
    local json_str = json.encode(state) .. "\n"
    
    local success, err = client:send(json_str)
    if not success then
        print("Failed to send state: " .. tostring(err))
        client = nil
    end
end

-- Call this after every game state change
-- Hook into Balatro's state change functions

--[[
================================================================================
EXAMPLE INTEGRATION
================================================================================

Add these hooks to your mod to send state at the right times:

1. After hand is drawn:
   local original_draw_hand = G.FUNCS.draw_hand
   G.FUNCS.draw_hand = function(...)
       original_draw_hand(...)
       send_state()
   end

2. After play animation completes:
   -- In your animation completion callback
   send_state()

3. When entering shop:
   local original_start_shop = G.FUNCS.start_shop  
   G.FUNCS.start_shop = function(...)
       original_start_shop(...)
       send_state()
   end

4. Continuous polling for commands:
   function update_rl_connection(dt)
       if client then
           client:settimeout(0)
           local line, err = client:receive("*l")
           if line then
               local cmd = json.decode(line)
               handle_command(cmd)
           end
       end
   end
   -- Add to your update loop

]]--