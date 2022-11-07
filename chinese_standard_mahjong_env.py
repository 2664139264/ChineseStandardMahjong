from copy import deepcopy
from collections import Counter
from typing import Any, Dict, List, Iterable, Union, Tuple

import numpy as np

# https://github.com/ailab-pku/PyMahjongGB
from MahjongGB import MahjongFanCalculator

from multiagent_env import MultiAgentEnv

class ChineseStandardMahjongEnv(MultiAgentEnv):

    # 上帝视角状态类型
    StateType = Dict[str, Any]
    # 玩家视角观测类型
    ObservationType = Dict[str, Any]
    # 动作类型：Pass/Hu/Chi/Peng/Gang/AnGang/BuGang
    ActionType = str
    # 动作名称：Pass/Hu/Chi[中间的牌]/Peng, Gang, AnGang, BuGang[牌]
    ActionNameType = str
    # 牌种类：风箭万条饼
    CardType = str
    # 牌名称：东西南北风/中发白箭/1-9万条饼
    CardNameType = str
    # step返回的信息
    StepInfoType = None
    # close返回的信息
    CloseInfoType = None
    # 玩家的id：0-3 / None表示环境本身
    PlayerIDType = Union[int, None]
    # 算番函数返回类型：番值、个数、番名、番名(英文)
    FanCalculatorReturnType = Union[Tuple[Tuple[int, int, str, str]], None]

    # 4人游戏
    _n_players = 4
    # 每个人初始手牌13张
    _n_hand_card = 13
    # 每种牌有4张
    _n_duplicate_cards = 4
    # 4种风向
    _n_winds = 4
    # 3种箭牌
    _n_dragons = 3
    # 序数牌1-9
    _n_ordinals = 9
    # 非序数牌有2种，风、箭
    _n_non_ordinal_types = 2
    # 序数牌有3种，万、条、饼
    _n_ordinal_types = 3
    # 吃牌副露长度为3
    _chi_tile_length = 3
    # 碰牌副露长度为3
    _peng_tile_length = 3
    # 杠牌副露长度为4
    _gang_tile_length = 4
    # 最小成和番数为8
    _min_win_fan = 8

    # 将种类和细分类结合到一起，生成完整牌张/动作名称
    _type_detail_combiner = lambda type_list, detail_list : tuple(f'{t}{d}' for (i, t) in enumerate(type_list) for d in detail_list[i])

    # 生成名称到id的映射
    _id_dict_generator = lambda _list : {k : i for i, k in enumerate(_list)}

    # 牌的种类：风箭万条饼
    _card_types = ('F', 'J', 'W', 'T', 'B')

    # 每种牌的细分
    _card_details = (
        tuple(range(1, _n_winds+1)),                             # F1-F4: 东南西北
        tuple(range(1, _n_dragons+1)),                           # J1-J3: 中发白
        *((tuple(range(1, _n_ordinals+1)), ) * _n_ordinal_types) # W1-W9, T1-T9, B1-B9
    )

    # 牌名列表
    _card_names = _type_detail_combiner(_card_types, _card_details)

    _chiable = lambda card : not (card.startswith('F') or card.startswith('J') or card.endswith('1') or card.endswith('9'))
    
    # 可以作为吃牌标识的牌张：包括2-8万条饼
    _chiable_card_names = tuple(filter(_chiable, _card_names))

    # 牌名到编号的映射
    _card_ids = _id_dict_generator(_card_names)

    # 动作种类
    _action_types = ('Pass', 'Hu', 'Play', 'Chi', 'Peng', 'Gang', 'AnGang', 'BuGang')

    # 每种动作的细分类
    _action_details = (
        ('',),               # Pass
        ('',),               # Hu
        _card_names,         # Play
        _chiable_card_names, # Chi
        _card_names,         # Peng
        _card_names,         # Gang
        _card_names,         # AnGang
        _card_names          # BuGang
    )

    # 动作名列表
    _action_names = _type_detail_combiner(_action_types, _action_details)

    # 动作名到编号的映射
    _action_ids = _id_dict_generator(_action_names)

    # 玩家人数：4
    @property
    def n_players(self) -> int: return ChineseStandardMahjongEnv._n_players

    # 最小和牌番数：8
    @property
    def min_win_fan(self) -> int: return ChineseStandardMahjongEnv._min_win_fan
    
    # 编号到牌名的映射
    @classmethod
    def card_name(cls, card_id:int) -> Union[CardNameType, None]:
        return None if card_id not in range(len(cls._card_names)) else cls._card_names[card_id]
    
    @classmethod
    def _chiable(cls, card:CardNameType) -> bool:
        return not (card is None or card.startswith('F') or card.startswith('J') or card.endswith('1') or card.endswith('9'))

    # 牌名到编号的映射
    @classmethod
    def card_id(cls, card_name:CardNameType) -> Union[int, None]:
        return cls._card_ids.get(card_name, None)
    
    # 动作编号到动作名的映射
    @classmethod
    def action_name(cls, action_id:int) -> Union[ActionNameType, None]:
        return None if action_id not in range(len(cls._action_names)) else cls._action_names[action_id]
    
    # 动作名到编号的映射
    @classmethod
    def action_id(cls, action_name:ActionNameType) -> Union[int, None]:
        return cls._action_ids.get(action_name, None)
    
    # 将字符串表示的牌转为 (牌类型, 牌面点数)
    @classmethod
    def card_to_tuple(cls, card:CardNameType) -> Union[Tuple[CardType], None]:
        return (None, None) if card is None else (card[0], card[1])

    # 把字符串表示的动作转为 (动作类型, 牌张)
    @classmethod
    def action_to_tuple(cls, action:ActionNameType) -> Tuple[ActionType, CardNameType]:
        for action_type in cls._action_types:
            if action.startswith(action_type):
                card = action[len(action_type):]
                return (action_type, card)

    # 风向：按照1东，2南，3西，4北的顺序轮转
    @classmethod
    def _next_wind(cls, wind:int, n:int=1) -> int:
        return (wind + n - 1) % cls._n_winds + 1

    # 玩家：按照0,1,2,3的顺序轮转
    @classmethod
    def _next_player(cls, player:PlayerIDType, n:int=1) -> PlayerIDType:
        return (player + n) % cls._n_players

    # 判断输入作为初始化项的牌墙是否合法
    @classmethod
    def _is_legal_wall(cls, cards:Iterable[CardNameType]) -> bool:
        cards_counter = Counter(cards)
        return (
            # 每种张数一致
            all(map(lambda v : v == cls._n_duplicate_cards, cards_counter.values())) and
            # 集合一致
            set(cards) == set(cls._card_names)
        )
    
    # 返回当前待决策的玩家
    @property
    def active_player(self): return self._active_player

    # 返回玩家的动作空间，调用_update_action_space_and_fan之后计算得出
    @property
    def action_space(self): return self._action_space

    # 是否结束
    @property
    def done(self): return self._done

    # 返回玩家已经形成的番，调用_update_action_space_and_fan之后计算得出
    @property
    def fan(self) -> FanCalculatorReturnType: return self._fan

    # 赢家，在某一方和牌时设置
    @property
    def winner(self) -> PlayerIDType: return self._winner

    # 每个玩家的分数
    @property
    def scores(self) -> Tuple[int]: return self._scores
    
    # 当前玩家的观测
    @property
    def observation(self) -> ObservationType: return deepcopy(self._observation)

    # 上帝视角的全局信息
    @property
    def state(self) -> StateType: return deepcopy(self._state)

    # 游戏进行的历史，第一项为游戏状态，之后均为玩家做出的动作
    @property
    def history(self) -> List[Union[StateType, ActionNameType]]: return deepcopy(self._history)

    # 根据算番库的返回，计算总番数
    @staticmethod
    def sum_fan(fan:FanCalculatorReturnType) -> int:
        return 0 if fan is None else sum(map(lambda f : f[0] * f[1], fan))

    # 当前待决策的牌以及来源：若from非None，则是玩家打出/补杠的牌。
    # 否则若card是None，当前玩家恰好吃碰杠完成且只能打出牌；若card非None则为从环境摸牌。
    @property
    def current_card_and_source(self):
        return (self._current_card, self._current_card_from)
    
    # 设置待决策的牌及其来源
    def _set_current_card_and_source(self, card:CardNameType, player:PlayerIDType):
        self._current_card, self._current_card_from = card, player

    def __init__(self, config:Dict):
        super().__init__()
        self.config = config
        # 设置牌墙：self._wall, 初始手牌：self._initial_hand_cards
        self._general_wall_initializer(config)
        # 设置圈风：self.prevalent_wind, 门风：self.seat_winds
        prevalent_wind = config.get('prevalent_wind', None)
        self._set_initial_winds(prevalent_wind)
        # 初始化游戏状态参数
        self._game_state_initializer()

    # 根据配置来决定如何重置环境
    def reset(self):
        reset_mode = self.config['reset_mode']
        cards_reset_mode = reset_mode.get('cards', 'random')
        if cards_reset_mode == 'fixed':
            self._general_wall_initializer(self.config)
        elif cards_reset_mode == 'random':
            self._seed_wall_initializer()

        wind_reset_mode = reset_mode.get('wind', 'fixed')
        if wind_reset_mode == 'next':
            self._set_next_winds()
        self._game_state_initializer()
    
    # 初始化门风、圈风
    def _set_initial_winds(self, prevalent_wind:int=None):
        
        self.prevalent_wind = prevalent_wind
        if prevalent_wind is None:
            self.prevalent_wind = np.random.randint(1, self._n_winds+1)

        self.seat_winds = tuple(
            self._next_wind(self.prevalent_wind, i)
            for i in range(self._n_winds)
        )

    # 轮转门风、圈风
    def _set_next_winds(self):
        next_wind_0 = self._next_wind(self.seat_winds[0])
        if next_wind_0 == self.prevalent_wind:
            self._set_initial_winds(self._next_wind(self.prevalent_wind))
        else:
            self.seat_winds = tuple(map(self._next_wind, self.seat_winds))

    # 初始化手牌和牌墙：分为给定随机种子、给定牌墙、完全随机三种方式
    def _general_wall_initializer(self, config:Dict):
        if 'cards' in config.keys():
            self._fixed_wall_initializer(config['cards'], need_validation=True)
        elif 'seed' in config.keys():
            self._seed_wall_initializer(config['seed'])
        else:
            self._seed_wall_initializer()

    # 用随机种子初始化手牌和牌墙
    def _seed_wall_initializer(self, seed:int=None):
        if seed is not None:
            np.random.seed(seed)
        cards = list(self._card_names * self._n_duplicate_cards)
        np.random.shuffle(cards)
        self._fixed_wall_initializer(tuple(cards))

    # 用给定牌墙初始化手牌和牌墙
    def _fixed_wall_initializer(self, cards:Tuple[CardNameType], need_validation=False):
        assert not need_validation or self._is_legal_wall(cards)

        # 每个人初始的手牌
        self._initial_hand_cards = tuple(
            cards[self._n_hand_card * i : self._n_hand_card * (i+1)] 
            for i in range(self.n_players)
        )

        complete_wall = tuple(cards[self._n_hand_card * self.n_players:])
        wall_len = len(complete_wall) // self.n_players

        # 每个玩家手上的牌墙
        self._walls = tuple(
            complete_wall[wall_len * i : wall_len * (i+1)]
            for i in range(self.n_players)
        )

    # 初始化游戏状态
    def _game_state_initializer(self):

        # 游戏是否结束
        self._done = False
        # 游戏中各家得分情况
        self._scores = (0,) * self.n_players
        # 玩家牌若成和牌型，则保存各个番种的番数、数量、名称，未成和型则为None，每次step后更新
        self._fan = None
        # 赢家
        self._winner = None
        # 游戏当前的状态（全局视角），每次step后更新
        self._state = None
        # 当前玩家的可观测信息，每次step后更新
        self._observation = None
        # 要发的下一张牌
        self._wall_pointers = [0 for _ in range(self.n_players)]
        # 当前应当决策的玩家，无论门风圈风，0号玩家固定为先决策的玩家。
        self._active_player = 0
        # 当前玩家的动作空间
        self._action_space = list()
        # 当前等待玩家决定吃碰杠的牌来自哪里。如果是None则来自牌墙，不可吃碰杠，但可补杠以及暗杠
        self._current_card_from = None
        # 当前等待玩家决定吃碰杠、补杠暗杠的牌
        self._current_card = None
        # 首次发牌
        self._deal_card()
        # 记录这一圈的动作，根据先和牌>后和牌>碰杠>吃的顺序处理
        self._unprocessed_actions = list()
        # 每个玩家形成的牌河
        self._discard_histories = tuple(list() for _ in range(self.n_players))
        # 每个玩家手上的副露（表示为('Chi/Peng/Gang', card, card_from_player)
        self._shown_packs = tuple(list() for _ in range(self.n_players))
        # 每个玩家手上的暗杠（表示为card牌张名称）
        self._hidden_packs = tuple(list() for _ in range(self.n_players))
        # 每个人手牌计数器（每种牌有多少张）
        self._hand_card_counters = tuple(Counter(self._initial_hand_cards[player]) for player in range(self.n_players))
        # 明牌计数器，计算牌河、副露中的牌已经出现了多少张，不计算暗杠
        self._shown_card_counter = Counter()
        # 当前是否是杠牌之后摸牌：判定杠上开花/抢杠和（抢补杠）
        self._is_about_kong = False
        # 当前是否进行到最后一圈牌：判定海底捞月、妙手回春
        self._is_wall_last = False
        # 更新玩家的动作空间，和当前成番情况
        self._update_action_space_and_fan()
        # 更新玩家的观测信息和全局状态信息
        self._update_observation_and_state()
        # 决策历史，第一项是初始状态，之后每一项是各个玩家的动作
        self._history = list()
        self._history.append(self.state)
    
    # 在玩家改变之后，更新当前待决策玩家的观测和状态信息
    def _update_observation_and_state(self):
        self._observation = {
            # 圈风
            'prevalent_wind' : self.prevalent_wind,
            # 门风
            'seat_winds' : self.seat_winds,
            # 每位玩家的牌墙各自还剩多少张
            'wall_remains' : tuple(self.wall_remain(i) for i in range(self.n_players)),
            # 游戏是否结束
            'done' : self.done,
            # 分数
            'scores' : self.scores,
            # 如果当前游戏未结束，则为自己形成的番；否则为胜者形成的番
            'fan' : self.fan,
            # 胜者
            'winner' : self.winner,
            # 自己的手牌
            'hand_card' : self._hand_card_counters[self.active_player],
            # 每个玩家的手牌数目
            'n_hand_cards' : tuple(map(lambda x : sum(x.values()), self._hand_card_counters)),
            # 每个玩家的副露
            'shown_packs' : self._shown_packs,
            # 每个玩家的暗杠数量
            'n_hidden_packs' : tuple(map(len, self._hidden_packs)),
            # 每个玩家的牌河
            'discard_histories' : self._discard_histories,
            # 当前待决策的牌，可能来自发牌也可能来自吃碰杠
            'current_card' : self._current_card,
            # 当前待决策的牌的来源，如果为None则为环境发牌
            'current_card_from' : self._current_card_from
        }
        self._state = {
            'prevalent_wind' : self.prevalent_wind,
            'seat_winds' : self.seat_winds,
            # 初始牌墙
            'walls' : self._walls,
            'wall_remains' : tuple(self.wall_remain(i) for i in range(self.n_players)),
            'done' : self.done,
            'fan' : self.fan,
            'winner' : self.winner,
            'scores' : self.scores,
            # 每个玩家的手牌
            'hand_cards' : self._hand_card_counters,
            'shown_packs' : self._shown_packs,
            # 每个玩家的暗杠
            'hidden_packs' : self._hidden_packs,
            'discard_histories' : self._discard_histories,
            'current_card' : self._current_card,
            'current_card_from' : self._current_card_from
        }

    # 某个玩家的牌墙还剩多少牌
    def wall_remain(self, player:PlayerIDType):
        return len(self._walls[player]) - self._wall_pointers[player]

    # 发一张牌
    def _deal_card(self):

        # 如果自己的牌发完了，游戏结束
        if self.wall_remain(self.active_player) == 0:
            self._done = True
            self._set_current_card_and_source(None, None)
            return
        
        # 如果下家牌墙空，进入海底状态，打出的牌不可吃碰杠，只能海底捞月/妙手回春和流局
        if self.wall_remain(self._next_player(self.active_player)) == 0:
            self._is_wall_last = True
        
        card = self._walls[self.active_player][self._wall_pointers[self.active_player]]
        self._wall_pointers[self.active_player] += 1

        self._set_current_card_and_source(card, None)
    
    # 在当前玩家改变之后，更新该玩家的动作空间和已经形成的番（考虑当前牌）
    def _update_action_space_and_fan(self):
        
        self._action_space.clear()
        has_unprocessed_actions = len(self._unprocessed_actions) != 0
        # 牌局已经结束
        if self.done:
            return
        
        # 别人打出牌，吃碰杠阶段
        if has_unprocessed_actions and self._unprocessed_actions[0].startswith('Play'):
            self._action_space.append('Pass')
            self._add_hu_actions_and_update_fan()
            # 海底牌不能吃碰杠
            if not self._is_wall_last:
                self._add_peng_actions()
                if len(self._unprocessed_actions) == 1:
                    self._add_chi_actions()
                # 自己牌墙里有牌才能杠
                if self.wall_remain(self.active_player) > 0:
                    self._add_gang_actions()
            
        # 别人补杠，抢杠和阶段
        elif has_unprocessed_actions and self._unprocessed_actions[0].startswith('BuGang'):
            self._action_space.append('Pass')
            self._add_hu_actions_and_update_fan()
        
        # 杠后摸打，可能开花阶段
        elif self._is_about_kong and self._current_card is not None and self._current_card_from is None:
            self._add_play_actions()
            self._add_hu_actions_and_update_fan()

        # 接受发牌阶段
        elif self._current_card is not None and self._current_card_from is None:
            self._add_play_actions()
            self._add_hu_actions_and_update_fan()
            # 自己牌墙里有牌才能杠
            if self.wall_remain(self.active_player) > 0:
                self._add_angang_actions()
                self._add_bugang_actions()

        # 吃碰完牌、只能打牌阶段
        elif self._current_card is None and self._current_card_from is None:
            self._add_play_actions()
        
    # 生成打牌动作
    def _add_play_actions(self):
        self._action_space.extend(f'Play{card}' 
            for card, card_num in self._hand_card_counters[self.active_player].items() if card_num > 0
        )
        if self._current_card is not None:
            self._action_space.append(f'Play{self._current_card}')

    # 生成吃牌动作
    def _add_chi_actions(self):
        hand_cards = self._hand_card_counters[self.active_player]
        current_card = self._current_card
        if not self._chiable(self.card_to_tuple(current_card)[0]):
            return
        current_card_id = self.card_id(current_card)
        hand_cards[current_card] += 1
        chiable_cards = [
            self.card_name(current_card_id + i) for i in range(-1, self._chi_tile_length-1)
            if self._chiable(self.card_name(current_card_id + i))
            and all(hand_cards[self.card_name(current_card_id + i + j)] > 0 for j in range(-1, self._chi_tile_length-1))
        ]
        hand_cards[current_card] -= 1
        self._action_space.extend(f'Chi{card}' for card in chiable_cards)

    # 添加碰牌动作
    def _add_peng_actions(self):
        if self._hand_card_counters[self.active_player][self._current_card] + 1 == self._peng_tile_length:
            self._action_space.append(f'Peng{self._current_card}')

    # 添加杠牌动作
    def _add_gang_actions(self):
        if self._hand_card_counters[self.active_player][self._current_card] + 1 == self._gang_tile_length:
            self._action_space.append(f'Gang{self._current_card}')

    # 添加暗杠动作，仅在摸牌时
    def _add_angang_actions(self):
        # 手牌暗杠
        for card, card_num in self._hand_card_counters[self.active_player].items():
            if card_num == self._gang_tile_length:
                self._action_space.append(f'AnGang{card}')
        # 摸牌暗杠
        if self._hand_card_counters[self.active_player][self._current_card] + 1 == self._gang_tile_length:
            self._action_space.append(f'AnGang{self._current_card}')

    # 查询副露中碰某牌的index，补杠时需要删去此副露
    def _peng_pack_index_of(self, card:Union[CardNameType, None]) -> Union[int, None]:
        for i, pack in enumerate(self._shown_packs[self.active_player]):
            if (pack[0], pack[1]) == ('Peng', card):
                return i
        return None
                
    # 添加补杠动作，仅在摸牌时
    def _add_bugang_actions(self):
        # 手牌补杠
        for card, card_num in self._hand_card_counters[self.active_player].items():
            if card_num == 1 and self._peng_pack_index_of(card) is not None:
                self._action_space.append(f'BuGang{card}')
        # 摸牌补杠
        if self._peng_pack_index_of(self._current_card) is not None:
            self._action_space.append(f'BuGang{self._current_card}')

    # 更新成番情况，如果大于等于起和番则添加和牌动作
    def _add_hu_actions_and_update_fan(self):
        self._call_fan_calculator()
        if self.sum_fan(self.fan) >= self.min_win_fan:
            self._action_space.append('Hu')
    
    # 判定当前这张牌是否是绝张
    @property
    def _is_last_card_shown(self) -> bool:
        return self._shown_card_counter[self._current_card] + 1 == self._n_duplicate_cards
    
    # 将暗杠和吃碰杠结合起来
    def _combine_packs(self) -> Tuple[Tuple[ActionType, CardNameType, int]]:
        return tuple(map(self._reformat_packs, self._shown_packs[self._active_player] + self._hidden_packs[self._active_player]))

    # 把动作tuple转为算番库的输入pack要求的格式
    @classmethod
    def _reformat_packs(cls, record:Union[Tuple[ActionType, CardNameType, PlayerIDType], CardNameType]) -> Tuple[ActionType, CardNameType, int]: 
        return (
            # 'AnGang'
            ('GANG', record, 0) if isinstance(record, cls.ActionNameType)
            # 'BuGang', 'Gang'
            else ('GANG', record[1], 1) if 'Gang' in record[0]
            # 'Chi', 'Peng'
            else (record[0].upper(), record[1], 1)
        )

    # 将手牌计数器转为tuple表示
    def _generate_hand(self):
        return sum(((card,) * card_num for card, card_num in self._hand_card_counters[self.active_player].items()), start=tuple())

    # 更新成番情况
    def _call_fan_calculator(self) -> FanCalculatorReturnType:
        try:
            self._fan = MahjongFanCalculator(
                pack = self._combine_packs(),
                hand = self._generate_hand(),
                winTile = self._current_card,
                flowerCount = 0,
                isSelfDrawn = self._current_card_from == None,
                is4thTile = self._is_last_card_shown,
                isAboutKong = self._is_about_kong,
                isWallLast = self._is_wall_last,
                seatWind = self.seat_winds[self.active_player]-1,
                prevalentWind = self.prevalent_wind-1,
                verbose = True
            )
        except TypeError:
            self._fan = None
    
    # 生成每个玩家的分数
    def _generate_scores(self):
        # 流局各玩家分数为0，不变
        if self.winner is None:
            return

        is_self_drawn = self._current_card_from == None
        fan_sum = self.sum_fan(self.fan)
        # 自摸成和分数：三倍(番+起和番)
        self_drawn_winner_score = (self.min_win_fan + fan_sum) * (self.n_players - 1)
        # 自摸输掉的分数：番+起和番
        self_drawn_loser_score = -(self.min_win_fan + fan_sum)
        # 点炮输掉的分数：番+起和番
        offer_loser_score = self_drawn_loser_score
        # 没点炮输掉的分数：起和番
        non_offer_loser_score = -self.min_win_fan
        # 被点炮赢的分数
        offered_winner_score = -offer_loser_score - non_offer_loser_score * (self.n_players-1-1)

        get_score = (
            # 自摸和牌的情况
            (lambda player : (
                self_drawn_winner_score if player == self.winner 
                else self_drawn_loser_score)
            # 点炮和牌的情况
            ) if is_self_drawn else (
                lambda player : (
                    offered_winner_score if player == self.winner 
                    else offer_loser_score if player == self._current_card_from 
                    else non_offer_loser_score
                )
            )
        )
        self._scores = tuple(map(get_score, range(self.n_players)))
    
    # 将某张牌暴露为明牌
    def _add_visible_card(self, card:CardNameType, n:int=1):
        self._shown_card_counter[card] += n

    # 把打出的牌加入牌河中
    def _add_discard_history(self, card:CardNameType):
        # 牌河中的牌也要加入明牌
        self._add_visible_card(card)
        self._discard_histories[self.active_player].append(card)
    
    # 玩家打出一张牌后过一圈，所有玩家决策完是否吃碰杠之后调用，将副露暴露出来
    def _add_shown_pack(self, action:ActionNameType, card_from:Union[PlayerIDType, None], card_to:PlayerIDType):
        action_type, card = self.action_to_tuple(action)
        assert action_type in {'Chi', 'Peng', 'Gang', 'BuGang'}
        self._shown_packs[card_to].append((action_type, card, card_from))
        if action_type == 'Chi':
            for i in range(-1, self._chi_tile_length-1):
                self._add_visible_card(self.card_name(self.card_id(card)+i))
        elif action_type == 'Peng':
            self._add_visible_card(card, self._peng_tile_length)
        elif action_type == 'Gang':
            self._add_visible_card(card, self._gang_tile_length)
        elif action_type == 'BuGang':
            self._add_visible_card(card)
    
    # 玩家暗杠之后调用，将暗杠的pack加入列表
    def _add_hidden_pack(self, action:ActionNameType):
        action_type, card = self.action_to_tuple(action)
        assert action_type == 'AnGang'
        self._hidden_packs[self.active_player].append(card)
    
    # 玩家增减手牌
    def _add_hand_card(self, card:CardNameType, n:int=1):
        self._hand_card_counters[self.active_player][card] += n

    # 将动作添加到历史
    def _add_history(self, action:ActionNameType, card_from:Union[PlayerIDType, None], card_to:PlayerIDType):
        self._history.append(self.action_to_tuple(action) + (card_from, card_to))

    def render(self):
        pass

    def close(self):
        pass

    # 状态发生转移，返回：执行动作过程中发生的事件信息
    def step(self, action:ActionNameType) -> StepInfoType:
        
        if self.done:
            return

        assert action in self.action_space
        action_type, card = self.action_to_tuple(action)

        # 如果游戏还未结束
        self._add_history(action, card_from=self.active_player, card_to=self._current_card_from)

        if action_type == 'Hu':
            self._done = True
            self._winner = self.active_player
            self._generate_scores()
            return

        # 暗杠不可抢杠和
        if action_type == 'AnGang':
            self._is_about_kong = True
            # 添加副露
            self._add_hidden_pack(action)
            # 修改手牌
            self._add_hand_card(self._current_card)
            self._add_hand_card(card, -self._gang_tile_length)
            self._deal_card()
            # 等待该玩家打牌/杠上开花

        # 打牌需要轮一圈决策
        elif action_type == 'Play':
            self._is_about_kong = False
            # 修改手牌
            if self._current_card is not None:
                self._add_hand_card(self._current_card)
            self._add_hand_card(card, -1)
            # 等待大家吃碰杠
            self._set_current_card_and_source(card, self.active_player)
            self._unprocessed_actions.append(action)
            self._active_player = self._next_player(self.active_player)
        
        # 补杠需要轮一圈决策
        elif action_type == 'BuGang':
            self._is_about_kong = True
            # 修改手牌
            self._add_hand_card(self._current_card)
            # 等待大家抢杠和
            self._set_current_card_and_source(card, self.active_player)
            self._unprocessed_actions.append(action)
            self._active_player = self._next_player(self.active_player)
            # 等到其他人都决定完是否抢杠和之后再 deal_card

        # 应对别人打出的牌
        elif action_type in {'Chi', 'Peng', 'Gang', 'Pass'}:
            self._unprocessed_actions.append(action)
            self._active_player = self._next_player(self.active_player)

        # 如果打出的牌/补杠的牌轮过一圈，则需要确定牌张归属、重新确定牌权
        if len(self._unprocessed_actions) == self.n_players:
            # 此时active_player回到打出/补杠的人手里
            last_round_player = self.active_player
            # 看看这一圈的第一个动作是打牌还是补杠
            a0 = self._unprocessed_actions[0]
            a0_type, a0_card = self.action_to_tuple(a0)

            # 补杠成功
            if a0_type == 'BuGang':
                self._is_about_kong = True
                # 把碰的那个tile删掉
                peng_pack_id = self._peng_pack_index_of(a0_card)
                del self._shown_packs[self.active_player][peng_pack_id]
                # 添加新的副露
                self._add_shown_pack(action=a0, card_from=None, card_to=self.active_player)
                # 修改手牌，删去补杠的那张
                self._add_hand_card(a0_card, -1)
                # 补牌，等待玩家打出一张牌
                self._deal_card()
            
            # 打出的海底牌无人成和，结束
            elif self._is_wall_last:
                self._done = True
                return

            elif a0_type == 'Play':
                # 这一圈其他人的动作
                for i, a in enumerate(self._unprocessed_actions[1:]):
                    a_type, a_card = self.action_to_tuple(a)
                    if a_type == 'Gang':
                        # 可以杠上开花
                        self._is_about_kong = True
                        # 牌权属于杠牌的人
                        self._active_player = self._next_player(last_round_player, i+1)
                        # 加入杠牌副露
                        self._add_shown_pack(action=a, card_from=last_round_player, card_to=self.active_player)
                        # 更新手牌
                        self._add_hand_card(a_card, 1-self._gang_tile_length)
                        # 杠牌需要补摸一张
                        self._deal_card()
                        # 等待杠牌的玩家决策
                        break
                    elif a_type == 'Peng':
                        self._is_about_kong = False
                        # 牌权属于碰牌的人
                        self._active_player = self._next_player(last_round_player, i+1)
                        # 加入碰牌副露
                        self._add_shown_pack(action=a, card_from=last_round_player, card_to=self.active_player)
                        # 更新手牌
                        self._add_hand_card(a_card, 1-self._peng_tile_length)
                        # 当前只能打牌
                        self._set_current_card_and_source(None, None)
                        # 等待碰牌的人决策
                        break
                # 如果没有遇到碰杠的情况：只有过或者吃
                else:
                    self._is_about_kong = False
                    # 看看有没有被下家吃
                    a1 = self._unprocessed_actions[1]
                    a1_type, a1_card = self.action_to_tuple(a1)
                    # 牌被吃了
                    if a1_type == 'Chi':
                        # 牌权属于吃牌的玩家
                        self._active_player = self._next_player(last_round_player)
                        # 添加副露
                        self._add_shown_pack(action=a1, card_from=last_round_player, card_to=self.active_player)
                        # 修改手牌
                        self._add_hand_card(self._current_card)
                        for i in range(-1, self._chi_tile_length-1):
                            self._add_hand_card(self.card_name(self.card_id(a1_card)+i), -1)
                        # 等待玩家打出一张牌
                        self._set_current_card_and_source(None, None)
                    # 牌没有被吃，则需要丢入牌河
                    else:
                        # 加入牌河中
                        self._add_discard_history(self._current_card)
                        # 牌权交给下一个玩家
                        self._active_player = self._next_player(self.active_player)
                        # 发一张牌
                        self._deal_card()

            self._unprocessed_actions.clear()

        self._update_action_space_and_fan()
        self._update_observation_and_state()

def generate_log(env):
    print(f'prevalent_wind: {env.prevalent_wind}, seat_winds: {env.seat_winds}')
    print(f'wall: {env._walls}')
    while not env.done:
        print(f'player: {env.active_player}')
        print(f'hand: {(env._generate_hand())}, len: {len(env._generate_hand())}, current_card: {env._current_card}, from: {env._current_card_from}')
        print(f'visible: {env._shown_packs[env.active_player]}')
        print(f'hidden: {env._hidden_packs[env.active_player]}')
        action_space = c.action_space
        print(f'action_space: {sorted(action_space)}')
        a = np.random.choice(action_space, 1)[0]
        for action in action_space:
            if not action.startswith('Pass') and not action.startswith('Play'):
                a = action
                break
        print(f'ACTION: {a}')
        print(f'wall_remain: {env.wall_remain(env.active_player)}')
        print(f'wall: {env._walls[env.active_player]}')
        print(env.fan)
        print(env._is_wall_last)
        env.step(a)
        print()

if __name__ == '__main__':

    c = ChineseStandardMahjongEnv({
        'prevalent_wind' : 1,
        'reset_mode' : {
            'cards' : 'random',
            'wind' : 'next'
        }
    })
    
    generate_log(c)