import re
import sys
from copy import deepcopy
from typing import List, Tuple, Counter, Iterable, Callable
from collections import Counter

from agent import Agent
from botzone_adapter import BotzoneAdapter
from chinese_standard_mahjong_env import ChineseStandardMahjongEnv

# 适配Botzone简单交互长时运行模式，交互格式参考https://wiki.botzone.org.cn/index.php?title=Chinese-Standard-Mahjong
class ChineseStandardMahjongBotzoneAdapter(BotzoneAdapter):

    def __init__(self):
        self.reset()
    
    def reset(self):
        # 内置环境
        self._env = ChineseStandardMahjongEnv(config=dict())
        # 我方的id，按照ChineseStandardMahjongEnv的表示
        self._my_id = None
        # 我方的门风，按照Botzone的表示方式
        self._my_seat_wind = None
        # 各方牌墙剩余
        self._wall_remains = [len(self._env._card_names) * self._env._n_duplicate_cards // self._env.n_players - self._env._n_hand_card for _ in range(self._env.n_players)]
        # 各方手牌剩余
        self._n_hand_cards = [self._env._n_hand_card for _ in range(self._env.n_players)]
        # 各方暗杠数量
        self._n_hidden_packs = [0 for _ in range(self._env.n_players)]
        # 记录玩家暗杠的牌
        self._last_anganged_card = None
        # 处理过的历史长度
        self._processed_history_length = 0
        # 等待发牌状态
        self._env._set_current_card_and_source(None, None)

    # 返回我方观测信息，同ChineseStandardMahjongEnv中的observation
    @property
    def observation(self) -> ChineseStandardMahjongEnv.ObservationType:
        return deepcopy({
            # 圈风
            'prevalent_wind' : self._env.prevalent_wind,
            # 门风
            'seat_winds' : self._env.seat_winds,
            # 每位玩家的牌墙各自还剩多少张
            'wall_remains' : tuple(self._wall_remains),
            # 游戏是否结束
            'done' : self._env.done,
            # 分数
            'scores' : self._env.scores,
            # 如果当前游戏未结束，则为自己形成的番；否则为胜者形成的番
            'fan' : self._env.fan,
            # 胜者
            'winner' : self._env.winner,
            # 动作空间
            'action_space' : self.action_space,
            # 自己的手牌
            'hand_card' : self._env._hand_card_counters[self._my_id],
            # 每个玩家的手牌数目
            'n_hand_cards' : tuple(self._n_hand_cards),
            # 每个玩家的副露
            'shown_packs' : self._env._shown_packs,
            # 我方暗杠
            'hidden_pack' : self._env._hidden_packs[self._my_id],
            # 每个玩家的暗杠数量
            'n_hidden_packs' : tuple(self._n_hidden_packs),
            # 每个玩家的牌河
            'discard_histories' : self._env._discard_histories,
            # 当前待决策的牌，可能来自发牌也可能来自吃碰杠
            'current_card' : self._env._current_card,
            # 当前待决策的牌的来源，如果为None则为环境发牌
            'current_card_from' : self._env._current_card_from
        })
    
    # 更新动作空间和成番情况
    def _update_action_space_and_fan(self):
        
        # 因为需要对其他人打牌等做出回应，需要先将active_player切换到自己
        active_player = self._env._active_player
        self._env._active_player = self._my_id
        self._env._update_action_space_and_fan()
        self._env._active_player = active_player

        # 无可选动作需要Pass，自己杠完牌也只能pass，别人摸牌自己只能Pass。
        if len(self._env._action_space) == 0 or self._is_my_gang or self._is_others_draw:
            self._env._action_space = ['Pass']

        # 不是上家打的不能吃，海底牌也不能吃
        if self._env._current_card_from != self._env._next_player(self._my_id, -1) or self._env._is_wall_last:
            self._env._action_space = list(filter(lambda x: not x.startswith('Chi'), self._env._action_space))
        
        # 环境发的和自己打的不能碰杠，海底牌也不能碰杠
        if self._env._current_card_from in {None, self._my_id} or self._env._is_wall_last:
            self._env._action_space = list(filter(lambda x: re.match(r'Peng|Gang', x) is None, self._env._action_space))
        
        # 自己牌墙没牌了不能杠牌
        if self._wall_remains[self._my_id] == 0:
            self._env._action_space = list(filter(lambda x: 'Gang' not in x, self._env._action_space))

    # 我方动作空间
    @property
    def action_space(self) -> List[ChineseStandardMahjongEnv.ActionNameType]:
        return self._env.action_space 

    # 初始的13张手牌
    @property
    def _my_initial_hand_card(self) -> Tuple[str]:
        return self._env._initial_hand_cards[self._my_id]
    
    # 修改我方手牌
    def _add_to_my_hand_card_counter(self, card:ChineseStandardMahjongEnv.CardNameType, n:int=1):
        self._env._hand_card_counters[self._my_id][card] += n

    # 处理成功补杠和打出未被吃碰杠的动作
    def _process_successful_bugang_and_play(self) -> None:
        
        # 如果没有打牌或者补杠操作
        if len(self._env._unprocessed_actions) == 0:
            return
        
        active_player = self._env._active_player
        # 现在对之前打牌/补杠的玩家的状态进行操作
        self._env._active_player = self._env._current_card_from
        # 成功打牌，未被吃碰杠的情况
        if self._env._unprocessed_actions[0].startswith('Play'):
            # 加入牌河，暴露为明牌
            self._env._add_discard_history(self._env._current_card)
            # 在之前打出这张牌(Play)的时候已经删除过我方手牌，现在无需考虑我方手牌的变化情况

        # 补杠成功，未被抢杠和的情况
        elif self._env._unprocessed_actions[0].startswith('BuGang'):
            # 查找碰牌的副露
            index = self._env._peng_pack_index_of(self._env._current_card)
            # 删去碰牌的副露
            del self._env._shown_packs[self._env._current_card_from][index]
            # 添加补杠的副露
            self._env._add_shown_pack(f'BuGang{self._env._current_card}', card_from=None)
            # 修改手牌数量
            self._n_hand_cards[self._env._active_player] -= 1
            # 如果是我方补杠，需要维护我的手牌
            if self._env._active_player == self._my_id:
                _, buganged_card = self._env.action_to_tuple(self._env._unprocessed_actions[0])
                self._add_to_my_hand_card_counter(buganged_card, -1)

            self._env._set_current_card_and_source(None, None)
        self._env._unprocessed_actions.clear()
        # 回到当前玩家
        self._env._active_player = active_player

    # 将Botzone格式的request加载入内置环境
    def _load_botzone_request_line(self, request:str) -> None:
        self._is_my_gang = False
        self._is_others_draw = False
        request_parts = request.split()
        
        # 设置门风圈风
        if int(request_parts[0]) == 0:
            seat_wind, prevalent_wind = map(int, request_parts[1:])
            self._my_seat_wind = seat_wind
            self._env.prevalent_wind = 1 + prevalent_wind
            self._env.seat_winds = tuple(1 + (prevalent_wind + i) % self._env.n_players for i in range(self._env.n_players))
            self._my_id = (seat_wind - prevalent_wind) % self._env.n_players
            self._env._active_player = self._my_id
            self._env._action_space = ['Pass']
            return

        # 发初始手牌
        if int(request_parts[0]) == 1:
            flower_count = request_parts[1:1+self._env.n_players]
            cards = tuple(request_parts[1+self._env.n_players:])
            self._env._initial_hand_cards = tuple(tuple() if i != self._my_id else cards for i in range(self._env.n_players))
            self._env._hand_card_counters = tuple(Counter() if i != self._my_id else Counter(cards) for i in range(self._env.n_players))
            self._env._action_space = ['Pass']
            return

        # 自己摸牌
        if int(request_parts[0]) == 2:
            self._env._active_player = self._my_id
            # 看看有没有成功的补杠和打牌待处理
            
            self._process_successful_bugang_and_play()
            # 牌墙牌数目减少
            self._wall_remains[self._my_id] -= 1
            # 手牌数目增加
            self._n_hand_cards[self._my_id] += 1      
            # 摸的牌
            card = request_parts[1]
            # 海底牌标记
            if self._wall_remains[self._env._next_player(self._my_id)] == 0:
                self._env._is_wall_last = True
            # 手牌暂不修改，摸到的牌仍然放在self._env._current_card里，直至打出一张牌/补杠/暗杠时才将摸到的牌添加到自己的手牌中
            # 因为这张可能就是听的牌，需要传入算番器
            self._env._set_current_card_and_source(card, None)
            return

        # 各个玩家动作
        if int(request_parts[0]) == 3:
            # 做这个动作的玩家，按照ChineseStandardMahjongEnv的id表示
            player = (int(request_parts[1]) + 1 - self._env.prevalent_wind) % self._env.n_players
            # 可能为DRAW/PLAY/CHI/PENG/GANG/BUGANG
            action_type = request_parts[2]
            
            # 其他玩家摸牌（自己摸牌的情况在2中已经处理掉了）
            if action_type == 'DRAW':
                self._is_others_draw = True
                self._env._active_player = player
                # 看看有没有成功的补杠和打牌待处理
                self._process_successful_bugang_and_play()
                # 牌墙牌数目减少
                self._wall_remains[player] -= 1
                # 玩家的手牌数目直接增加
                self._n_hand_cards[player] += 1
                # 海底牌标记
                if self._wall_remains[self._env._next_player(player)] == 0:
                    self._env._is_wall_last = True
                # 标记为摸牌后阶段，但是不知道摸的是什么牌
                self._env._set_current_card_and_source(None, None)
                return
            
            # 一旦没有杠上开花/抢杠和，标记失效
            self._env._is_about_kong = False
            # 某个玩家打出一张牌（包括自己）
            if action_type == 'PLAY':
                self._env._active_player = player
                # 维护手牌数量
                card = request_parts[3]
                self._n_hand_cards[player] -= 1
                # 如果是我方打牌，还需要维护我方手牌
                if player == self._my_id:
                    # 如果自己之前摸了牌，还要把摸的牌加入自己的手牌
                    if self._env._current_card is not None and self._env._current_card_from is None:
                        self._add_to_my_hand_card_counter(self._env._current_card)
                    self._add_to_my_hand_card_counter(card, -1)
                
                self._env._unprocessed_actions.clear()
                # 当前牌为打出的牌
                self._env._set_current_card_and_source(card, player)
                self._env._unprocessed_actions.append(f'Play{card}')
                return
            
            # 某个玩家吃牌，并且顺手打出一张
            if action_type == 'CHI':
                self._env._active_player = player
                chi_central_card, played_card = request_parts[3:]
                # 吃牌需要添加该玩家副露
                self._env._add_shown_pack(f'Chi{chi_central_card}', card_from=self._env._current_card_from)
                # 如果是自己吃牌，还需要维护自己的手牌
                if player == self._my_id:
                    # 吃进去1张牌
                    self._add_to_my_hand_card_counter(self._env._current_card)
                    # 打出去1张牌
                    self._add_to_my_hand_card_counter(played_card, -1)
                    # 变成副露的牌
                    for i in range(-1, -1+self._env._chi_tile_length):
                        self._add_to_my_hand_card_counter(self._env.card_name(self._env.card_id(chi_central_card)+i), -1)

                # 待决策的牌转为打出的牌
                self._env._set_current_card_and_source(played_card, player)
                self._env._unprocessed_actions.clear()
                self._env._unprocessed_actions.append(f'Play{played_card}')
                # 手牌数目增减
                self._n_hand_cards[player] -= self._env._chi_tile_length
                return

            # 某个玩家碰牌，并且顺手打出一张
            if action_type == 'PENG':
                self._env._active_player = player
                played_card = request_parts[3]
                # 碰牌需要添加该玩家副露
                self._env._add_shown_pack(f'Peng{self._env._current_card}', card_from=self._env._current_card_from)
                # 如果是我碰的牌
                if player == self._my_id:
                    # 碰进去1张牌，变成副露3张牌
                    self._add_to_my_hand_card_counter(self._env._current_card, 1 - self._env._peng_tile_length)
                    # 打出来1张牌
                    self._add_to_my_hand_card_counter(played_card, -1)
                # 待决策的牌转为打出的牌
                self._env._set_current_card_and_source(played_card, player)
                self._env._unprocessed_actions.clear()
                self._env._unprocessed_actions.append(f'Play{played_card}')
                # 手牌数目增减
                self._n_hand_cards[player] -= self._env._peng_tile_length
                return
            
            # 某个玩家杠上一张打出来的牌/从环境摸来的牌
            if action_type == 'GANG':
                self._env._is_about_kong = True
                self._env._active_player = player
                # 如果杠的上一回合是从环境摸牌，那么就是暗杠
                if self._env._current_card_from is None and self._env._current_card is not None:
                    # 如果是我的暗杠，需要从self._last_anganged_card读取暗杠的是哪一张牌
                    if player == self._my_id:
                        self._env._add_hidden_pack(f'AnGang{self._last_anganged_card}')
                        # 把摸的牌放进手牌中
                        self._add_to_my_hand_card_counter(self._env._current_card)
                        # 把暗杠的牌删去
                        self._add_to_my_hand_card_counter(self._last_anganged_card, -self._env._gang_tile_length)
                    
                    # 暗杠个数增加
                    self._n_hidden_packs[player] += 1
                    # 修改手牌数目：摸进1张，杠掉4张
                    self._n_hand_cards[player] -= self._env._gang_tile_length - 1
                    self._env._set_current_card_and_source(None, None)
                    self._is_my_gang = True

                # 如果杠的上一回合是别人打牌，那么就是明杠
                else:
                    # 如果是我的明杠
                    if player == self._my_id:
                        # 修改我的手牌，杠掉了3张
                        self._add_to_my_hand_card_counter(self._env._current_card, 1-self._env._gang_tile_length)
                    # 添加副露
                    self._env._add_shown_pack(f'Gang{self._env._current_card}', card_from=self._env._current_card_from)
                    # 修改手牌数目：摸进1张，杠掉4张
                    self._n_hand_cards[player] -= self._env._gang_tile_length - 1
                    self._env._set_current_card_and_source(None, None)
                    self._is_my_gang = True
                self._env._unprocessed_actions.clear()
                # 下一个动作一定是摸牌
                return
            
            if action_type == 'BUGANG':
                self._env._is_about_kong = True
                self._env._active_player = player
                buganged_card = request_parts[3]
                
                # 如果补杠的人是我，需要维护我的手牌
                if player == self._my_id:
                    # 先把环境摸来的牌加入手牌中，不然会信息丢失
                    self._add_to_my_hand_card_counter(self._env._current_card)
                    self._is_my_gang = True
                
                self._n_hand_cards[player] += 1
                self._env._unprocessed_actions.clear()
                self._env._unprocessed_actions.append(f'BuGang{buganged_card}')
                self._env._set_current_card_and_source(buganged_card, player)

    # 从request_loader（默认为input）中加载botzone的request序列，使用agent生成符合botzone格式的response并输出到sys.stdout
    # agent.select_action接收observation为参数，输出observation['action_space']中的某个动作
    def load_botzone_request_and_generate_response(self, agent:Agent, request_loader:Callable) -> None:
        
        while True:
            request = request_loader().strip()
            if not request or len(request.split()) == 1:
                continue

            # 确定风圈、发初始手牌
            if request.startswith('0') or request.startswith('1'):
                self._load_botzone_request_line(request)
                print('PASS')
            
            # 行牌过程request
            else:
                self._load_botzone_request_line(request)
                self._update_action_space_and_fan()
                print(self._generate_botzone_response(agent))
            
            # Botzone长时运行标记
            print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
            sys.stdout.flush()
        
    # 如果动作选择了暗杠，会将动作记录在self._last_anganged_card中，以备后续加载状态
    def _generate_botzone_response(self, agent:Agent) -> str:

        action = agent.select_action(self.observation)

        action_type, card = self._env.action_to_tuple(action)

        if action_type in {'Pass', 'Hu'}:
            return action.upper()

        if action_type == 'Play':
            return f'PLAY {card}'
        
        if action_type == 'AnGang':
            self._last_anganged_card = card
            return f'GANG {card}'

        if action_type == 'Gang':
            return 'GANG'

        if action_type == 'BuGang':
            return f'BUGANG {card}'
        
        if action_type == 'Chi':
            # 复制一份环境，假装吃牌成功了
            new_adapter = deepcopy(self)
            new_adapter._env._active_player = self._my_id
            new_adapter._env._add_shown_pack(action, card_from=new_adapter._env._current_card_from)
            # 修改手牌
            new_adapter._add_to_my_hand_card_counter(new_adapter._env._current_card)
            for i in range(-1, -1+new_adapter._env._chi_tile_length):
                new_adapter._add_to_my_hand_card_counter(new_adapter._env.card_name(new_adapter._env.card_id(card)+i), -1)
            new_adapter._env._set_current_card_and_source(None, None)
            new_adapter._n_hand_cards[new_adapter._my_id] -= new_adapter._env._chi_tile_length - 1
            new_adapter._env._unprocessed_actions.clear()
            new_adapter._update_action_space_and_fan()
            # 生成新的observation再调用agent决策吃完打什么牌
            play_action = agent.select_action(new_adapter.observation)
            play, played_card = new_adapter._env.action_to_tuple(play_action)
            assert play == 'Play'
            return f'CHI {card} {played_card}'
        
        if action_type == 'Peng':
            # 复制一份环境，假装碰牌成功了
            new_adapter = deepcopy(self)
            new_adapter._env._active_player = self._my_id
            new_adapter._env._add_shown_pack(action, card_from=new_adapter._env._current_card_from)
            new_adapter._add_to_my_hand_card_counter(new_adapter._env._current_card, 1-new_adapter._env._peng_tile_length)
            new_adapter._env._set_current_card_and_source(None, None)
            new_adapter._n_hand_cards[new_adapter._my_id] -= new_adapter._env._chi_tile_length - 1
            new_adapter._env._unprocessed_actions.clear()
            new_adapter._update_action_space_and_fan()
            # 生成新的observation再调用agent决策碰完打什么牌
            play_action = agent.select_action(new_adapter.observation)
            play, played_card = new_adapter._env.action_to_tuple(play_action)
            assert play == 'Play'
            return f'PENG {played_card}'