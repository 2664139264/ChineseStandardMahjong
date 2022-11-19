import re
from copy import deepcopy
from typing import List, Tuple, Counter
from collections import Counter

from agent import Agent
from botzone_adapter import BotzoneAdapter
from chinese_standard_mahjong_env import ChineseStandardMahjongEnv

# 适配Botzone简单交互，交互格式参考https://wiki.botzone.org.cn/index.php?title=Chinese-Standard-Mahjong
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
        # 上一次决策的玩家是谁
        self._last_player_id = None
        # 初始化当前牌张
        self._env._set_current_card_and_source(None, None)

    # 我方观测信息
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
        print('Unprocessed_actions: ', self._env._unprocessed_actions)
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
        request_parts = request.split()
        
        # 设置门风圈风
        if int(request_parts[0]) == 0:
            seat_wind, prevalent_wind = map(int, request_parts[1:])
            self._my_seat_wind = seat_wind
            self._env.prevalent_wind = 1 + prevalent_wind
            self._env.seat_winds = tuple(1 + (prevalent_wind + i) % self._env.n_players for i in range(self._env.n_players))
            self._my_id = (seat_wind - prevalent_wind) % self._env.n_players
            self._env._active_player = self._my_id
            return

        # 发初始手牌
        if int(request_parts[0]) == 1:
            flower_count = request_parts[1:1+self._env.n_players]
            cards = tuple(request_parts[1+self._env.n_players:])
            self._env._initial_hand_cards = tuple(tuple() if i != self._my_id else cards for i in range(self._env.n_players))
            self._env._hand_card_counters = tuple(Counter() if i != self._my_id else Counter(cards) for i in range(self._env.n_players))
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
                print('draw: ', player)
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
                self._env._unprocessed_actions.append(f'Play{played_card}')
                # 手牌数目增减
                self._n_hand_cards[player] -= self._env._chi_tile_length
                self._env._unprocessed_actions.clear()
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
                self._env._unprocessed_actions.append(f'Play{played_card}')
                # 手牌数目增减
                self._n_hand_cards[player] -= self._env._peng_tile_length
                self._env._unprocessed_actions.clear()
                return
            
            # 某个玩家杠上一张打出来的牌/从环境摸来的牌
            if action_type == 'GANG':
                self._env._is_about_kong = True
                self._env._active_player = player
                # 如果杠的上一回合是从环境摸牌，那么就是暗杠
                if self._env._current_card_from is None and self._env._current_card is not None:
                    # 如果是我的暗杠，需要从self._last_anganged_card读取暗杠的是哪一张牌
                    if player == self._my_id:
                        self._env._add_hidden_pack(self._last_anganged_card)
                        # 把摸的牌放进手牌中
                        self._add_to_my_hand_card_counter(self._env._current_card)
                        # 把暗杠的牌删去
                        self._add_to_my_hand_card_counter(self._last_anganged_card, -self._env._gang_tile_length)
                    
                    # 暗杠个数增加
                    self._n_hidden_packs[player] += 1
                    # 修改手牌数目：摸进1张，杠掉4张
                    self._n_hand_cards[player] -= self._env._gang_tile_length - 1
                    self._env._set_current_card_and_source(None, None)

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
                
                self._n_hand_cards[player] += 1
                self._env._unprocessed_actions.append(f'BuGang{buganged_card}')
                self._env._set_current_card_and_source(buganged_card, player)

    def load_botzone_request(self, requests:List[str]) -> None:
        requests = list(filter(lambda line: len(line) > 0, map(lambda line: line.strip(), requests)))
        
        round_to_load = int(requests[0])
        requests = requests[1:]

        responses = [r for r in requests if re.match(r'\d', r) is None]
        requests = [r for r in requests if re.match(r'\d', r) is not None]
        # 采用长时运行模式，需要记录处理了多少条记录
        if len(responses) == 0:
            for request in requests[self._processed_history_length:]:
                self._load_botzone_request_line(request)
                self._last_player_id = self._env._active_player
            self._processed_history_length = len(request)
        
        # 没有采用长时运行模式，需要从头处理
        else:
            assert len(requests) == len(responses) + 1
            for i in range(len(responses)):
                request, response = requests[i], responses[i]
                print('***', request, '*****', response, '***')
                self._load_botzone_request_line(request)
                # test
                # 测试
                self._env._update_action_space_and_fan()
                
                # GANG有明杠和暗杠两种情况，如果是暗杠，需要把动作保存到self._last_anganged_card以备处理
                if response.startswith('GANG'):
                    response_parts = response.split()
                    # 如果是暗杠，需要考虑杠的是哪一张牌
                    if len(response_parts) > 1:
                        self._last_anganged_card = response_parts[-1]
                
                print('Action_space: ', self.action_space)
                print('\tactive_player: ', self._env._active_player, '\tlast_active: ', self._last_player_id)
                print('\tmy_id: ', self._my_id, '\tmy_wind: ', self._my_seat_wind)
                print('\twall_remain: ', self._wall_remains)
                
                print('\tmy_hand_card: ', self._env._hand_card_counters[self._my_id], self._env._current_card, self._env._current_card_from)
                print('\tdiscard_history: ', self._env._discard_histories)
                print('\tshown_packs: ', self._env._shown_packs)
                print()
                self._last_player_id = self._env._active_player
            self._load_botzone_request_line(requests[-1])
            self._env._update_action_space_and_fan()

    
    def generate_botzone_response(self, agent:Agent) -> str:
        
        pass



if __name__ == '__main__':
    adapter = ChineseStandardMahjongBotzoneAdapter()
    with open('log.txt', 'r', encoding='utf-8') as log_file:
        lines = log_file.readlines()
        
        adapter.load_botzone_request(lines)
        
        print(adapter._wall_remains)
        print(adapter._my_id)
        print(adapter.observation)
        print(adapter.action_space)

'''
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
'''

'''
110
0 1 3
PASS
1 0 0 0 0 W7 F4 B8 B5 W3 B5 J1 F1 B8 T1 B3 T5 W4
PASS
3 0 DRAW
PASS
3 0 PLAY T1
PASS
2 F1
PLAY F4
3 1 PLAY F4
PASS
3 2 DRAW
PASS
3 2 PLAY F2
PASS
3 3 DRAW
PASS
3 3 PLAY J1
PASS
3 0 DRAW
PASS
3 0 PLAY T2
PASS
2 W9
PLAY J1
3 1 PLAY J1
PASS
3 2 DRAW
PASS
3 2 PLAY F4
PASS
3 3 DRAW
PASS
3 3 PLAY F3
PASS
3 0 PENG T6
PASS
2 W1
PLAY T1
3 1 PLAY T1
PASS
3 2 DRAW
PASS
3 2 PLAY J3
PASS
3 3 DRAW
PASS
3 3 PLAY T7
PASS
3 0 DRAW
PASS
3 0 PLAY T5
PASS
2 B4
PLAY W1
3 1 PLAY W1
PASS
3 2 DRAW
PASS
3 2 PLAY B5
PASS
3 3 DRAW
PASS
3 3 PLAY W4
PASS
3 0 DRAW
PASS
3 0 PLAY T5
PASS
2 W5
PLAY W9
3 1 PLAY W9
PASS
3 2 DRAW
PASS
3 2 PLAY J2
PASS
3 3 DRAW
PASS
3 3 PLAY W3
PASS
3 0 CHI W3 J1
PASS
2 T8
PLAY T8
3 1 PLAY T8
PASS
3 2 DRAW
PASS
3 2 PLAY B9
PASS
3 3 DRAW
PASS
3 3 PLAY B6
PASS
3 0 DRAW
PASS
3 0 PLAY F2
PASS
2 W9
PLAY W9
3 1 PLAY W9
PASS
3 2 DRAW
PASS
3 2 PLAY T1
PASS
3 3 DRAW
PASS
3 3 PLAY T6
PASS
3 0 DRAW
PASS
3 0 PLAY T9
PASS
2 F2
PLAY F2
3 1 PLAY F2
PASS
3 2 DRAW
PASS
3 2 PLAY T2
PASS
3 3 DRAW
PASS
3 3 PLAY J1
PASS
3 0 DRAW
PASS
3 0 PLAY J3
PASS
2 F3
PLAY F3
3 1 PLAY F3
PASS
3 2 DRAW
PASS
3 2 PLAY T8
PASS
3 3 DRAW
PASS
3 3 PLAY T3
PASS
3 0 DRAW
PASS
3 0 PLAY W9
PASS
2 B4
PLAY F1
3 1 PLAY F1
PASS
3 2 DRAW
PASS
3 2 PLAY F4
PASS
3 3 DRAW
PASS
3 3 PLAY T3
PASS
3 0 DRAW
PASS
3 0 PLAY B8
PASS
2 B9
PLAY F1
3 1 PLAY F1
PASS
3 2 DRAW
PASS
3 2 PLAY W1
PASS
3 3 DRAW
PASS
3 3 PLAY J2
PASS
3 0 PENG W6
PASS
2 T4
PLAY B9
3 1 PLAY B9
PASS
3 2 DRAW
PASS
3 2 PLAY B7
PASS
3 3 DRAW
PASS
3 3 PLAY T2
PASS
3 0 DRAW
PASS
3 0 PLAY J3
PASS
2 F1
PLAY F1
3 1 PLAY F1
PASS
3 2 DRAW
PASS
3 2 PLAY T3
PASS
3 3 DRAW
PASS
3 3 PLAY B9
PASS
3 0 DRAW
PASS
3 0 PLAY B3
CHI B4 W7
3 3 PENG T5
PASS
3 0 DRAW
PASS
3 0 PLAY B1
PASS
2 T9
PLAY T9
3 1 PLAY T9





'''