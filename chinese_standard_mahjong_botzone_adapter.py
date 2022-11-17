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
        self._env = ChineseStandardMahjongEnv(config=dict())
        self._my_id = None
        self._wall_remains = [len(self._env._card_names) // self._env.n_players - self._env._n_hand_card for _ in range(self._env.n_players)]
        self._n_hand_cards = [self._env._n_hand_card for _ in range(self._env.n_players)]
        self._n_hidden_packs = [0 for _ in range(self._env.n_players)]

    # 我方观测信息
    @property
    def observation(self) -> ChineseStandardMahjongEnv.ObservationType:
        pass
        ## return self._env.observation
    
    @property
    def action_space(self) -> List[ChineseStandardMahjongEnv.ActionNameType]:
        pass

    # 初始13张手牌
    @property
    def _my_initial_hand_card(self) -> Tuple[str]:
        return self._env._initial_hand_cards[self._my_id]
    
    # 我方当前手牌
    @property
    def _my_hand_card_counter(self) -> Counter[str]:
        return self._env._hand_card_counters[self._my_id]

    # 修改我方手牌
    def _add_to_my_hand_card_counter(self, card:ChineseStandardMahjongEnv.CardNameType, n:int=1):
        self._env._hand_card_counters[self._my_id][card] += n

    # 处理成功补杠和打出未被吃碰杠的动作
    def _process_successful_bugang_and_play(self) -> None:
        
        if len(self._env._unprocessed_actions) == 0:
            return
        active_player = self._env._active_player
        # 现在对之前打牌/补杠的玩家的状态进行操作
        self._env._active_player = self._env._current_card_from
        # 成功打牌，未被吃碰杠的情况
        if self._env._unprocessed_actions[0].startswith('Play'):
            # 加入牌河，暴露为明牌
            self._env._add_discard_history(self._env._current_card)
            # 在之前打出这张牌(Play)的时候已经维护过我方手牌，现在无需考虑我方手牌的变化情况

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
                self._env._hand_card_counters[self._my_id][buganged_card] -= 1
        
        self._env._unprocessed_actions.clear()
        # 回到当前玩家
        self._env._active_player = active_player

    # 将Botzone格式的request加载入内置环境，返回
    def load_botzone_request(self, request:str) -> None:
        components = request.split()

        # 交互第一行，区分长时运行或非长时运行模式
        if len(components) == 1:
            return
        
        # 设置门风圈风
        if int(components[0]) == 0:
            prevalent_wind, seat_wind = map(int, components[1:])
            self._env.prevalent_wind = 1 + prevalent_wind
            self._env.seat_winds = tuple(1 + prevalent_wind + i for i in range(self._env.n_players))
            self._my_id = (seat_wind - prevalent_wind) % self._env.n_players
            self._env._active_player = self._my_id
            return

        # 发初始手牌
        if int(components[0]) == 1:
            flower_count = components[1:1+self._env.n_players]
            cards = tuple(components[1+self._env.n_players:])
            self._env._initial_hand_cards = tuple(tuple() if i != self._my_id else cards for i in range(self._env.n_players))
            self._env._hand_card_counters[self._my_id] = Counter(cards)
            return

        # 自己摸牌
        if int(components[0]) == 2:
            # 看看有没有补杠和打牌待处理
            self._process_successful_bugang_and_play()

            self._wall_remains[self._my_id] -= 1
            self._env._active_player = self._my_id
            # 摸出一张牌
            card = components[1]
            self._env._set_current_card_and_from(card, None)
            return

        # 各个玩家动作
        if int(components[0]) == 3:
            # 做这个动作的玩家
            player = (int(components[1]) - self._env.prevalent_wind) % self._env.n_players
            # 可能为DRAW/PLAY/CHI/PENG/GANG/BUGANG
            action_type = components[2]
            
            # 其他玩家摸牌（自己摸牌的情况在2中已经处理掉了）
            if action_type == 'DRAW':
                # 看看有没有补杠和打牌待处理
                self._env._active_player = player
                self._process_successful_bugang_and_play()
                # 牌墙牌数目减少
                self._wall_remains[player] -= 1
                # 手牌数目增加
                self._n_hand_cards[player] += 1
                # 标记为摸牌后阶段
                self._env._current_card_from = None
                return
            
            self._env._is_about_kong = False
            # 某个玩家打出一张牌
            if action_type == 'PLAY':
                # 维护手牌数量
                card = components[3]
                self._n_hand_cards[player] -= 1
                # 如果是我方打牌，还需要维护我方手牌
                if player == self._my_id:
                    self._add_to_my_hand_card_counter(card, -1)
                self._env._active_player = player
                # 当前牌为打出的牌
                self._env._set_current_card_and_source(card, player)
                self._env._unprocessed_actions.append(f'Play{card}')
                return
            
            # 某个玩家吃牌，并且顺手打出一张
            if action_type == 'CHI':
                chi_central_card, played_card = components[3:]
                self._env._active_player = player
                self._env._add_shown_pack(f'Chi{chi_central_card}', card_from=self._env._current_card_from)
                
                # 如果是自己吃牌，还需要维护自己的手牌
                if player == self._my_id:
                    self._add_to_my_hand_card_counter(self._env._current_card, 1)
                    self._add_to_my_hand_card_counter(played_card, -1)
                    for i in range(-1, -1+self._env._chi_tile_length):
                        self._add_to_my_hand_card_counter(self._card_name(self._env._card_id(chi_central_card)+i), -1)

                self._env._set_current_card_and_source(played_card, player)
                self._n_hand_cards[player] -= self._env._chi_tile_length

                return

            # 某个玩家碰牌，并且顺手打出一张
            if action_type == 'PENG':
                played_card = components[3]
                self._env._active_player = player
                if player == self._my_id:
                    self._add_to_my_hand_card_counter(self._env._current_card, 1 - self._env._peng_tile_length)
                    self._add_to_my_hand_card_counter(played_card, -1)

                self._env._add_shown_pack(f'Peng{self._env._current_card}', card_from=self._env._current_card_from)
                self._env._set_current_card_and_source(played_card, player)
                self._n_hand_cards[player] -= self._env._peng_tile_length
                return
            
            # 某个玩家杠上一张打出来的牌/从环境摸来的牌
            if action_type == 'GANG':
                self._env._is_about_kong = True
                self._env._active_player = player
                # 如果杠的上一回合是从环境摸牌，那么就是暗杠
                if self._env._current_card_from is None and self._env._current_card is not None:
                    # 如果是我的暗杠
                    if player == self._my_id:
                        self._env._add_hidden_pack(self._env._current_card)
                        self._add_to_my_hand_card_counter(self._env._current_card, 1)
                        self._

                    self._n_hidden_packs[player] += 1
                    self._n_hand_cards[player] -= self._env._gang_tile_length - 1

                # 如果杠的上一回合是别人打牌，那么就是明杠
                else:
                    self._env._add_shown_pack(f'Gang{self._env._current_card}', card_from=self._env._current_card_from)
                    self._n_hand_cards[player] -= self._env._gang_tile_length - 1
                
                # 下一个动作一定是摸牌
                return
            
            if action_type == 'BUGANG':
                self._env._is_about_kong = True
                self._env._active_player = player





            

    
    def load_botzone_requests(self, requests:List[str]) -> None:
        pass
        
    def action_to_botzone_response(self, agent:Agent) -> str:
        pass



if __name__ == '__main__':
    adapter = ChineseStandardMahjongBotzoneAdapter()


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
1
0 2 1


1 0 0 0 0 T6 B5 B7 B8 W8 T1 B7 W3 W6 B4 J2 T1 W5
3 0 DRAW
3 0 PLAY F2
3 1 DRAW
3 1 PLAY T7
2 T2
3 2 PLAY J2
3 3 DRAW
3 3 PLAY J3
3 0 DRAW
3 0 PLAY J3
3 1 DRAW
3 1 PLAY B8
2 T6
3 2 PLAY W3
3 3 DRAW
3 3 PLAY B2
3 0 DRAW
3 0 PLAY F1
3 1 DRAW
3 1 PLAY B9
3 2 CHI B8 W8
3 3 DRAW
3 3 PLAY B9
3 0 DRAW
3 0 PLAY J1
3 1 DRAW
3 1 PLAY W6
2 T7
3 2 PLAY T6
3 3 DRAW
3 3 PLAY J1
3 0 DRAW
3 0 PLAY F1
3 1 DRAW
3 1 PLAY F2
2 B1
3 2 PLAY B1
3 3 DRAW
3 3 PLAY T4
3 0 CHI T3 B2
3 1 CHI B2 J2
2 T8
3 2 PLAY T2
3 3 DRAW
3 3 PLAY W8
3 0 DRAW
3 0 PLAY T9
3 1 DRAW
3 1 PLAY W3
2 W5
3 2 PLAY W5
3 3 CHI W5 W2
3 0 DRAW
3 0 PLAY B1
3 1 DRAW
3 1 PLAY T6
2 W1
3 2 PLAY W1
3 3 DRAW
3 3 PLAY T8
3 0 DRAW
3 0 PLAY T8
3 1 DRAW
3 1 PLAY B8
2 B4
3 2 PLAY B4
3 3 DRAW
3 3 PLAY F2
3 0 DRAW
3 0 PLAY J2
3 1 DRAW
3 1 PLAY W7
3 2 CHI W6 B7
3 3 DRAW
3 3 PLAY W1
3 0 DRAW
3 0 PLAY B6
'''
