from typing import List, Tuple, Counter
from collections import Counter
from agent import Agent
from chinese_standard_mahjong_env import ChineseStandardMahjongEnv

# 适配Botzone简单交互，交互格式参考https://wiki.botzone.org.cn/index.php?title=Chinese-Standard-Mahjong
class ChineseStandardMahjongBotzoneAdapter:

    def __init__(self):
        self.reset()
    
    def reset(self):
        self._env = ChineseStandardMahjongEnv(config=dict())
        self._my_id = None

    @property
    def observation(self) -> ChineseStandardMahjongEnv.ObservationType:
        return self._env.observation
    
    @property
    def _my_initial_hand_card(self) -> Tuple[str]:
        return self._env._initial_hand_cards[self._my_id]

    @property
    def _my_hand_card_counter(self) -> Counter[str]:
        return self._env._hand_card_counters[self._my_id]

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

        # 发初始手牌
        if int(components[0]) == 1:
            flower_count = components[1:1+self._env.n_players]
            cards = tuple(components[1+self._env.n_players:])
            self._env._initial_hand_cards = tuple(tuple() if i != self._my_id else cards for i in range(self._env.n_players))
            self._env._hand_card_counters[self._my_id] = Counter(cards)
            return

        # 自己摸牌
        if int(components[0]) == 2:
            card = components[1]
            self._env._set_current_card_and_from(card, None)
            return

        # 各玩家动作
        if int(components[0]) == 3:
            player = (int(components[1]) - self._env.prevalent_wind) % self._env.n_players

            action_type = components[2]
            
            # 其他玩家摸牌
            if action_type == 'DRAW':
                return
            
            if action_type == 'PLAY':
                card = components[3]
                self._env._set_current_card_and_source(card, player)


    
    def load_botzone_requests(self, requests:List[str]) -> None:
        pass
        
    def action_to_botzone_response(self, agent:Agent) -> str:
        pass



if __name__ == '__main__':
    adapter = ChineseStandardMahjongBotzoneAdapter()


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
