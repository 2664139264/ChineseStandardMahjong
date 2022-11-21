from random_mahjong_agent import RandomMahjongAgent
from chinese_standard_mahjong_botzone_adapter import ChineseStandardMahjongBotzoneAdapter

if __name__ == '__main__':
    agent = RandomMahjongAgent()
    adapter = ChineseStandardMahjongBotzoneAdapter()
    adapter.load_botzone_request_and_generate_response(agent, input)