import pygame
import numpy as np
import sys




# 常量
BOARD_SIZE = 15
CELL_SIZE = 40  # 每个单元格的大小
MARGIN = 20     # 棋盘边缘的距离
WINDOW_SIZE = BOARD_SIZE * CELL_SIZE + 2 * MARGIN  # 窗口大小

# 颜色定义
BLACK_COLOR = (0, 0, 0)
WHITE_COLOR = (255, 255, 255)
GRID_COLOR = (0, 0, 0)
BG_COLOR = (185, 122, 87)

# 棋子定义
EMPTY, BLACK, WHITE = 0, 1, 2
Real_board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)

# 初始化 pygame
pygame.init()
pygame.font.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("五子棋")
font = pygame.font.SysFont("Arial", 36)
def place_stone(board, row, col, color):
    if board[row, col] == EMPTY:
        board[row, col] = color
        return True
    else:
        return False

def user_move(board):
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # 检测鼠标左键点击
                row, col = get_click_position(event.pos)  # 获取鼠标点击位置对应的行和列
                if row is not None and col is not None:
                    if place_stone(board, row, col, BLACK):  # 在棋盘上放置棋子
                        return row,col
                    else:

                        show_message("Not Valid!")

def show_message(message):
    text = font.render(f"{message} ", True, (255, 0, 0))
    text_rect = text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2))
    screen.blit(text, text_rect)
    pygame.display.update()
    pygame.time.delay(1000)

def check_win(board, row, col, color):
    def check_direction(delta_row, delta_col):
        count = 1
        for i in range(1, 5):
            r, c = row + i * delta_row, col + i * delta_col
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == color:
                count += 1
            else:
                break
        for i in range(1, 5):
            r, c = row - i * delta_row, col - i * delta_col
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == color:
                count += 1
            else:
                break
        return count >= 5

    # 检查四个方向：横、竖、斜
    return (check_direction(1, 0) or  # 水平方向
            check_direction(0, 1) or  # 垂直方向
            check_direction(1, 1) or  # 正斜方向
            check_direction(1, -1))   # 反斜方向

class Node:
    def __init__(self, node=None,posX=0, posY=0):
        self.value =-np.inf
        self.depth = 0
        self.father = None
        self.children = []
        self.posX = 0
        self.posY = 0
        self.is_max=True
        self.board =np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)

        if node is not None:
            if node.is_max:
                self.is_max=False
                self.value =-np.inf
            else:
                self.is_max=True
                self.value=np.inf
            self.depth =node.depth + 1
            self.father =node
            self.posX = posX
            self.posY = posY
            self.board = node.board.copy()
            if self.is_max:
                self.board[self.posX][self.posY] =WHITE
            else:
                self.board[self.posX][self.posY] =BLACK

    def evaluate(self,board):
        def evaluate_black(s):
            patterns = [
                "B0000", "0B000", "00B00", "000B0", "0000B",
                "BB000", "0BB00", "00BB0", "000BB", "B0B00", "0B0B0", "00B0B", "B00B0", "0B00B", "B000B",
                "BBB00", "0BBB0", "00BBB", "BB0B0", "0BB0B", "B0BB0", "0B0BB", "BB00B", "B00BB", "B0B0B",
                "BBBB0", "BBB0B", "BB0BB", "B0BBB", "0BBBB", "BBBBB",
            ]
            scores = [
                1, 1, 1, 1, 1,
                10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                10000, 10000, 10000, 10000, 10000, 1000000,
            ]
            for i in range(31):
                if s == patterns[i]:

                    return scores[i]
            return 0

        def evaluate_white(s):
            patterns = [
                "W0000", "0W000", "00W00", "000W0", "0000W",
                "WW000", "0WW00", "00WW0", "000WW", "W0W00", "0W0W0", "00W0W", "W00W0", "0W00W", "W000W",
                "WWW00", "0WWW0", "00WWW", "WW0W0", "0WW0W", "W0WW0", "0W0WW", "WW00W", "W00WW", "W0W0W",
                "WWWW0", "WWW0W", "WW0WW", "W0WWW", "0WWWW", "WWWW",
            ]
            scores = [
                1, 1, 1, 1, 1,10,
                10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                1000, 2000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
                100000, 100000, 100000, 100000, 100000, 10000000,
            ]
            for i in range(31):
                if s == patterns[i]:
                    print(scores[i])
                    return scores[i]
            return 0

        def convert(pos):
            if pos == 0:
                return "0"
            elif pos == BLACK:
                return "B"
            else:
                return "W"
        self.value = 0
        for i in range(15):
            for j in range(15):
                if j + 4 < 15:
                    s = ''
                    for k in range(5):
                        s +=convert(board[i][j + k])
                    self.value += evaluate_black(s) - evaluate_white(s)
                if i + 4 < 15:
                    s = ''
                    for k in range(5):
                        s += convert(board[i+k][j])
                self.value += evaluate_black(s) - evaluate_white(s)
                if i + 4 < 15 and j + 4 < 15:
                    s = ''
                    for k in range(5):
                        s += convert(board[i+k][j + k])
                self.value += evaluate_black(s) - evaluate_white(s)
                if i + 4 < 15 and j - 4 >= 0:
                    s = ''
                    for k in range(5):
                        s += convert(board[i+k][j-k])
                self.value += evaluate_black(s) - evaluate_white(s)
        print("the whole value",self.value)

class GameTree:
    def __init__(self, maxDepth, next_size,
                  board= None):
        self.next_size=next_size
        self.maxDepth = maxDepth
        self.nodeRoot = Node()
        self.nodeNext = None
        self.openTable=[]
        self.closedTable=[]
        self.nodeRoot.board = board

    def get_search_nodes(self, node):
        empty=True
        newBoard =np.zeros((BOARD_SIZE,BOARD_SIZE),dtype=int)
        for i in range(15):
            for j in range(15):
                if node.board[i,j] == 0:
                    continue
                empty=False
                x1 = max(0, i - self.next_size)
                x2 = min(14, i + self.next_size)
                y1 = max(0, j - self.next_size)
                y2 = min(14, j + self.next_size)
                for x in range(x1, x2 + 1):
                    for y in range(y1, y2 + 1):
                        if node.board[x,y] == 0:
                            newBoard[x,y] = 1

        allNodes = []
        if empty==True:
            allNodes.append((7, 7))
        else:
            for i in range(15):
                for j in range(15):
                    if newBoard[i,j]:
                        allNodes.append((i, j))

        return allNodes

    def expand_children_nodes(self, node):
        expandNodes= self.get_search_nodes(node)
        for pos in expandNodes:
            n = Node(node, pos[0], pos[1])
            node.children.append(n)
            self.openTable.append(n)
        return len(expandNodes)

    def is_alpha_beta_cut(self, node):
        if node is None or node.father is None:
            return False
        if node.is_max and node.value >= node.father.value:
            return True
        if not node.is_max and node.value <= node.father.value:
            return True
        return self.is_alpha_beta_cut(node.father)

    def update_value_from_node(self, node):
        if node is None:
            return
        if not node.children:
            self.update_value_from_node(node.father)
            return
        if node.is_max:
            cnt = -np.inf
            for n in node.children:
                if n.value != np.inf:
                    cnt = max(cnt, n.value)
            if cnt > node.value:
                node.value = cnt
                self.update_value_from_node(node.father)
        else:
            cnt = np.inf
            for n in node.children:
                if n.value != -np.inf:
                    cnt = min(cnt, n.value)
            if cnt < node.value:
                node.value = cnt
                self.update_value_from_node(node.father)

    def set_next_pos(self):
        #找到根节点的子结点中值最大的，并将其更新
        self.nodeNext = self.nodeRoot.children[0] if self.nodeRoot.children else None
        for n in self.nodeRoot.children:
            if n.value > self.nodeNext.value:
                self.nodeNext = n

    def minmax(self):
        self.openTable.append(self.nodeRoot)
        while self.openTable:
            node = self.openTable.pop(0)
            self.closedTable.append(node)
            if self.is_alpha_beta_cut(node.father):
                continue
            if node.depth < self.maxDepth:
                numExpand = self.expand_children_nodes(node)
                if numExpand != 0:
                    continue
            node.evaluate(node.board)
            self.update_value_from_node(node)
        self.set_next_pos()
        return 0

    def get_next_pos(self):
        if self.nodeNext is None:
            return (255, 255)
        else:
            return (self.nodeNext.posX, self.nodeNext.posY)


def make_move(board, move, color):
    new_board = np.copy(board)
    new_board[move[0], move[1]] = color
    return new_board

def ai_move(board,pos):
    place_stone(board, pos[0], pos[1], WHITE)
    return pos[0], pos[1]


def draw_board():
    screen.fill(BG_COLOR)
    for i in range(BOARD_SIZE+1 ):
        pygame.draw.line(screen, GRID_COLOR,
                         (MARGIN + i * CELL_SIZE, MARGIN),
                         (MARGIN + i * CELL_SIZE, WINDOW_SIZE - MARGIN), 1)
        pygame.draw.line(screen, GRID_COLOR,
                         (MARGIN, MARGIN + i * CELL_SIZE),
                         (WINDOW_SIZE - MARGIN, MARGIN + i * CELL_SIZE), 1)

# 棋子绘制
def draw_stones():
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if Real_board[r, c] == BLACK:
                pygame.draw.circle(screen, BLACK_COLOR,
                                   (MARGIN + c * CELL_SIZE+0.5*CELL_SIZE, MARGIN + r * CELL_SIZE+0.5*CELL_SIZE), CELL_SIZE //(2.5) )
            elif Real_board[r, c] == WHITE:
                pygame.draw.circle(screen, WHITE_COLOR,
                                   (MARGIN + c * CELL_SIZE+0.5*CELL_SIZE, MARGIN + r * CELL_SIZE+0.5*CELL_SIZE), CELL_SIZE //(2.5)  )

# 获取点击位置
def get_click_position(pos):
    x, y = pos
    if MARGIN <= x <= WINDOW_SIZE - MARGIN and MARGIN <= y <= WINDOW_SIZE - MARGIN:
        row = (y - MARGIN) // CELL_SIZE
        col = (x - MARGIN) // CELL_SIZE
        return row, col
    return None, None

# 用户回合
def handle_user_move():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return "quit"
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            row, col = get_click_position(event.pos)
            if row is not None and col is not None:
                if Real_board[row, col] == EMPTY:
                    place_stone(Real_board, row, col, BLACK)
                    if check_win(Real_board, row, col, BLACK):
                        return "player_win"
                    return "continue"
    return "continue"



player_x,player_y=8,8
# 主游戏循环
def game_loop():
    total_step=BOARD_SIZE*BOARD_SIZE
    step=0
    while step<total_step:
        draw_board()
        draw_stones()
        pygame.display.update()
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                pygame.quit()
                sys.exit()
                # 用户下棋
        global player_x,player_y
        player_x,player_y=user_move(Real_board)
        draw_stones()
        pygame.display.update()
        if check_win(Real_board,player_x,player_y,BLACK):
            show_message("Human Win!")
        temp_board=Real_board.copy()
        ai=GameTree(1,4,temp_board)
        ai.minmax()
        print(ai.get_next_pos()[0],ai.get_next_pos()[1])
        ai_x,ai_y=ai_move(Real_board,ai.get_next_pos())
        draw_stones()
        pygame.display.update()
        if check_win(Real_board, ai_x,ai_y,WHITE):
            show_message("AI Win！")
        step+=1
game_loop()
