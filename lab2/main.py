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
            if node.is_max==True:
                self.is_max=False
                self.value =-np.inf
            else:
                self.is_max=True
                self.value=np.inf
            self.depth =node.depth + 1
            self.father = node
            self.posX = posX
            self.posY = posY
            self.board = node.board.copy()
            if self.is_max:
                self.board[self.posX][self.posY] =WHITE
            else:
                self.board[self.posX][self.posY] =BLACK


    def evaluate(self):
        score=0
        for i in range(3,BOARD_SIZE-3):
            for j in range(3,BOARD_SIZE-3):
                if(j<=10):
                    pattern1 = ''
                    for k in range(5):
                        if self.board[i, j + k] == 1:
                            pattern1+="B"
                        elif self.board[i, j + k]==2:
                            pattern1+="W"
                        else:
                            pattern1+="O"
                    black_count = pattern1.count('B')
                    white_count = pattern1.count('W')
                    empty_count = pattern1.count('O')
                    print(black_count,white_count,empty_count)
                    # （1）同时含有黑子和白子，得0分
                    if black_count > 0 and white_count > 0:
                        score -= 0
                    # （2）含有1个黑子和4个空点，得+1分
                    elif black_count == 1 and empty_count == 4:
                        score -= 1
                    # （3）含有2个黑子和3个空点，得+10分
                    elif black_count == 2 and empty_count == 3:
                        score -= 10
                    # （4）含有3个黑子和2个空点，得+100分
                    elif black_count == 3 and empty_count == 2:
                        score -= 100
                    # （5）含有4个黑子和1个空点，得+10000分
                    elif black_count == 4 and empty_count == 1:
                        print("got line 141")
                        score -= 100000
                    # （6）含有5个黑子，得+1000000分
                    if black_count==5:
                        score-=1000000
                    # （7）含有1个白子和4个空点，得-1分
                    elif white_count == 1 and empty_count == 4:
                        score += 1
                    # （8）含有2个白子和3个空点，得-10分
                    elif white_count == 2 and empty_count == 3:
                        score += 10
                    # （9）形如“0WWW0”，得-2000分
                    elif white_count == 3 and empty_count == 2 and pattern1[0] == 'O' and pattern1[-1] == 'O':
                        score +=2000
                    # （10）含有3个白子和2个空点，得-1000分
                    elif white_count == 3 and empty_count == 2:
                        score +=1000
                    # （11）含有4个白子和1个空点，得-100000分
                    elif white_count == 4 and empty_count == 1:

                        score +=100000
                    # （12）含有5个白子，得-10000000分
                    if white_count==5:
                        score+=10000000
                if(i<=10):
                    pattern2 = ''
                    for k in range(5):
                        if self.board[i+k, j] == BLACK:
                            pattern2+="B"
                        elif self.board[i+k, j] == WHITE:
                            pattern2+="W"
                        else:
                            pattern2+="O"
                    black_count = pattern2.count('B')
                    white_count = pattern2.count('W')
                    empty_count = pattern2.count('O')
                    # （1）同时含有黑子和白子，得0分
                    if black_count > 0 and white_count > 0:
                        score -= 0
                    # （2）含有1个黑子和4个空点，得+1分
                    elif black_count == 1 and empty_count == 4:

                        score -= 1
                    # （3）含有2个黑子和3个空点，得+10分
                    elif black_count == 2 and empty_count == 3:
                        score -= 10
                    # （4）含有3个黑子和2个空点，得+100分
                    elif black_count == 3 and empty_count == 2:
                        score -= 100
                    # （5）含有4个黑子和1个空点，得+10000分
                    elif black_count == 4 and empty_count == 1:
                        score -= 10000
                    # （6）含有5个黑子，得+1000000分
                    if black_count == 5:
                        score -= 1000000
                    # （7）含有1个白子和4个空点，得-1分
                    elif white_count == 1 and empty_count == 4:
                        score += 1
                    # （8）含有2个白子和3个空点，得-10分
                    elif white_count == 2 and empty_count == 3:
                        score += 10
                    # （9）形如“0WWW0”，得-2000分
                    elif white_count == 3 and empty_count == 2 and pattern1[0] == 'O' and pattern1[-1] == 'O':
                        score += 2000
                    # （10）含有3个白子和2个空点，得-1000分
                    elif white_count == 3 and empty_count == 2:
                        score += 1000
                    # （11）含有4个白子和1个空点，得-100000分
                    elif white_count == 4 and empty_count == 1:

                        score += 100000
                    # （12）含有5个白子，得-10000000分
                    if white_count == 5:
                        score += 10000000

                if  j<=10 and i <= 10:
                    pattern3 = ''
                    for k in range(5):
                        if self.board[i+k, j + k] == BLACK:
                            pattern3+="B"
                        elif self.board[i+k, j + k] == WHITE:
                            pattern3+="B"
                        else:
                            pattern3+="O"
                    black_count = pattern3.count('B')
                    white_count = pattern3.count('W')
                    empty_count = pattern3.count('O')
                    # （1）同时含有黑子和白子，得0分
                    if black_count > 0 and white_count > 0:
                        score -= 0
                    # （2）含有1个黑子和4个空点，得+1分
                    elif black_count == 1 and empty_count == 4:

                        score -= 1
                    # （3）含有2个黑子和3个空点，得+10分
                    elif black_count == 2 and empty_count == 3:
                        score -= 10
                    # （4）含有3个黑子和2个空点，得+100分
                    elif black_count == 3 and empty_count == 2:
                        score -= 100
                    # （5）含有4个黑子和1个空点，得+10000分
                    elif black_count == 4 and empty_count == 1:
                        score -= 10000
                    # （6）含有5个黑子，得+1000000分
                    if black_count == 5:
                        score -= 1000000
                    # （7）含有1个白子和4个空点，得-1分
                    elif white_count == 1 and empty_count == 4:
                        score += 1
                    # （8）含有2个白子和3个空点，得-10分
                    elif white_count == 2 and empty_count == 3:
                        score += 10
                    # （9）形如“0WWW0”，得-2000分
                    elif white_count == 3 and empty_count == 2 and pattern1[0] == 'O' and pattern1[-1] == 'O':
                        score += 2000
                    # （10）含有3个白子和2个空点，得-1000分
                    elif white_count == 3 and empty_count == 2:
                        score += 1000
                    # （11）含有4个白子和1个空点，得-100000分
                    elif white_count == 4 and empty_count == 1:

                        score += 100000
                    # （12）含有5个白子，得-10000000分
                    if white_count == 5:
                        score += 10000000


                if i>=5 and j<=10:
                    pattern4 = ''
                    for k in range(5):
                        if self.board[i - k, j + k] == BLACK:
                            pattern4+="B"
                        elif self.board[i - k, j + k] == WHITE:
                            pattern4+="W"
                        else:
                            pattern4+="O"
                    black_count = pattern4.count('B')
                    white_count = pattern4.count('W')
                    empty_count = pattern4.count('O')
                    # （1）同时含有黑子和白子，得0分
                    if black_count > 0 and white_count > 0:
                        score -= 0
                    # （2）含有1个黑子和4个空点，得+1分
                    elif black_count == 1 and empty_count == 4:

                        score -= 1
                    # （3）含有2个黑子和3个空点，得+10分
                    elif black_count == 2 and empty_count == 3:
                        score -= 10
                    # （4）含有3个黑子和2个空点，得+100分
                    elif black_count == 3 and empty_count == 2:
                        score -= 100
                    # （5）含有4个黑子和1个空点，得+10000分
                    elif black_count == 4 and empty_count == 1:
                        score -= 10000
                    # （6）含有5个黑子，得+1000000分
                    if black_count == 5:
                        score -= 1000000
                    # （7）含有1个白子和4个空点，得-1分
                    elif white_count == 1 and empty_count == 4:
                        score += 1
                    # （8）含有2个白子和3个空点，得-10分
                    elif white_count == 2 and empty_count == 3:
                        score += 10
                    # （9）形如“0WWW0”，得-2000分
                    elif white_count == 3 and empty_count == 2 and pattern1[0] == 'O' and pattern1[-1] == 'O':
                        score += 2000
                    # （10）含有3个白子和2个空点，得-1000分
                    elif white_count == 3 and empty_count == 2:
                        score += 1000
                    # （11）含有4个白子和1个空点，得-100000分
                    elif white_count == 4 and empty_count == 1:

                        score += 100000
                    # （12）含有5个白子，得-10000000分
                    if white_count == 5:
                        score += 10000000
        return score



class GameTree:
    def __init__(self, maxDepth, next_size,
                  board= None):
        self.next_size=next_size
        self.maxDepth = maxDepth
        self.nodeRoot = Node()
        self.nodeNext = None
        self.openTable=[]
        self.closedTable=[]

        if board is not None:
            self.nodeRoot.board = board

    def get_search_nodes(self, node):
        enpty=True
        newBoard =np.zeros((BOARD_SIZE,BOARD_SIZE))
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
        if node.is_max and node.value > node.father.value:
            return True
        if not node.is_max and node.value < node.father.value:
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
            node.evaluate()
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
        ai=GameTree(2,2,temp_board)
        ai.minmax()
        print(ai.get_next_pos()[0],ai.get_next_pos()[1])
        ai_x,ai_y=ai_move(Real_board,ai.get_next_pos())
        draw_stones()
        pygame.display.update()
        if check_win(Real_board, ai_x,ai_y,WHITE):
            show_message("AI Win！")
        step+=1

game_loop()
