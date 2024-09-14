import pygame
import sys
import math

# 常量设置
BOARD_SIZE = 15  # 棋盘大小15x15
GRID_SIZE = 40  # 每个格子的大小
SCREEN_SIZE = BOARD_SIZE * GRID_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LINE_COLOR = (0, 0, 0)
HUMAN = 1
AI = -1
EMPTY = 0

# 初始化pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("五子棋 - 人机对战")

# 初始化棋盘
board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]


def draw_board():
    screen.fill(WHITE)
    for i in range(BOARD_SIZE):
        pygame.draw.line(screen, LINE_COLOR, (GRID_SIZE // 2, GRID_SIZE // 2 + i * GRID_SIZE),
                         (SCREEN_SIZE - GRID_SIZE // 2, GRID_SIZE // 2 + i * GRID_SIZE))
        pygame.draw.line(screen, LINE_COLOR, (GRID_SIZE // 2 + i * GRID_SIZE, GRID_SIZE // 2),
                         (GRID_SIZE // 2 + i * GRID_SIZE, SCREEN_SIZE - GRID_SIZE // 2))


def draw_pieces():
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board[y][x] == HUMAN:
                pygame.draw.circle(screen, BLACK, (x * GRID_SIZE + GRID_SIZE // 2, y * GRID_SIZE + GRID_SIZE // 2),
                                   GRID_SIZE // 2 - 2)
            elif board[y][x] == AI:
                pygame.draw.circle(screen, WHITE, (x * GRID_SIZE + GRID_SIZE // 2, y * GRID_SIZE + GRID_SIZE // 2),
                                   GRID_SIZE // 2 - 2)
                pygame.draw.circle(screen, BLACK, (x * GRID_SIZE + GRID_SIZE // 2, y * GRID_SIZE + GRID_SIZE // 2),
                                   GRID_SIZE // 2 - 2, 2)


def is_valid_move(x, y):
    return board[y][x] == EMPTY


def is_full():
    for row in board:
        if EMPTY in row:
            return False
    return True


def check_winner(player):
    # 检查所有的行、列和对角线
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if check_line(x, y, 1, 0, player) or check_line(x, y, 0, 1, player) or \
                    check_line(x, y, 1, 1, player) or check_line(x, y, 1, -1, player):
                return True
    return False


def check_line(x, y, dx, dy, player):
    count = 0
    for i in range(5):
        nx = x + i * dx
        ny = y + i * dy
        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and board[ny][nx] == player:
            count += 1
        else:
            break
    return count == 5


# 使用 alpha-beta 剪枝的 minimax 算法
def minimax(depth, alpha, beta, is_maximizing):
    if check_winner(AI):
        return 1000
    if check_winner(HUMAN):
        return -1000
    if is_full() or depth == 0:
        return 0

    if is_maximizing:
        max_eval = -math.inf
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if is_valid_move(x, y):
                    board[y][x] = AI
                    eval = minimax(depth - 1, alpha, beta, False)
                    board[y][x] = EMPTY
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
        return max_eval
    else:
        min_eval = math.inf
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if is_valid_move(x, y):
                    board[y][x] = HUMAN
                    eval = minimax(depth - 1, alpha, beta, True)
                    board[y][x] = EMPTY
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
        return min_eval


def ai_move():
    best_score = -math.inf
    best_move = None
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if is_valid_move(x, y):
                board[y][x] = AI
                score = minimax(3, -math.inf, math.inf, False)  # 深度3，使用alpha-beta剪枝
                board[y][x] = EMPTY
                if score > best_score:
                    best_score = score
                    best_move = (x, y)
    if best_move:
        board[best_move[1]][best_move[0]] = AI


def human_move(x, y):
    if is_valid_move(x, y):
        board[y][x] = HUMAN
        return True
    return False


def main():
    draw_board()
    draw_pieces()
    pygame.display.flip()
    turn = HUMAN

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and turn == HUMAN:
                x, y = event.pos
                grid_x = x // GRID_SIZE
                grid_y = y // GRID_SIZE
                if human_move(grid_x, grid_y):
                    if check_winner(HUMAN):
                        print("你赢了!")
                        pygame.quit()
                        sys.exit()
                    turn = AI

        if turn == AI:
            ai_move()
            if check_winner(AI):
                print("AI赢了!")
                pygame.quit()
                sys.exit()
            turn = HUMAN

        draw_board()
        draw_pieces()
        pygame.display.flip()


if __name__ == "__main__":
    main()
