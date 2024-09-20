import pygame
import numpy as np
import sys

# Constants
BOARD_SIZE = 15
CELL_SIZE = 40  # Size of each cell
MARGIN = 20     # Margin around the board
WINDOW_SIZE = BOARD_SIZE * CELL_SIZE + 2 * MARGIN  # Window size

# Color definitions
BLACK_COLOR = (0, 0, 0)
WHITE_COLOR = (255, 255, 255)
GRID_COLOR = (0, 0, 0)
BG_COLOR = (185, 122, 87)

# Piece definitions
EMPTY, BLACK, WHITE = 0, 1, 2
Real_board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)

# Initialize pygame
pygame.init()
pygame.font.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Gomoku")
font = pygame.font.SysFont("Arial", 36)

def place_stone(board, row, col, color):
    if board[row, col] == EMPTY:
        board[row, col] = color
        return True
    else:
        return False

def human_step(board):
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left mouse click
                row, col = get_click_position(event.pos)
                if row is not None and col is not None:
                    if place_stone(board, row, col, BLACK):
                        return row, col
                    else:
                        show_message("Not Valid!")

def show_message(message):
    text = font.render(f"{message}", True, (255, 0, 0))
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

    return (check_direction(1, 0) or  # Horizontal
            check_direction(0, 1) or  # Vertical
            check_direction(1, 1) or  # Diagonal
            check_direction(1, -1))   # Anti-diagonal

def evaluate(board):
    total_score = 0

    def get_line(board, start_row, start_col, delta_row, delta_col):
        line = ''
        for i in range(5):
            r = start_row + i * delta_row
            c = start_col + i * delta_col
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                if board[r][c] == EMPTY:
                    line += '0'
                elif board[r][c] == BLACK:
                    line += 'B'
                else:
                    line += 'W'
            else:
                line += 'N'  # N for None (out of bounds)
        return line

    patterns = [
        ("WWWWW", 1000000),
        ("0WWWW0", 100000),
        ("0WWWW", 10000),
        ("WWWW0", 10000),
        ("0WWW0", 1000),
        ("0WWW", 100),
        ("WWW0", 100),
        ("0WW0", 10),
        ("0WW", 10),
        ("WW0", 10),
        ("0W0", 1),
        ("0W", 1),
        ("W0", 1),
        ("BBBBB", -1000000),
        ("0BBBB0", -100000),
        ("0BBBB", -10000),
        ("BBBB0", -10000),
        ("0BBB0", -1000),
        ("0BBB", -100),
        ("BBB0", -100),
        ("0BB0", -10),
        ("0BB", -10),
        ("BB0", -10),
        ("0B0", -1),
        ("0B", -1),
        ("B0", -1),
    ]

    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            for delta_row, delta_col in [(0,1), (1,0), (1,1), (1,-1)]:
                line = get_line(board, i, j, delta_row, delta_col)
                for pattern, score in patterns:
                    if pattern in line:
                        total_score += score
    return total_score

def generate_moves(board):
    moves = []
    next_size = 2
    occupied = np.argwhere(board != EMPTY)
    if len(occupied) == 0:
        return [(BOARD_SIZE // 2, BOARD_SIZE // 2)]
    potential_moves = set()
    for (i, j) in occupied:
        for x in range(max(0, i - next_size), min(BOARD_SIZE, i + next_size + 1)):
            for y in range(max(0, j - next_size), min(BOARD_SIZE, j + next_size + 1)):
                if board[x, y] == EMPTY:
                    potential_moves.add((x, y))
    return list(potential_moves)

def minimax(board, depth, alpha, beta, maximizingPlayer):
    if depth == 0:
        return evaluate(board), None

    # Check for a win
    for color in [WHITE, BLACK]:
        positions = np.argwhere(board == color)
        for pos in positions:
            if check_win(board, pos[0], pos[1], color):
                if color == WHITE:
                    return np.inf, None
                else:
                    return -np.inf, None

    moves = generate_moves(board)
    if not moves:
        return evaluate(board), None
    best_move = None
    if maximizingPlayer:
        maxEval = -np.inf
        for move in moves:
            new_board = board.copy()
            new_board[move[0], move[1]] = WHITE
            eval, _ = minimax(new_board, depth - 1, alpha, beta, False)
            if eval > maxEval:
                maxEval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return maxEval, best_move
    else:
        minEval = np.inf
        for move in moves:
            new_board = board.copy()
            new_board[move[0], move[1]] = BLACK
            eval, _ = minimax(new_board, depth - 1, alpha, beta, True)
            if eval < minEval:
                minEval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return minEval, best_move

def ai_step(board, depth):
    _, move = minimax(board, depth, -np.inf, np.inf, True)
    if move is not None:
        place_stone(board, move[0], move[1], WHITE)
        return move[0], move[1]
    else:
        return None, None

def draw_board():
    screen.fill(BG_COLOR)
    for i in range(BOARD_SIZE + 1):
        pygame.draw.line(screen, GRID_COLOR,
                         (MARGIN + i * CELL_SIZE, MARGIN),
                         (MARGIN + i * CELL_SIZE, WINDOW_SIZE - MARGIN), 1)
        pygame.draw.line(screen, GRID_COLOR,
                         (MARGIN, MARGIN + i * CELL_SIZE),
                         (WINDOW_SIZE - MARGIN, MARGIN + i * CELL_SIZE), 1)

def draw_stones():
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if Real_board[r, c] == BLACK:
                pygame.draw.circle(screen, BLACK_COLOR,
                                   (MARGIN + c * CELL_SIZE + 0.5 * CELL_SIZE, MARGIN + r * CELL_SIZE + 0.5 * CELL_SIZE), CELL_SIZE // 2.5)
            elif Real_board[r, c] == WHITE:
                pygame.draw.circle(screen, WHITE_COLOR,
                                   (MARGIN + c * CELL_SIZE + 0.5 * CELL_SIZE, MARGIN + r * CELL_SIZE + 0.5 * CELL_SIZE), CELL_SIZE // 2.5)

def get_click_position(pos):
    x, y = pos
    if MARGIN <= x <= WINDOW_SIZE - MARGIN and MARGIN <= y <= WINDOW_SIZE - MARGIN:
        row = (y - MARGIN) // CELL_SIZE
        col = (x - MARGIN) // CELL_SIZE
        return row, col
    return None, None

def game_loop():
    total_step = BOARD_SIZE * BOARD_SIZE
    step = 0
    while step < total_step:
        draw_board()
        draw_stones()
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        # User move
        player_x, player_y = human_step(Real_board)
        draw_stones()
        pygame.display.update()
        if check_win(Real_board, player_x, player_y, BLACK):
            show_message("Human Win!")
            break
        # AI move
        ai_x, ai_y = ai_step(Real_board, depth=2)
        if ai_x is not None and ai_y is not None:
            draw_stones()
            pygame.display.update()
            if check_win(Real_board, ai_x, ai_y, WHITE):
                show_message("AI Win!")
                break
        else:
            show_message("Draw!")
            break
        step += 1

game_loop()
