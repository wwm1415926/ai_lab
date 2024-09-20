import pygame
import numpy as np
import sys


board_cnt = 13
cell_size = 40
margin = 20
bottom_margin = 30


window_size = board_cnt * cell_size + 2 * margin + bottom_margin


pygame.init()
pygame.font.init()
screen = pygame.display.set_mode((window_size, window_size))
pygame.display.set_caption("Gomoku")
font = pygame.font.SysFont("Arial", 36)

# Piece definitions
Real_board = np.zeros((board_cnt, board_cnt), dtype=int)
move_history = []

def place_stone(board, row, col, color):
    if board[row, col] == 0:
        board[row, col] = color
        move_history.append((row, col, color))
        return True
    else:
        return False

def undo_move():
    if len(move_history) > 0:
        last_move = move_history.pop()
        Real_board[last_move[0], last_move[1]] = 0
        if last_move[2] == 2 and len(move_history) > 0:
            last_move = move_history.pop()
            Real_board[last_move[0], last_move[1]] = 0

def human_step(board):
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    row, col = get_click_position(event.pos)
                    if row is not None and col is not None:
                        if place_stone(board, row, col, 1):
                            return row, col
                        else:
                            show_message("Not Valid!")
                elif event.button == 3:
                    undo_move()
                    return None, None

def show_message(message):
    text = font.render(f"{message}", True, (255, 0, 0))
    text_rect = text.get_rect(center=(window_size // 2, window_size - 20))
    screen.blit(text, text_rect)
    pygame.display.update()
    pygame.time.delay(1000)

def check_win(board, row, col, color):
    def check_direction(delta_row, delta_col):
        count = 1
        for i in range(1, 5):
            r, c = row + i * delta_row, col + i * delta_col
            if 0 <= r < board_cnt and 0 <= c < board_cnt and board[r, c] == color:
                count += 1
            else:
                break
        for i in range(1, 5):
            r, c = row - i * delta_row, col - i * delta_col
            if 0 <= r < board_cnt and 0 <= c < board_cnt and board[r, c] == color:
                count += 1
            else:
                break
        return count >= 5

    return (check_direction(1, 0) or
            check_direction(0, 1) or
            check_direction(1, 1) or
            check_direction(1, -1))

def evaluate(board):
    total_score = 0

    def get_line(board, start_row, start_col, delta_row, delta_col):
        line = ''
        for i in range(5):
            r = start_row + i * delta_row
            c = start_col + i * delta_col
            if 0 <= r < board_cnt and 0 <= c < board_cnt:
                if board[r][c] == 0:
                    line += '0'
                elif board[r][c] == 1:
                    line += 'B'
                else:  # WHITE
                    line += 'W'
            else:
                line += 'N'
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

    for i in range(board_cnt):
        for j in range(board_cnt):
            for delta_row, delta_col in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                line = get_line(board, i, j, delta_row, delta_col)
                for pattern, score in patterns:
                    if pattern in line:
                        total_score += score
    return total_score

def generate_moves(board):
    moves = []
    next_size = 2
    occupied = np.argwhere(board != 0)
    if len(occupied) == 0:
        return [(board_cnt // 2, board_cnt // 2)]
    potential_moves = set()
    for (i, j) in occupied:
        for x in range(max(0, i - next_size), min(board_cnt, i + next_size + 1)):
            for y in range(max(0, j - next_size), min(board_cnt, j + next_size + 1)):
                if board[x, y] == 0:  # EMPTY
                    potential_moves.add((x, y))
    return list(potential_moves)

def minimax(board, depth, alpha, beta, maximizingPlayer):
    if depth == 0:
        return evaluate(board), None


    for color in [2, 1]:
        positions = np.argwhere(board == color)
        for pos in positions:
            if check_win(board, pos[0], pos[1], color):
                if color == 2:  # WHITE
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
            new_board[move[0], move[1]] = 2
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
            new_board[move[0], move[1]] = 1  # BLACK
            eval, _ = minimax(new_board, depth - 1, alpha, beta, True)
            if eval < minEval:
                minEval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return minEval, best_move

def ai_step(board, depth):
    show_message("AI is thinking...")
    _, move = minimax(board, depth, -np.inf, np.inf, True)
    if move is not None:
        place_stone(board, move[0], move[1], 2)  # WHITE
        return move[0], move[1]
    else:
        return None, None

def draw_board():
    screen.fill((185, 122, 87))  # BG_COLOR directly set
    for i in range(board_cnt + 1):
        pygame.draw.line(screen, (0, 0, 0),  # GRID_COLOR directly set
                         (margin + i * cell_size, margin),
                         (margin + i * cell_size, margin + board_cnt * cell_size), 1)
        pygame.draw.line(screen, (0, 0, 0),
                         (margin, margin + i * cell_size),
                         (margin + board_cnt * cell_size, margin + i * cell_size), 1)

    # Draw the undo button
    undo_button = pygame.Rect(window_size - 100, window_size - 50, 80, 40)  # Button dimensions
    pygame.draw.rect(screen, (0, 0, 0), undo_button)  # Draw button
    text = font.render('Undo', True, (255, 255, 255))
    screen.blit(text, (window_size - 90, window_size - 45))  # Draw "Undo" text on the button

def draw_stones():
    for r in range(board_cnt):
        for c in range(board_cnt):
            if Real_board[r, c] == 1:  # BLACK
                pygame.draw.circle(screen, (0, 0, 0),  # BLACK_COLOR directly set
                                   (margin + c * cell_size + cell_size // 2, margin + r * cell_size + cell_size // 2), cell_size // 2.5)
            elif Real_board[r, c] == 2:  # WHITE
                pygame.draw.circle(screen, (255, 255, 255),  # WHITE_COLOR directly set
                                   (margin + c * cell_size + cell_size // 2, margin + r * cell_size + cell_size // 2), cell_size // 2.5)

def get_click_position(pos):
    x, y = pos
    if window_size - 100 <= x <= window_size - 20 and window_size - 50 <= y <= window_size:
        undo_move()
        draw_board()
        draw_stones()
        pygame.display.update()
        return None, None
    if margin <= x <= margin + board_cnt * cell_size and margin <= y <= margin + board_cnt * cell_size:
        row = (y - margin) // cell_size
        col = (x - margin) // cell_size
        return row, col
    return None, None

def game_loop():
    step = 0
    total_steps = board_cnt * board_cnt
    while step < total_steps:
        draw_board()
        draw_stones()
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                get_click_position(event.pos)
                continue

        # User move
        player_x, player_y = human_step(Real_board)
        if player_x is None and player_y is None:
            continue
        draw_stones()
        pygame.display.update()
        if check_win(Real_board, player_x, player_y, 1):
            show_message("Human Win!")
            break

        # AI move
        ai_x, ai_y = ai_step(Real_board, depth=2)
        if ai_x is not None and ai_y is not None:
            draw_stones()
            pygame.display.update()
            if check_win(Real_board, ai_x, ai_y, 2):
                show_message("AI Win!")
                break
        else:
            show_message("Draw!")
            break
        step += 1

game_loop()
