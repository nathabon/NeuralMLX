import pygame
import mlx.core as mx
import numpy as np
import sys
import neuralNetwork2 as nn

# -------------------------------------------------------
# Config
# -------------------------------------------------------
CELL      = 20          # taille d'une cellule en pixels
GRID      = 28          # 28x28
PANEL     = 320         # largeur panneau de droite
MARGIN    = 20
W         = GRID * CELL + PANEL
H         = GRID * CELL
FPS       = 60

BG        = (18, 18, 24)
GRID_COL  = (35, 35, 45)
DRAW_COL  = (255, 255, 255)
PANEL_BG  = (24, 24, 32)
ACCENT    = (100, 200, 255)
BAR_COL   = (100, 200, 255)
BAR_BG    = (40, 40, 55)
TEXT_COL  = (200, 200, 220)
DIM_COL   = (80, 80, 100)

LABELS = ["0","1","2","3","4","5","6","7","8","9"]




def load_network():
    """Charge le réseau — adapte ce chemin à ton projet."""
    try:
        return nn.NeuralNetwork.fromFileH5("saves/mnist/train_full_conv_h_2.h5")
    except Exception as e:
        print(f"Réseau non chargé : {e}")
        return None


def grid_to_mlx(grid: np.ndarray) -> mx.array:
    """Convertit la grille numpy (28,28) en vecteur MLX (28,28, 1)."""
    flat = grid.flatten().astype(np.float32) / 255.0
    return mx.array(flat).reshape(1, 28, 28, 1)
 
 
def predict(net, grid: np.ndarray):
    if net is None:
        return np.ones(10) / 10, np.ones(10) / 10
    x = grid_to_mlx(grid)
    out = net(x)
    logits = np.array(out.tolist()).flatten()
    # Normalisation simple au lieu de softmax
    total = logits.sum()
    probs = logits / total if total > 0 else np.ones(10) / 10
    return logits, probs
 
 
def draw_panel(surface, logits, probs, font_big, font_sm):
    ox = GRID * CELL
    pygame.draw.rect(surface, PANEL_BG, (ox, 0, PANEL, H))
 
    # Titre
    title = font_big.render("Prédiction", True, ACCENT)
    surface.blit(title, (ox + MARGIN, MARGIN))
 
    # Meilleure prédiction
    best = int(np.argmax(probs))
    conf = probs[best]
    big = pygame.font.SysFont("monospace", 72, bold=True)
    digit_surf = big.render(str(best), True, DRAW_COL)
    surface.blit(digit_surf, (ox + PANEL // 2 - digit_surf.get_width() // 2, 55))
 
    conf_surf = font_sm.render(f"{conf*100:.1f}%", True, ACCENT)
    surface.blit(conf_surf, (ox + PANEL // 2 - conf_surf.get_width() // 2, 138))
 
    # Séparateur
    pygame.draw.line(surface, GRID_COL, (ox + MARGIN, 165), (ox + PANEL - MARGIN, 165), 1)
 
    # En-têtes colonnes
    hdr_val = font_sm.render("valeur", True, DIM_COL)
    hdr_pct = font_sm.render("  %  ", True, DIM_COL)
    surface.blit(hdr_val, (ox + PANEL - 130, 170))
    surface.blit(hdr_pct, (ox + PANEL - 55,  170))
 
    # Barres + valeurs brutes pour chaque chiffre
    bar_x     = ox + MARGIN + 22
    bar_w_max = PANEL - MARGIN * 2 - 22 - 140
    bar_h     = 15
    spacing   = (H - 190) // 10
 
    for i, (label, p, v) in enumerate(zip(LABELS, probs, logits)):
        y = 190 + i * spacing
        col = ACCENT if i == best else DIM_COL
 
        # Label chiffre
        lbl = font_sm.render(label, True, col)
        surface.blit(lbl, (ox + MARGIN, y + 1))
 
        # Barre fond
        pygame.draw.rect(surface, BAR_BG, (bar_x, y, bar_w_max, bar_h), border_radius=3)
 
        # Barre remplie
        fill_w = int(p * bar_w_max)
        if fill_w > 0:
            bar_color = ACCENT if i == best else (60, 120, 170)
            pygame.draw.rect(surface, bar_color, (bar_x, y, fill_w, bar_h), border_radius=3)
 
        # Valeur brute (sortie réseau)
        val_str = f"{v:+.3f}"
        val_col = (100, 220, 120) if v >= 0 else (220, 100, 100)
        val_surf = font_sm.render(val_str, True, val_col if i == best else DIM_COL)
        surface.blit(val_surf, (ox + PANEL - 130, y + 1))
 
        # Pourcentage
        pct_surf = font_sm.render(f"{p*100:4.0f}%", True, col)
        surface.blit(pct_surf, (ox + PANEL - 50, y + 1))
 
 
def draw_grid(surface, grid: np.ndarray):
    for row in range(GRID):
        for col in range(GRID):
            val = grid[row, col]
            color = (val, val, val)
            pygame.draw.rect(surface, color,
                             (col * CELL, row * CELL, CELL, CELL))
            if val < 200:
                pygame.draw.rect(surface, GRID_COL,
                                 (col * CELL, row * CELL, CELL, CELL), 1)
 
 
def paint(grid, mx_pos, brush=2, val=255):
    """Peint avec un pinceau carré centré sur mx_pos."""
    cx = mx_pos[0] // CELL
    cy = mx_pos[1] // CELL
    for dy in range(-brush, brush + 1):
        for dx in range(-brush, brush + 1):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < GRID and 0 <= ny < GRID:
                # grid[ny, nx] = min(255, grid[ny, nx] + val // ((abs(dx) + abs(dy) + 1)))
                grid[ny, nx] = 200
 
 
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("MNIST Draw — Réseau de neurones")
    clock  = pygame.time.Clock()
 
    font_big = pygame.font.SysFont("monospace", 18, bold=True)
    font_sm  = pygame.font.SysFont("monospace", 13)
    hint_fnt = pygame.font.SysFont("monospace", 11)
 
    grid  = np.zeros((GRID, GRID), dtype=np.uint8)
    net = load_network()
    probs  = np.ones(10) / 10
    logits = np.zeros(10)
 
    drawing   = False
    erasing   = False
    dirty     = False   # grille modifiée depuis dernière prédiction
    pred_tick = 0       # frame du dernier predict
 
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
 
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    grid[:] = 0
                    probs  = np.ones(10) / 10
                    logits = np.zeros(10)
                    dirty  = False
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
 
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: drawing = True
                if event.button == 3: erasing = True
 
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1: drawing = False
                if event.button == 3: erasing = False
 
        # Dessin / gomme
        if drawing or erasing:
            mx_pos = pygame.mouse.get_pos()
            if mx_pos[0] < GRID * CELL:
                if drawing:
                    paint(grid, mx_pos, brush=1, val=255)
                else:
                    paint(grid, mx_pos, brush=2, val=-255)
                    grid = np.clip(grid, 0, 255).astype(np.uint8)
                dirty = True
 
        # Prédiction toutes les 10 frames si la grille a changé
        frame = pygame.time.get_ticks()
        if dirty and frame - pred_tick > 150:
            logits, probs = predict(net, grid)
            pred_tick = frame
            dirty = False
 
        # Rendu
        screen.fill(BG)
        draw_grid(screen, grid)
        draw_panel(screen, logits, probs, font_big, font_sm)
 
        # Hints
        hint = hint_fnt.render("Clic gauche : dessiner  |  Clic droit : gomme  |  C : effacer", True, DIM_COL)
        screen.blit(hint, (MARGIN, H - 18))
 
        pygame.display.flip()
        clock.tick(FPS)
 
 
if __name__ == "__main__":
    main()
 