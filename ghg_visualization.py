"""
Greenhouse Gas Visualization (2001-2025)
----------------------------------------
Distributed visualization of NOAA greenhouse gas data (CO2, CH4, N2O)
rendered across a 2x2 multi-monitor MPI cluster using Python, mpi4py, and Pygame.
"""

import os
import random
import numpy as np
import pandas as pd
import pygame
from mpi4py import MPI


# ---------------------------------------------------------------------
# 1. LOAD NOAA DATA
# ---------------------------------------------------------------------
def load_gas_data(url, gas_name):
    """Load and process NOAA greenhouse gas dataset from CSV."""
    df = pd.read_csv(url, comment="#")
    df = df.rename(columns=lambda x: x.strip())
    df = df[["year", "average"]].dropna()
    df = df.groupby("year", as_index=False)["average"].mean()
    df = df.rename(columns={"average": f"{gas_name.upper()} avg"})
    return df


def load_noaa_data():
    """Fetch and merge NOAA greenhouse gas datasets into one DataFrame."""
    urls = {
        "co2": "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv",
        "ch4": "https://gml.noaa.gov/webdata/ccgg/trends/ch4/ch4_mm_gl.csv",
        "n2o": "https://gml.noaa.gov/webdata/ccgg/trends/n2o/n2o_mm_gl.csv"
    }
    co2 = load_gas_data(urls["co2"], "co2")
    ch4 = load_gas_data(urls["ch4"], "ch4")
    n2o = load_gas_data(urls["n2o"], "n2o")
    return co2.merge(ch4, on="year").merge(n2o, on="year")


def normalize_values(df, col, min_particles=20, max_particles=200):
    """Scale gas concentrations to particle counts for visualization."""
    vals = df[col]
    scaled = np.interp(vals, [vals.min(), vals.max()], [min_particles, max_particles])
    return scaled.astype(int)


# Load and normalize datasets
df = load_noaa_data()
df["CO2_particles"] = normalize_values(df, "CO2 avg")
df["CH4_particles"] = normalize_values(df, "CH4 avg")
df["N2O_particles"] = normalize_values(df, "N2O avg")


# ---------------------------------------------------------------------
# 2. MPI + DISPLAY SETUP
# ---------------------------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Local display setup
os.environ["DISPLAY"] = ":0.0"
os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"

pygame.display.init()
pygame.font.init()
pygame.mouse.set_visible(False)

WIDTH, HEIGHT = 1920, 1080
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.NOFRAME)
pygame.display.set_caption(f"GHG Visualization - Node {rank}")


# ---------------------------------------------------------------------
# 3. GLOBAL CANVAS + OFFSETS
# ---------------------------------------------------------------------
TOTAL_WIDTH = WIDTH * 2
TOTAL_HEIGHT = HEIGHT * 2
CENTER = (TOTAL_WIDTH // 2, TOTAL_HEIGHT // 2)

# Assign offsets for each node (2×2 grid)
if rank == 0:
    OFFSET_X, OFFSET_Y = 0, 0              # top-left
elif rank == 1:
    OFFSET_X, OFFSET_Y = WIDTH, 0           # top-right
elif rank == 2:
    OFFSET_X, OFFSET_Y = 0, HEIGHT          # bottom-left
elif rank == 3:
    OFFSET_X, OFFSET_Y = WIDTH, HEIGHT      # bottom-right
else:
    OFFSET_X, OFFSET_Y = 0, 0


# ---------------------------------------------------------------------
# 4. VISUAL CONSTANTS
# ---------------------------------------------------------------------
COLORS = {
    "co2": (255, 60, 60),     # red
    "ch4": (255, 165, 0),     # orange
    "n2o": (255, 255, 100)    # yellow
}
EARTH_COLOR = (70, 130, 180)
ATMOS_COLOR = (220, 220, 220)
BG_COLOR = (5, 5, 20)

EARTH_RADIUS = 360
ATMOS_RADIUS = 700


# ---------------------------------------------------------------------
# 5. PARTICLE SYSTEM
# ---------------------------------------------------------------------
class Particle:
    """Represents a gas particle moving within the Earth's atmosphere."""

    def __init__(self, gas, center, inner_r, outer_r):
        angle = random.uniform(0, 2 * np.pi)
        radius = random.uniform(inner_r + 10, outer_r - 10)
        self.x = center[0] + radius * np.cos(angle)
        self.y = center[1] + radius * np.sin(angle)
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-2, 2)
        self.gas = gas
        self.color = COLORS[gas]
        self.radius = 5

    def move(self, center, inner_r, outer_r):
        """Update position and handle bouncing off atmosphere boundaries."""
        self.x += self.vx
        self.y += self.vy

        dx, dy = self.x - center[0], self.y - center[1]
        dist = np.hypot(dx, dy) or 1.0

        # Bounce off Earth surface
        if dist < inner_r + self.radius:
            nx, ny = dx / dist, dy / dist
            dot = self.vx * nx + self.vy * ny
            self.vx -= 2 * dot * nx
            self.vy -= 2 * dot * ny
            self.x = center[0] + (inner_r + self.radius) * nx
            self.y = center[1] + (inner_r + self.radius) * ny

        # Bounce off outer atmosphere
        if dist > outer_r - self.radius:
            nx, ny = dx / dist, dy / dist
            dot = self.vx * nx + self.vy * ny
            self.vx -= 2 * dot * nx
            self.vy -= 2 * dot * ny
            self.x = center[0] + (outer_r - self.radius) * nx
            self.y = center[1] + (outer_r - self.radius) * ny

    def draw(self, surface, offset_x, offset_y):
        """Draw particle if visible in this node's viewport."""
        if (offset_x <= self.x < offset_x + WIDTH) and (offset_y <= self.y < offset_y + HEIGHT):
            pygame.draw.circle(
                surface, self.color,
                (int(self.x - offset_x), int(self.y - offset_y)),
                self.radius
            )


# ---------------------------------------------------------------------
# 6. SIMULATION LOOP
# ---------------------------------------------------------------------
particles = []


def adjust_particles(target_counts):
    """Adjust particle count for each gas to match data targets."""
    global particles
    new_particles = []
    for gas, target in target_counts.items():
        target = int(target)
        current = [p for p in particles if p.gas == gas]
        diff = target - len(current)
        if diff > 0:
            for _ in range(diff):
                current.append(Particle(gas, CENTER, EARTH_RADIUS, ATMOS_RADIUS))
        elif diff < 0:
            current = current[:target]
        new_particles += current
    particles = new_particles


clock = pygame.time.Clock()
running = True

# Generate static land patches
land_patches = []
for _ in range(8):
    angle = random.uniform(0, 2 * np.pi)
    r = EARTH_RADIUS * random.uniform(0.3, 0.7)
    land_x = int(CENTER[0] + r * np.cos(angle))
    land_y = int(CENTER[1] + r * np.sin(angle))
    land_width = random.randint(80, 160)
    land_height = random.randint(50, 100)
    land_patches.append((land_x, land_y, land_width, land_height))

# Preload fonts
small_font = pygame.font.Font(None, 48)
large_font = pygame.font.Font(None, 200)


# ---------------------------------------------------------------------
# 7. MAIN YEARLY ANIMATION LOOP
# ---------------------------------------------------------------------
for _, row in df.iterrows():
    year = int(row["year"])
    target_counts = {
        "co2": row["CO2_particles"],
        "ch4": row["CH4_particles"],
        "n2o": row["N2O_particles"]
    }

    adjust_particles(target_counts)
    year_text = large_font.render(f"Year: {year}", True, (255, 255, 255))

    frame_count = 0
    while frame_count < 120 and running:
        comm.Barrier()  # sync across all nodes per frame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                comm.Abort()  # clean exit for all MPI processes

        screen.fill(BG_COLOR)

        # Earth + atmosphere
        pygame.draw.circle(screen, ATMOS_COLOR,
                           (CENTER[0] - OFFSET_X, CENTER[1] - OFFSET_Y),
                           ATMOS_RADIUS, 2)
        pygame.draw.circle(screen, EARTH_COLOR,
                           (CENTER[0] - OFFSET_X, CENTER[1] - OFFSET_Y),
                           EARTH_RADIUS)

        # Land patches
        for x, y, w, h in land_patches:
            if (OFFSET_X <= x + w and x < OFFSET_X + WIDTH and
                OFFSET_Y <= y + h and y < OFFSET_Y + HEIGHT):
                pygame.draw.ellipse(screen, (34, 139, 34),
                                    (x - OFFSET_X, y - OFFSET_Y, w, h))

        # Particles
        for p in particles:
            p.move(CENTER, EARTH_RADIUS, ATMOS_RADIUS)
            p.draw(screen, OFFSET_X, OFFSET_Y)

        # Year label and legend (bottom-left node)
        if rank == 2:
            screen.blit(year_text, (60, HEIGHT - 200))

            # On-screen legend
            legend_items = [("CO2", COLORS["co2"]),
                            ("CH4", COLORS["ch4"]),
                            ("N2O", COLORS["n2o"])]
            for i, (label, color) in enumerate(legend_items):
                pygame.draw.circle(screen, color, (80, 80 + i * 60), 10)
                text = small_font.render(label, True, (255, 255, 255))
                screen.blit(text, (100, 65 + i * 60))

        pygame.display.flip()
        clock.tick(60)
        frame_count += 1

    if not running:
        break

pygame.quit()
