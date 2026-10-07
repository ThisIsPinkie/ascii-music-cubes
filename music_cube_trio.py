import pygame
import math
import numpy as np
import pyaudiowpatch as pyaudio
import threading

# ============ НАСТРОЙКИ ============
WIDTH, HEIGHT = 1100, 650
FPS = 120
COLS, ROWS = 110, 48
CHAR_W = WIDTH // COLS
CHAR_H = HEIGHT // ROWS
BG_COLOR = (6, 6, 14)
CHARS = " .:-=+*#%@"

# ============ АУДИО ============
CHUNK = 1024
GAIN = 50.0
SMOOTHING = 0.85
NUM_BARS = 40

bass_level = mid_level = high_level = 0.0
spectrum_bars = np.zeros(NUM_BARS)

# ============ НОРМАЛИЗАЦИЯ ============
baseline_bass = 0.01
baseline_mid = 0.01
baseline_high = 0.01
BASELINE_SPEED = 0.002

# ============ ВИЗУАЛ ============
FIXED_ZOOM = 0.9          # фиксированный размер куба
BEAT_THRESHOLD = 0.12
THICKNESS_MAX = 5         # максимальная толщина рёбер (0/1/2)

# ============ БИТ ============
prev_bass = prev_mid = prev_high = 0.0
flash_bass = flash_mid = flash_high = 0.0

def audio_loop():
    global bass_level, mid_level, high_level, spectrum_bars
    global baseline_bass, baseline_mid, baseline_high

    with pyaudio.PyAudio() as p:
        try:
            default_speakers = p.get_default_wasapi_loopback()
            print(f"Захват звука с: {default_speakers['name']}")
        except Exception as e:
            print(f"Ошибка: {e}")
            return

        rate = int(default_speakers["defaultSampleRate"])
        channels = default_speakers["maxInputChannels"]

        with p.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=default_speakers["index"],
            frames_per_buffer=CHUNK,
        ) as stream:
            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                samples = np.frombuffer(data, dtype=np.float32)
                if channels > 1:
                    samples = samples.reshape(-1, channels).mean(axis=1)

                fft = np.abs(np.fft.rfft(samples))
                freqs = np.fft.rfftfreq(len(samples), 1.0 / rate)

                bass_mask = (freqs >= 20) & (freqs < 250)
                mid_mask  = (freqs >= 250) & (freqs < 4000)
                high_mask = (freqs >= 4000) & (freqs < 16000)

                raw_bass = float(np.log1p(np.mean(fft[bass_mask])))
                raw_mid  = float(np.log1p(np.mean(fft[mid_mask])))
                raw_high = float(np.log1p(np.mean(fft[high_mask])))

                baseline_bass = baseline_bass * (1 - BASELINE_SPEED) + raw_bass * BASELINE_SPEED
                baseline_mid  = baseline_mid  * (1 - BASELINE_SPEED) + raw_mid  * BASELINE_SPEED
                baseline_high = baseline_high * (1 - BASELINE_SPEED) + raw_high * BASELINE_SPEED

                bass_level = min(1.0, max(0.0, (raw_bass - baseline_bass) * 4))
                mid_level  = min(1.0, max(0.0, (raw_mid  - baseline_mid)  * 4))
                high_level = min(1.0, max(0.0, (raw_high - baseline_high) * 4))

                bar_edges = np.logspace(np.log10(30), np.log10(16000), NUM_BARS + 1)
                bars = np.zeros(NUM_BARS)
                for i in range(NUM_BARS):
                    mask = (freqs >= bar_edges[i]) & (freqs < bar_edges[i + 1])
                    if mask.any():
                        bars[i] = np.log1p(np.mean(fft[mask]))
                spectrum_bars = np.clip(bars / (np.max(bars) + 0.001), 0, 1)

def start_audio():
    threading.Thread(target=audio_loop, daemon=True).start()

# ============ ГЕОМЕТРИЯ ============
cube_points = [
    (-1, -1, -1), ( 1, -1, -1), ( 1,  1, -1), (-1,  1, -1),
    (-1, -1,  1), ( 1, -1,  1), ( 1,  1,  1), (-1,  1,  1),
]
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

def rotate(point, ax, ay, az):
    x, y, z = point
    cx, sx = math.cos(ax), math.sin(ax)
    y, z = y * cx - z * sx, y * sx + z * cx
    cy, sy = math.cos(ay), math.sin(ay)
    x, z = x * cy + z * sy, -x * sy + z * cy
    cz, sz = math.cos(az), math.sin(az)
    x, y = x * cz - y * sz, x * sz + y * cz
    return (x, y, z)

def project(point, zoom, offset_x, offset_y):
    x, y, z = point
    distance = 3
    factor = distance / (distance + z)
    sx = int(offset_x + x * factor * COLS / 14 * zoom)
    sy = int(offset_y - y * factor * ROWS / 12 * zoom)
    return (sx, sy)

def depth_to_char(z):
    t = max(0.0, min(1.0, (1.7 - z) / 3.4))
    return CHARS[int(t * (len(CHARS) - 1))]

def lerp(a, b, t):
    return a + (b - a) * t

def bass_color(z, level, flash):
    t = max(0.0, min(1.0, (1.7 - z) / 3.4))
    r = int(lerp(60, 255, t * level) + 255 * flash)
    g = int(lerp(20, 140, t * level) + 200 * flash)
    b = int(lerp(20, 40,  t * level) + 150 * flash)
    return (min(255, r), min(255, g), min(255, b))

def mid_color(z, level, flash):
    t = max(0.0, min(1.0, (1.7 - z) / 3.4))
    r = int(lerp(20, 80,  t * level) + 150 * flash)
    g = int(lerp(60, 255, t * level) + 255 * flash)
    b = int(lerp(50, 180, t * level) + 200 * flash)
    return (min(255, r), min(255, g), min(255, b))

def high_color(z, level, flash):
    t = max(0.0, min(1.0, (1.7 - z) / 3.4))
    r = int(lerp(40, 160, t * level) + 200 * flash)
    g = int(lerp(20, 120, t * level) + 180 * flash)
    b = int(lerp(80, 255, t * level) + 255 * flash)
    return (min(255, r), min(255, g), min(255, b))

def draw_point(screen, font, x, y, ch, color, thickness):
    """Рисует символ. thickness = 0/1/2 — сколько соседей добавить."""
    if not (0 <= x < COLS and 0 <= y < ROWS):
        return
    text = font.render(ch, True, color)
    screen.blit(text, (x * CHAR_W, y * CHAR_H))
    if thickness >= 1:
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < COLS and 0 <= ny < ROWS:
                screen.blit(text, (nx * CHAR_W, ny * CHAR_H))

def draw_cube(screen, font, rotated, projected, color_func, level, flash):
    # Толщина рёбер растёт от уровня: 0 → 1 → 2
    thickness = int(level * THICKNESS_MAX)
    thickness = max(0, min(THICKNESS_MAX, thickness))

    for a, b in edges:
        x1, y1 = projected[a]
        x2, y2 = projected[b]
        z_mid = (rotated[a][2] + rotated[b][2]) / 2
        color = color_func(z_mid, level, flash)
        ch = depth_to_char(z_mid)
        steps = max(abs(x2 - x1), abs(y2 - y1), 1)
        for i in range(steps + 1):
            t = i / steps
            x = int(x1 + (x2 - x1) * t)
            y = int(y1 + (y2 - y1) * t)
            draw_point(screen, font, x, y, ch, color, thickness)

def main():
    global smooth_bass, smooth_mid, smooth_high, smooth_bars
    global prev_bass, prev_mid, prev_high, flash_bass, flash_mid, flash_high

    smooth_bass = smooth_mid = smooth_high = 0.0
    smooth_bars = np.zeros(NUM_BARS)
    prev_bass = prev_mid = prev_high = 0.0
    flash_bass = flash_mid = flash_high = 0.0

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Music Cube — Trio (Color + Brightness + Thickness)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 12)
    big_font = pygame.font.SysFont("consolas", 15)

    start_audio()
    print("Запущено. Включи музыку.")

    time_counter = 0.0

    cx1 = COLS // 6
    cx2 = COLS // 2
    cx3 = COLS * 5 // 6
    cy = ROWS // 2 - 5

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        s = SMOOTHING
        smooth_bass = smooth_bass * s + bass_level * (1 - s)
        smooth_mid  = smooth_mid  * s + mid_level  * (1 - s)
        smooth_high = smooth_high * s + high_level * (1 - s)
        smooth_bars = smooth_bars * s + spectrum_bars * (1 - s)

        # === БИТЫ ===
        if bass_level - prev_bass > BEAT_THRESHOLD:
            flash_bass = 1.0
        prev_bass = bass_level
        flash_bass *= 0.85

        if mid_level - prev_mid > BEAT_THRESHOLD:
            flash_mid = 1.0
        prev_mid = mid_level
        flash_mid *= 0.85

        if high_level - prev_high > BEAT_THRESHOLD:
            flash_high = 1.0
        prev_high = high_level
        flash_high *= 0.85

        # === ДЫХАНИЕ (только углы, без тряски) ===
        time_counter += 0.05
        breath = math.sin(time_counter) * 0.06

        angles = [
            (0.4 + breath, 0.6 + math.cos(time_counter * 0.7) * 0.06, 0.0),
            (0.4 - breath, 0.6 + math.sin(time_counter * 0.5) * 0.06, 0.0),
            (0.4 + breath * 0.5, 0.6 - math.cos(time_counter * 0.6) * 0.06, 0.0),
        ]
        centers = [(cx1, cy), (cx2, cy), (cx3, cy)]
        colors = [bass_color, mid_color, high_color]
        levels = [smooth_bass, smooth_mid, smooth_high]
        flashes = [flash_bass, flash_mid, flash_high]

        screen.fill(BG_COLOR)

        # === ТРИ КУБА ===
        for i in range(3):
            ax, ay, az = angles[i]
            ox, oy = centers[i]
            rot = [rotate(p, ax, ay, az) for p in cube_points]
            proj = [project(p, FIXED_ZOOM, ox, oy) for p in rot]
            draw_cube(screen, font, rot, proj, colors[i], levels[i], flashes[i])

        # === СПЕКТР ===
        bar_width = WIDTH // NUM_BARS
        for i in range(NUM_BARS):
            h = int(smooth_bars[i] * 90)
            t = i / NUM_BARS
            if t < 0.5:
                k = t * 2
                col = (int(255 * (1 - k * 0.7)), int(120 + 100 * k), int(80 * k))
            else:
                k = (t - 0.5) * 2
                col = (int(80 * (1 - k)), int(220 - 100 * k), int(80 + 175 * k))
            x = i * bar_width + 2
            y = HEIGHT - 20 - h
            pygame.draw.rect(screen, col, (x, y, bar_width - 4, h))

        # === HUD ===
        screen.blit(big_font.render(f"FPS: {int(clock.get_fps())}", True, (200, 200, 200)), (10, 10))
        screen.blit(big_font.render(f"BASS: {smooth_bass:.2f}", True, (255, 160, 100)), (10, 32))
        screen.blit(big_font.render(f"MID:  {smooth_mid:.2f}",  True, (120, 255, 160)), (10, 54))
        screen.blit(big_font.render(f"HIGH: {smooth_high:.2f}", True, (160, 160, 255)), (10, 76))
        screen.blit(big_font.render("[ESC] выход", True, (140, 180, 140)), (WIDTH - 150, 10))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()