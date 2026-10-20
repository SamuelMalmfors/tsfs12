import queue

import carla
import numpy as np
import pygame


WIDTH = 1280
HEIGHT = 720

PANEL_BG = (15, 15, 20, 200)
WHITE = (235, 235, 235)
MUTED = (160, 160, 170)
RED = (255, 80, 80)


class CameraView:
    """
    Follow camera shown in a pygame window with a text panel
    in the corner. Optionally also writes every frame to an mp4.

    Requires synchronous mode, so every world.tick() produces
    exactly one frame.
    """

    def __init__(self, world, dt, video_path=None, fov=90.0):
        bp = world.get_blueprint_library().find(
            "sensor.camera.rgb"
        )

        bp.set_attribute("image_size_x", str(WIDTH))
        bp.set_attribute("image_size_y", str(HEIGHT))
        bp.set_attribute("fov", str(fov))
        bp.set_attribute("sensor_tick", str(dt))

        self.camera = world.spawn_actor(
            bp,
            world.get_spectator().get_transform()
        )

        # Pinhole focal length in pixels, for projecting labels
        self.focal = WIDTH / (2.0 * np.tan(np.radians(fov) / 2.0))

        self.frames = queue.Queue()
        self.camera.listen(self.frames.put)

        pygame.init()
        pygame.display.set_caption("ACC: PID vs MPC")
        self.display = pygame.display.set_mode((WIDTH, HEIGHT))

        self.font = pygame.font.SysFont("consolas", 22, bold=True)
        self.small_font = pygame.font.SysFont("consolas", 19)
        self.label_font = pygame.font.SysFont("arial", 20, bold=True)

        self.writer = None

        if video_path is not None:
            # Imported here so imageio is only needed when recording
            import imageio

            self.writer = imageio.get_writer(
                str(video_path),
                fps=round(1.0 / dt),
                codec="libx264",
                quality=8,
                macro_block_size=8,
                # Fragmented mp4: stays playable even if the
                # process is killed before close()
                output_params=[
                    "-movflags", "frag_keyframe+empty_moov"
                ]
            )

    def set_transform(self, transform):
        self.camera.set_transform(transform)

    def show(self, frame_id, lines, labels=(), timeout=5.0):
        """
        Show the frame belonging to frame_id (returned by
        world.tick()) with a text panel in the top-left corner.

        lines: list of (text, (r, g, b)); an empty text adds a gap.
        labels: list of (text, carla.Location, (r, g, b)) drawn
        at that world position in the image.
        Raises KeyboardInterrupt if the window is closed.
        """

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt

        try:
            image = self.frames.get(timeout=timeout)

            # Drop stale frames from earlier ticks
            while image.frame < frame_id:
                image = self.frames.get(timeout=timeout)

        except queue.Empty:
            print("Viewer warning: missing frame")
            return

        bgra = np.frombuffer(
            image.raw_data,
            dtype=np.uint8
        ).reshape((image.height, image.width, 4))

        rgb = bgra[:, :, 2::-1]

        surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        self.display.blit(surface, (0, 0))

        world_to_camera = np.array(image.transform.get_inverse_matrix())
        self.draw_labels(labels, world_to_camera)

        self.draw_panel(lines)
        pygame.display.flip()

        if self.writer is not None:
            frame = pygame.surfarray.array3d(self.display).swapaxes(0, 1)
            self.writer.append_data(frame)

    def project(self, location, world_to_camera):
        point = world_to_camera @ np.array(
            [location.x, location.y, location.z, 1.0]
        )

        # Unreal camera axes: x forward, y right, z up
        forward, right, up = point[:3]

        if forward <= 0.1:
            return None

        return (
            WIDTH / 2.0 + self.focal * right / forward,
            HEIGHT / 2.0 - self.focal * up / forward
        )

    def draw_labels(self, labels, world_to_camera):
        items = []

        for text, location, color in labels:
            pixel = self.project(location, world_to_camera)

            if pixel is None:
                continue

            rendered = self.label_font.render(text, True, (255, 255, 255))
            box = rendered.get_rect(center=(int(pixel[0]), int(pixel[1])))
            box.inflate_ip(16, 6)

            items.append((box, rendered, color))

        # Cars in neighbouring lanes are close in the image:
        # push overlapping labels apart sideways
        items.sort(key=lambda item: item[0].centerx)

        for (left, _, _), (right, _, _) in zip(items, items[1:]):
            overlap = left.right + 6 - right.left

            if overlap > 0 and left.colliderect(right.inflate(12, 0)):
                left.move_ip(-(overlap // 2 + 1), 0)
                right.move_ip(overlap // 2 + 1, 0)

        for box, rendered, color in items:
            pygame.draw.rect(self.display, color, box, border_radius=6)
            pygame.draw.rect(
                self.display, (20, 20, 20), box, width=2, border_radius=6
            )
            self.display.blit(rendered, rendered.get_rect(center=box.center))

    def draw_panel(self, lines):
        rendered = []

        for i, (text, color) in enumerate(lines):
            font = self.font if i == 0 else self.small_font
            rendered.append(font.render(text, True, color) if text else None)

        line_height = 26
        width = max(
            (r.get_width() for r in rendered if r is not None),
            default=0
        ) + 32
        height = line_height * len(lines) + 20

        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        self.display.blit(panel, (16, 16))

        y = 26
        for r in rendered:
            if r is not None:
                self.display.blit(r, (32, y))
            y += line_height

    def close(self):
        # Finish the video file first so it is never left
        # unplayable if destroying the camera fails
        try:
            if self.writer is not None:
                self.writer.close()
        finally:
            if self.camera.is_alive:
                self.camera.stop()
                self.camera.destroy()

            pygame.quit()
