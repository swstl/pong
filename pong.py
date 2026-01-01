## ping pong game in python
from abc import ABC, abstractmethod
import pygame
import sys
import random


class Window:
    def __init__(self, width, height, title):
        self.width = width
        self.height = height
        self.title = title
        self.background = (0, 0, 0)
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.title)

    def draw(self):
        self.screen.fill(self.background)

    def draw_entity(self, entity):
        if isinstance(entity, list):
            for e in entity:
                e.draw(self.screen)
        else:
            entity.draw(self.screen)


class Entity(ABC):
    def __init__(self, x, y):
        self.color = (255, 255, 255)
        self.x = x
        self.y = y
        self.window: Window | None = None
        self.speed = 500

    @abstractmethod
    def draw(self, screen):
        pass

    @abstractmethod
    def move(self, delta):
        pass

    @classmethod
    def from_window(cls, window):
        entity = cls(0, 0)
        entity.window = window
        return entity

    @abstractmethod
    def get_rect(self):
        return pygame.Rect(self.x, self.y, 0, 0)

    def collides_with(self, entity):
        return self.get_rect().colliderect(entity.get_rect())


class Ball(Entity):
    def __init__(self, x, y, size, vel_x, vel_y):
        super().__init__(x, y)
        self.size = size
        self.vel_x = self.speed * vel_x
        self.vel_y = self.speed * vel_y

    def move(self, delta):
        if self.window is None:
            print("No window assigned to ball")
            return

        self.x += self.vel_x * delta
        self.y += self.vel_y * delta

        if self.y <= 0 or self.y + self.size >= self.window.height:
            self.vel_y = -self.vel_y

    def draw(self, screen):
        pygame.draw.rect(
            screen, (255, 255, 255), (self.x, self.y, self.size, self.size)
        )

    @classmethod
    def from_window(cls, window):
        size = 20
        x = window.width // 2 - size // 2
        y = window.height // 2 - size // 2
        vel_x = random.choice([-1, 1])
        vel_y = random.choice([-1, 1])
        ball = Ball(x, y, size, vel_x, vel_y)
        ball.window = window
        return ball

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.size, self.size)

    def is_out_of_bounds(self):
        if self.window is None:
            print("No window assigned to ball")
            return False

        if self.x < 0:
            return "left"
        elif self.x + self.size > self.window.width:
            return "right"
        return None


class Player(Entity):
    def __init__(self, x, y, width, height):
        super().__init__(x, y)
        self.width = width
        self.height = height
        self.vel = 0

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    @classmethod
    def from_window(cls, window):
        player = Player(0, window.height // 2 - 60, 10, 120)
        player.window = window
        return player

    def move(self, delta):
        self.y += self.vel * delta
        if self.y < 0:
            self.y = 0
        if self.y + self.height > 800:
            self.y = 800 - self.height

    def draw(self, screen):
        pygame.draw.rect(
            screen, (255, 255, 255), (self.x, self.y, self.width, self.height)
        )

    def at_side(self, side):
        if side == "left":
            self.x = 50 - 10
        elif side == "right":
            self.x = 1000 - 50 - 10
        return self

    def at_right_side(self):
        return self.at_side("right")

    def at_left_side(self):
        return self.at_side("left")

    def vel_up(self):
        self.vel = -self.speed

    def vel_down(self):
        self.vel = self.speed

    def vel_stop(self):
        self.vel = 0

    def is_above(self, entity):
        return self.y + self.height / 2 > entity.y + entity.size / 2

    def is_below(self, entity):
        return self.y + self.height / 2 < entity.y + entity.size / 2


class PongGame:
    def __init__(self, render=True):
        pygame.init()
        self.render = render

        if self.render:
            self.window = Window(1000, 800, "Ping Pong Game")
        else:
            self.window = type("obj", (object,), {"width": 1000, "height": 800})()

        # Set up display

        self.players = []
        self.players.append(Player.from_window(self.window).at_left_side())
        self.players.append(Player.from_window(self.window).at_right_side())
        self.ball = Ball.from_window(self.window)
        self.clock = pygame.time.Clock()
        self.winning_side = "none"

    def update(self):
        if self.render and isinstance(self.window, Window):
            dt = self.clock.tick(60) / 1000
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("quitting")
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    match event.key:
                        case pygame.K_l:
                            for player in self.players:
                                player.speed += 50
                            self.ball.speed += 50

                        case pygame.K_h:
                            for player in self.players:
                                player.speed -= 50
                            self.ball.speed -= 50

            self.window.draw()
        else:
            dt = 1 / 60

        if random.random() > 0.7:  # Only move 70% of the time
            if self.players[0].is_above(self.ball):
                self.players[0].vel_up()
            elif self.players[0].is_below(self.ball):
                self.players[0].vel_down()

        for player in self.players:
            player.move(dt)

        if self.ball.collides_with(self.players[0]) or self.ball.collides_with(
            self.players[1]
        ):
            self.ball.vel_x = -self.ball.vel_x

        self.ball.move(dt)

        if self.render and isinstance(self.window, Window):
            self.window.draw_entity(self.players)
            self.window.draw_entity(self.ball)
            pygame.display.flip()

        return self.get_state()

    def get_possible_actions(self):
        return {1: "up", 2: "down", 0: "idle"}

    def get_state(self):
        return {
            "ball_x": self.ball.x,
            "ball_y": self.ball.y,
            "ball_vel_x": self.ball.vel_x,
            "ball_vel_y": self.ball.vel_y,
            "player_right_y": self.players[1].y,
            "player_left_y": self.players[0].y,
            "ball_colliding_left": self.ball.collides_with(self.players[0]),
            "ball_colliding_right": self.ball.collides_with(self.players[1]),
            "side_lost": self.ball.is_out_of_bounds(),
        }

    def move_player(self, player, action):
        p = 0 if player == "left" else 1

        if action == "up":
            self.players[p].vel_up()
        elif action == "down":
            self.players[p].vel_down()
        else:
            self.players[p].vel_stop()


if __name__ == "__main__":
    game = PongGame()
    while True:
        game.update()
