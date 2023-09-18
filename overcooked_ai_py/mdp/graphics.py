import os
import pygame
import time
import numpy as np
import overcooked_ai_py
from overcooked_ai_py import ASSETS_DIR, PCG_EXP_IMAGE_DIR
from overcooked_ai_py.mdp.actions import Action, Direction
pygame.init()

INFO_PANEL_HEIGHT = 300  # height of the game info panel
INFO_PANEL_WIDTH = 0#100
INFO_PANEL_COLOR = (230, 180, 83)  # some sort of yellow
SPRITE_LENGTH = 50  # length of each sprite square
TERRAIN_DIR = 'terrain'
CHEF_DIR = 'chefs'
OBJECT_DIR = 'objects'
FONTS_DIR = 'fonts'
ARIAL_FONT = os.path.join(ASSETS_DIR, FONTS_DIR, 'arial.ttf')
TEXT_SIZE = 25

TERRAIN_TO_IMG = {
    ' ': os.path.join(ASSETS_DIR, TERRAIN_DIR, 'floor.png'),
    'X': os.path.join(ASSETS_DIR, TERRAIN_DIR, 'counter.png'),
    'P': os.path.join(ASSETS_DIR, TERRAIN_DIR, 'pot.png'),
    'O': os.path.join(ASSETS_DIR, TERRAIN_DIR, 'onions.png'),
    'T': os.path.join(ASSETS_DIR, TERRAIN_DIR, 'tomatoes.png'),
    'D': os.path.join(ASSETS_DIR, TERRAIN_DIR, 'dishes.png'),
    'S': os.path.join(ASSETS_DIR, TERRAIN_DIR, 'serve.png'),
    'M': os.path.join(ASSETS_DIR, TERRAIN_DIR, 'steaks.png'),
    'W': os.path.join(ASSETS_DIR, TERRAIN_DIR, 'sink.png'),
    'B': os.path.join(ASSETS_DIR, TERRAIN_DIR, 'board_knife.png'),
}

PLAYER_HAT_COLOR = {
    0: 'greenhat',
    1: 'bluehat',
}

PLAYER_ARROW_COLOR = {0: (0, 255, 0, 128), 1: (0, 0, 255, 128)}

PLAYER_ARROW_ORIENTATION = {
    Direction.DIRECTION_TO_STRING[Direction.NORTH]:
    ((15, 300), (35, 300), (35, 100), (50, 100), (25, 0), (0, 100), (15, 100)),
    Direction.DIRECTION_TO_STRING[Direction.SOUTH]:
    ((15, 0), (35, 0), (35, 200), (50, 200), (25, 300), (0, 200), (15, 200)),
    Direction.DIRECTION_TO_STRING[Direction.EAST]:
    ((0, 15), (0, 35), (200, 35), (200, 50), (300, 25), (200, 0), (200, 15)),
    Direction.DIRECTION_TO_STRING[Direction.WEST]:
    ((300, 15), (300, 35), (100, 35), (100, 50), (0, 25), (100, 0), (100, 15)),
}

PLAYER_ARROW_POS_SHIFT = {
    Direction.DIRECTION_TO_STRING[Direction.NORTH]:
    ((1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0)),
    Direction.DIRECTION_TO_STRING[Direction.SOUTH]:
    ((1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0)),
    Direction.DIRECTION_TO_STRING[Direction.EAST]:
    ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)),
    Direction.DIRECTION_TO_STRING[Direction.WEST]:
    ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)),
}


class Checkbox:
    def __init__(self, surface, x, y, idnum, color=(230, 230, 230),
        caption="", outline_color=(0, 0, 0), check_color=(0, 0, 0),
        font_size=26, font_color=(0, 0, 0), 
    text_offset=(18, 1), font='Ariel Black'):
        self.surface = surface
        self.x = x
        self.y = y
        self.color = color
        self.caption = caption
        self.oc = outline_color
        self.cc = check_color
        self.fs = font_size
        self.fc = font_color
        self.to = text_offset
        self.ft = font

        #identification for removal and reorginazation
        self.idnum = idnum

        # checkbox object
        self.checkbox_obj = pygame.Rect(self.x, self.y, 18, 18)
        self.checkbox_outline = self.checkbox_obj.copy()

        # variables to test the different states of the checkbox
        self.checked = False

    def _get_check_box_size(self):
        txt_size = self.font_surf.get_size()
        self.w = txt_size[0] + 18
        self.h = txt_size[1] + 18

    def _draw_button_text(self):
        self.font = pygame.font.SysFont(self.ft, self.fs)
        self.font_surf = self.font.render(self.caption, True, self.fc)
        w, h = self.font.size(self.caption)
        self.font_pos = (self.x + self.to[0], self.y + 18 / 2 - h / 2 + 
        self.to[1])
        self.surface.blit(self.font_surf, self.font_pos)
        self._get_check_box_size()

    def render_checkbox(self):
        if self.checked:
            pygame.draw.rect(self.surface, self.color, self.checkbox_obj)
            pygame.draw.rect(self.surface, self.oc, self.checkbox_outline, 1)
            pygame.draw.circle(self.surface, self.cc, (self.x + 9, self.y + 9), 4)

        elif not self.checked:
            pygame.draw.rect(self.surface, self.color, self.checkbox_obj)
            pygame.draw.rect(self.surface, self.oc, self.checkbox_outline, 1)
        self._draw_button_text()

    def _update(self, event_object):
        x, y = pygame.mouse.get_pos()
        px, py, w, h = self.checkbox_obj
        if px < x < px + self.w and py < y < py + self.h:
            if self.checked:
                self.checked = False
            else:
                self.checked = True

    def update_checkbox(self, event_object):
        if event_object.type == pygame.MOUSEBUTTONDOWN:
            self.click = True
            self._update(event_object)
            

def get_curr_pos(x, y, mode="human"):
    """
    Returns pygame.Rect object that specifies the position

    Args:
        x, y: position of the terrain in the terrain matrix
        mode: mode of rendering
    """
    if mode == "full":
        return pygame.Rect(
            x * SPRITE_LENGTH,
            y * SPRITE_LENGTH + INFO_PANEL_HEIGHT,
            SPRITE_LENGTH,
            SPRITE_LENGTH,
        )

    else:
        return pygame.Rect(
            x * SPRITE_LENGTH,
            y * SPRITE_LENGTH,
            SPRITE_LENGTH,
            SPRITE_LENGTH,
        )


def get_text_sprite(show_str):
    """
    Returns pygame.Surface object to show the text

    Args:
        show_str(string): The text to show
    """
    font = pygame.font.Font(ARIAL_FONT, TEXT_SIZE)
    text_surface = font.render(show_str, True, (255, 0, 0))
    return text_surface


def load_image(path):
    """
    Returns loaded pygame.Surface object from file path

    Args:
        path(string): file path to the image file
    """
    obj = pygame.image.load(path).convert()
    obj.set_colorkey((255, 255, 255))
    return pygame.transform.scale(obj, (SPRITE_LENGTH, SPRITE_LENGTH))


def blit_terrain(x, y, terrain_mtx, viewer, mode="human", in_view=True):
    """
    Helper function to blit given position to specified terrain

    Args:
        x, y: position of the terrain in the terrain matrix
        terrain_mtx: terrain matrix
        viewer: pygame surface that displays the game
    """
    curr_pos = get_curr_pos(x, y, mode)
    # render the terrain
    terrain = terrain_mtx[y][x]
    terrain_pgobj = load_image(TERRAIN_TO_IMG[terrain])
    if mode == 'fog' and not in_view:
        terrain_pgobj.set_alpha(1)
    viewer.blit(terrain_pgobj, curr_pos)


def get_player_sprite(player, player_index):
    """
    Returns loaded image of player(aka chef), the player's hat, and the color of the array to draw on top of the player

    Args:
        player(PlayerState)
        player_index(int)
    """
    orientation_str = get_orientation_str(player)

    player_img_path = ""
    hat_color = PLAYER_HAT_COLOR[player_index]
    hat_img_path = os.path.join(
        ASSETS_DIR, CHEF_DIR,
        '%s-%s.png' % (orientation_str, PLAYER_HAT_COLOR[player_index]))

    player_object = player.held_object
    # player holding object
    if player_object:
        # player holding soup
        obj_name = player_object.name
        if obj_name == 'soup':
            soup_type = player_object.state[0]
            player_img_path = os.path.join(
                ASSETS_DIR, CHEF_DIR,
                '%s-soup-%s.png' % (orientation_str, soup_type))

        # player holding non-soup
        else:
            player_img_path = os.path.join(
                ASSETS_DIR, CHEF_DIR,
                '%s-%s.png' % (orientation_str, obj_name))

    # player not holding object
    else:
        player_img_path = os.path.join(ASSETS_DIR, CHEF_DIR,
                                       '%s.png' % orientation_str)

    return load_image(player_img_path), load_image(hat_img_path)


def get_object_sprite(obj, on_pot=False, on=None, time=None):
    """
    Returns loaded image of object

    Args:
        obj(ObjectState)
        on_pot(boolean): whether the object lies on a pot
    """
    obj_name = obj.name

    if on_pot:
        soup_type, num_items, cook_time = obj.state
        obj_img_path = os.path.join(
            ASSETS_DIR, OBJECT_DIR,
            'soup-%s-%d-cooking.png' % (soup_type, num_items))
    elif on == "board":
        obj_img_path = os.path.join(
            ASSETS_DIR, OBJECT_DIR,
            'chopping.png')
    elif on == "sink":
        obj_img_path = os.path.join(
            ASSETS_DIR, OBJECT_DIR,
            'full_sink.png')
    else:
        if obj_name == 'soup':
            soup_type = obj.state[0]
            obj_img_path = os.path.join(ASSETS_DIR, OBJECT_DIR,
                                        'soup-%s-dish.png' % soup_type)
        else:
            obj_img_path = os.path.join(ASSETS_DIR, OBJECT_DIR,
                                        '%s.png' % obj_name)
    return load_image(obj_img_path)


def draw_arrow(window, player, player_index, pos, time_left):
    """
    Draw an arrow indicating orientation of the player
    """
    shift = 10.0
    orientation_str = get_orientation_str(player)
    arrow_orientation = PLAYER_ARROW_ORIENTATION[orientation_str]
    arrow_position = [[j * shift * time_left for j in i]
                      for i in PLAYER_ARROW_POS_SHIFT[orientation_str]]
    arrow_orientation = np.add(np.array(arrow_orientation),
                               arrow_position).tolist()
    arrow_color = PLAYER_ARROW_COLOR[player_index]

    arrow = pygame.Surface((300, 300)).convert()

    pygame.draw.polygon(arrow, arrow_color, arrow_orientation)
    arrow.set_colorkey((0, 0, 0))

    arrow = pygame.transform.scale(arrow, (SPRITE_LENGTH, SPRITE_LENGTH))
    window.blit(arrow, pos)
    # tmp = input()


def get_orientation_str(player):
    orientation = player.orientation
    # make sure the orientation exists
    assert orientation in Direction.ALL_DIRECTIONS

    orientation_str = Direction.DIRECTION_TO_STRING[orientation]
    return orientation_str


def render_from_grid(lvl_grid, log_dir, filename):
    """
    Render a single frame of game from grid level.
    This function is used for visualization the levels generated which
    are possibily broken or invalid. It also does not take the orientation
    of the players into account. So this method should not be used for
    actual game rendering.
    """
    width = len(lvl_grid[0])
    height = len(lvl_grid)
    window_size = width * SPRITE_LENGTH, height * SPRITE_LENGTH
    viewer = pygame.display.set_mode(window_size)
    viewer.fill((255, 255, 255))
    for y, terrain_row in enumerate(lvl_grid):
        for x, terrain in enumerate(terrain_row):
            curr_pos = get_curr_pos(x, y)

            # render player
            if str.isdigit(terrain):
                player = overcooked_ai_py.mdp.overcooked_mdp.PlayerState(
                    (x, y), Direction.SOUTH)
                player_idx = int(terrain)
                player_pgobj, player_hat_pgobj = get_player_sprite(
                    player, player_idx - 1)

                # render floor as background
                terrain_pgobj = load_image(TERRAIN_TO_IMG[" "])
                viewer.blit(terrain_pgobj, curr_pos)

                # then render the player
                viewer.blit(player_pgobj, curr_pos)
                viewer.blit(player_hat_pgobj, curr_pos)

            # render terrain
            else:
                terrain_pgobj = load_image(TERRAIN_TO_IMG[terrain])
                viewer.blit(terrain_pgobj, curr_pos)

    pygame.display.update()

    # save image
    pygame.image.save(viewer, os.path.join(log_dir, filename))


def render_game_info_panel(window, game_window_size, num_orders_remaining,
                           time_passed, curr_s='None', init=False):

    game_window_width, game_window_height = game_window_size

    # get panel rect
    panel_rect = pygame.Rect(0, game_window_height, game_window_width,
                             INFO_PANEL_HEIGHT)

    # fill with background color
    window.fill(INFO_PANEL_COLOR, rect=panel_rect)

    # update num orders left
    if num_orders_remaining == np.inf:
        num_orders_remaining = "inf"
    num_order_t_surface = get_text_sprite(
        f"Number of orders left: {num_orders_remaining}")
    num_order_text_pos = num_order_t_surface.get_rect()
    num_order_text_pos.topleft = panel_rect.topleft
    window.blit(num_order_t_surface, num_order_text_pos)

    # update time passed
    t_surface = get_text_sprite("Time passed: %3d s" % time_passed)
    time_passed_text_pos = t_surface.get_rect()
    _, num_order_txt_height = num_order_t_surface.get_size()
    time_passed_text_pos.y = num_order_text_pos.y + num_order_txt_height
    window.blit(t_surface, time_passed_text_pos)

    # # get panel rect
    # right_panel_rect = pygame.Rect(game_window_width, 0, INFO_PANEL_WIDTH,
    #                          game_window_height+INFO_PANEL_HEIGHT)

    # # fill with background color
    # window.fill(INFO_PANEL_COLOR, rect=right_panel_rect)
    
    # info text
    info_t_surface = get_text_sprite(
        f"Selected next action: {curr_s}")
    info_t_pos = info_t_surface.get_rect()
    _, t_surface_h = t_surface.get_size()
    info_t_pos.y = time_passed_text_pos.y + t_surface_h*2
    window.blit(info_t_surface, info_t_pos)

    _, info_t_h = info_t_surface.get_size()
    bh_offset = 5
    bw_offset = 5
    
    if init:
        b1 = Checkbox(window, panel_rect.x, info_t_pos.y+info_t_h, 0, caption='pickup meat')
        b1.render_checkbox()
        b2 = Checkbox(window, panel_rect.x + b1.font_pos[0]+b1.w+bw_offset, info_t_pos.y+info_t_h, 1, caption='pickup onion')
        b2.render_checkbox()
        b3 = Checkbox(window, panel_rect.x + b2.font_pos[0]+b2.w+bw_offset, info_t_pos.y+info_t_h, 2, caption='pickup plate')
        b3.render_checkbox()
        b4 = Checkbox(window, panel_rect.x + b3.font_pos[0]+b3.w+bw_offset, info_t_pos.y+info_t_h, 3, caption='pickup hot plate')
        b4.render_checkbox()
        b5 = Checkbox(window, panel_rect.x + b4.font_pos[0]+b4.w+bw_offset, info_t_pos.y+info_t_h, 4, caption='pickup garnish')
        b5.render_checkbox()
        

        second_row_y = info_t_pos.y+info_t_h+b1.h+bh_offset
        b1_2 = Checkbox(window, panel_rect.x, second_row_y, 6, caption='drop meat')
        b1_2.render_checkbox()
        b2_2 = Checkbox(window, b2.x, second_row_y, 7, caption='drop onion')
        b2_2.render_checkbox()
        b3_2 = Checkbox(window, b3.x, second_row_y, 8, caption='drop plate')
        b3_2.render_checkbox()
        b6 = Checkbox(window, b4.x, second_row_y, 5, caption='pickup steak')
        b6.render_checkbox()
        b3_3 = Checkbox(window, b5.x, second_row_y, 14, caption='deliver dish')
        b3_3.render_checkbox()

        third_row_y = info_t_pos.y+info_t_h+(b1.h*2)+bh_offset
        b1_3 = Checkbox(window, b2_2.x, third_row_y, 12, caption='chop onion')
        b1_3.render_checkbox()
        b2_3 = Checkbox(window, b3_2.x, third_row_y, 13, caption='heat hot plate')
        b2_3.render_checkbox()

        third_row_y = info_t_pos.y+info_t_h+(b1.h*3)+bh_offset
        b1_4 = Checkbox(window, panel_rect.x, third_row_y, 15, caption='stay')
        b1_4.render_checkbox()
        b2_4 = Checkbox(window, panel_rect.x + b1_4.font_pos[0]+50, third_row_y, 16, caption='up')
        b2_4.render_checkbox()
        b3_4 = Checkbox(window, panel_rect.x + b2_4.font_pos[0]+50, third_row_y, 17, caption='interact')
        b3_4.render_checkbox()

        forth_row_y = info_t_pos.y+info_t_h+(b1.h*4)+bh_offset
        b4_4 = Checkbox(window, b1_4.x, forth_row_y, 18, caption='left')
        b4_4.render_checkbox()
        b5_4 = Checkbox(window, b2_4.x, forth_row_y, 19, caption='down')
        b5_4.render_checkbox()
        b6_4 = Checkbox(window, b3_4.x, forth_row_y, 20, caption='right')
        b6_4.render_checkbox()

        box_list = [b1, b2, b3, b4, b5, b6, b1_2, b2_2, b3_2, b1_3, b2_3, b3_3, b1_4, b2_4, b3_4, b4_4, b5_4, b6_4]



    if init: return box_list