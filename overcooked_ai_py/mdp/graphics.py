import os
import pygame
import time
from overcooked_ai_py.mdp.actions import Action, Direction
pygame.init()

SPRITE_LENGTH = 50 # length of each sprite square
ASSETS_DIR = 'assets'
TERRAIN_DIR = 'terrain'
CHEF_DIR = 'chefs'
OBJECT_DIR = 'objects'
FONTS_DIR = 'fonts'
ARIAL_FONT = os.path.join(ASSETS_DIR, FONTS_DIR, 'arial.ttf')
TEXT_SIZE = 25

TERRAIN_TO_IMG = {
    ' ': os.path.join(ASSETS_DIR,TERRAIN_DIR,'floor.png'),
    'X': os.path.join(ASSETS_DIR,TERRAIN_DIR,'counter.png'),
    'P': os.path.join(ASSETS_DIR,TERRAIN_DIR,'pot.png'),
    'O': os.path.join(ASSETS_DIR,TERRAIN_DIR,'onions.png'),
    'T': os.path.join(ASSETS_DIR,TERRAIN_DIR,'tomatoes.png'),
    'D': os.path.join(ASSETS_DIR,TERRAIN_DIR,'dishes.png'),
    'S': os.path.join(ASSETS_DIR,TERRAIN_DIR,'serve.png'),
}

PLAYER_HAT_COLOR = {
    0: 'greenhat',
    1: 'bluehat',
}

def get_curr_pos(x, y):
    """
    Returns pygame.Rect object that specifies the position

    Args:
        x, y: position of the terrain in the terrain matrix
    """
    return pygame.Rect(x*SPRITE_LENGTH, y*SPRITE_LENGTH, SPRITE_LENGTH, SPRITE_LENGTH)

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
    obj = pygame.image.load(path)
    return pygame.transform.scale(obj, (SPRITE_LENGTH, SPRITE_LENGTH))

def blit_terrain(x, y, terrain_mtx, viewer):
    """
    Helper function to blit given position to specified terrain

    Args:
        x, y: position of the terrain in the terrain matrix
        terrain_mtx: terrain matrix
        viewer: pygame surface that displays the game
    """
    curr_pos = get_curr_pos(x, y)
    # render the terrain
    terrain = terrain_mtx[y][x]
    terrain_pgobj = load_image(TERRAIN_TO_IMG[terrain])
    viewer.blit(terrain_pgobj, curr_pos)

def get_player_sprite(player, player_index):
    """
    Returns loaded image of player(aka chef) and the player's hat

    Args:
        player(PlayerState)
        player_index(int)
    """
    orientation = player.orientation
    # make sure the orientation exists
    assert orientation in Direction.ALL_DIRECTIONS

    orientation_str = Direction.DIRECTION_TO_STRING[orientation]

    player_img_path = ""
    hat_color = PLAYER_HAT_COLOR[player_index]
    hat_img_path = os.path.join(ASSETS_DIR, CHEF_DIR, '%s-%s.png' % (orientation_str, PLAYER_HAT_COLOR[player_index]))

    player_object = player.held_object
    # player holding object
    if player_object:
        # player holding soup
        obj_name = player_object.name
        if obj_name == 'soup':
            soup_type = player_object.state[0]
            player_img_path = os.path.join(ASSETS_DIR, CHEF_DIR, '%s-soup-%s.png' % (orientation_str, soup_type))

        # player holding non-soup
        else:
            player_img_path = os.path.join(ASSETS_DIR, CHEF_DIR, '%s-%s.png' % (orientation_str, obj_name))

    # player not holding object
    else:
        player_img_path = os.path.join(ASSETS_DIR, CHEF_DIR, '%s.png' % orientation_str)

    return load_image(player_img_path), load_image(hat_img_path)

def get_object_sprite(obj, on_pot = False):
    """
    Returns loaded image of object

    Args:
        obj(ObjectState)
        on_pot(boolean): whether the object lies on a pot
    """
    obj_name = obj.name

    if not on_pot:
        if obj_name == 'soup':
            soup_type = obj.state[0]
            obj_img_path = os.path.join(ASSETS_DIR, OBJECT_DIR, 'soup-%s-dish.png' % soup_type)
        else:
            obj_img_path = os.path.join(ASSETS_DIR, OBJECT_DIR, '%s.png' % obj_name)
    else:
        soup_type, num_items, cook_time = obj.state
        obj_img_path = os.path.join(ASSETS_DIR, OBJECT_DIR, 'soup-%s-%d-cooking.png' % (soup_type, num_items)) 
    return load_image(obj_img_path)