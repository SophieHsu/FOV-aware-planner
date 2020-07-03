import os
import pygame
import time
from overcooked_ai_py.mdp.actions import Action, Direction
pygame.init()

SPRITE_LENGTH = 50 # length of each sprite square
ASSETS_DIR = 'assets'
TERRAIN_DIR = 'terrain'
CHEF_DIR = 'chefs'

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

def load_image(path):
    obj = pygame.image.load(path)
    return pygame.transform.scale(obj, (SPRITE_LENGTH, SPRITE_LENGTH))

def get_player_sprite(player, player_index):
    '''
    Returns image path to player(aka chef) and the player's hat

    Args:
        player(PlayerState)
        player_index(int)
    '''
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
            soup_type = player_object[0]
            player_img_path = os.path.join(ASSETS_DIR, CHEF_DIR, '%s-soup-%s.png' % (orientation_str, soup_type))
        
        # player holding non-soup
        else:
            player_img_path = os.path.join(ASSETS_DIR, CHEF_DIR, '%s-%s.png' % (orientation_str, obj_name))
    
    # player not holding object
    else:
        player_img_path = os.path.join(ASSETS_DIR, CHEF_DIR, '%s.png' % orientation_str)
    
    return load_image(player_img_path), load_image(hat_img_path)