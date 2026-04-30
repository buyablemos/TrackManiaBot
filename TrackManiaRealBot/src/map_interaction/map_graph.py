import ast
import os
import json

from src.config import Config

def _read_map_layout(map_layout_path):
    """
    Read the map layout from the output of the GBX file. First you need to read the GBX file and read it using pygbx.
    :return: A list of the blocks of the map, containing, for each block, the name and the position
    """
    blocks = []
    names_lut = {"StadiumRoadMainStartLine": "Start", "StadiumPlatformToRoadMain": "Transition",
                 "StadiumRoadMainFinishLine": "Finish", "StadiumCircuitBase": "Road"}
    with open(map_layout_path, "r") as f:
        for line in f:
            if line.startswith("Flags") or line.startswith("Rotation"):
                continue
            if line.startswith("Name"):
                name = line.split(" ")[1].strip()
                blocks.append([names_lut.get(name, name)])
            elif line.startswith("Position"):
                position = line.split(" ", maxsplit=1)[-1].strip()
                blocks[-1].append(ast.literal_eval(position))
    return blocks

def _keep_highest_blocks(blocks):
    """
    Keep only the blocks that are the highest in their column
    :param blocks: the list of blocks
    :return: the list of blocks that are the highest in their column
    """
    blocks = sorted(blocks, key=lambda x: x[1][1], reverse=True)  # Sort by height
    highest_blocks = []

    for i, block in enumerate(blocks):
        without_y = [block[1][0], block[1][2]]
        if i == 0:
            highest_blocks.append([block[0], without_y])

        elif without_y not in [b[1] for b in highest_blocks]:
            highest_blocks.append([block[0], without_y])

    return highest_blocks


def is_next_to(pos1, pos2):
    """
    Sprawdza czy bloki są blisko siebie (zwiększona tolerancja do 2 jednostek)
    """
    dist_x = abs(pos1[0] - pos2[0])
    dist_z = abs(pos1[1] - pos2[1])  # w Twoim kodzie pos[1] to współrzędna Z

    # Szukamy klocków w promieniu 2 jednostek (pozwala na małe skoki i luki)
    return dist_x <= 3 and dist_z <= 3 and (dist_x != 0 or dist_z != 0)
def _order_blocks_starting_with_start_block(blocks):
    """
    Order the blocks starting with the start block and then the transition block (to be sure to go in the right direction at the beginning)
    :param blocks: the list of blocks containing their name and position
    :return: the ordered list of blocks
    """
    start_block = None
    transition_block= None
    for block in blocks:
        if block[0] == "Start":
            start_block = block[1]
            blocks.remove(block)
            break

    for block in blocks:
        if block[0] == "Transition" and is_next_to(start_block, block[1]):
            transition_block = block[1]
            blocks.remove(block)
            break

    if not start_block or not transition_block:
        raise Exception("Start or Transition block not found")

    ordered_blocks = [start_block, transition_block]

    while blocks:
        found_neighbor = False  # <-- Dodajemy flagę
        for i, block in enumerate(blocks):
            if is_next_to(ordered_blocks[-1], block[1]):
                ordered_blocks.append(block[1])
                blocks.remove(block)
                found_neighbor = True  # <-- Znaleźliśmy!
                break

        if not found_neighbor:  # <-- Jeśli przeszukał wszystko i nic nie znalazł
            print(f"BŁĄD: Przerwa w torze po klocku {ordered_blocks[-1]}")
            print(f"Zostało jeszcze {len(blocks)} klocków, których nie potrafię połączyć.")
            break  # <-- WYCHODZIMY Z PĘTLI, żeby nie było freezu

    return ordered_blocks

def order_blocks_of_map(map_layout_path):
    """
    Order the blocks of a map. First you need to read the GBX file and read it using pygbx.
    :param map_layout_path: the name of the file containing the map
    :return: the ordered list of blocks
    """
    blocks = _read_map_layout(map_layout_path)
    blocks = _keep_highest_blocks(blocks)
    blocks = _order_blocks_starting_with_start_block(blocks)

    return blocks

def calculate_direction(block1, block2):
    """
    Calculate the direction between two blocks, it will be normalized
    :param block1: the first block
    :param block2: the second block
    :return: the direction between the two blocks
    """
    direction = (block2[0] - block1[0], block2[1] - block1[1])
    if direction[0] != 0:
        direction = (direction[0] / abs(direction[0]), 0)
    elif direction[1] != 0:
        direction = (0, direction[1] / abs(direction[1]))
    return direction

def get_nodes(ordered_blocks):
    """
    Get the nodes of the map. Nodes are the blocks where a turn happens. They will be the nodes of the graph.
    :param ordered_blocks: the ordered list of blocks
    :return: a list of the nodes
    """
    nodes = [ordered_blocks[0]]
    for i in range(1, len(ordered_blocks) - 1):
        if calculate_direction(ordered_blocks[i - 1], ordered_blocks[i]) != calculate_direction(ordered_blocks[i], ordered_blocks[i + 1]):
            nodes.append(ordered_blocks[i])

    nodes.append(ordered_blocks[-1])
    return nodes

def get_turns(nodes):
    """
    Get the turns of the map.
    A left turn is represented by a -1 and a right turn by a 1
    :param nodes: the list of the nodes
    :return: a list of the turns
    """
    turns = []
    for i in range(1, len(nodes) - 1):
        if calculate_direction(nodes[i - 1], nodes[i]) == (-1, 0) and calculate_direction(nodes[i], nodes[i + 1]) == (0, 1):
            turns.append(-1)
        elif calculate_direction(nodes[i - 1], nodes[i]) == (0, 1) and calculate_direction(nodes[i], nodes[i + 1]) == (1, 0):
            turns.append(-1)
        elif calculate_direction(nodes[i - 1], nodes[i]) == (1, 0) and calculate_direction(nodes[i], nodes[i + 1]) == (0, -1):
            turns.append(-1)
        elif calculate_direction(nodes[i - 1], nodes[i]) == (0, -1) and calculate_direction(nodes[i], nodes[i + 1]) == (-1, 0):
            turns.append(-1)
        else:
            turns.append(1)

    return turns

def dump_map_layout_to_json(map_layout_path):
    """
    Dump the map layout to a json file. The layout will contain the nodes and the turns of the map
    :param map_layout_path: The name of the file containing the layout of the map (output of the GBX parsing script)
    :return: None
    """
    print("1")
    blocks = order_blocks_of_map(map_layout_path)
    print("2")
    nodes = get_nodes(blocks)
    print("3")
    turns = get_turns(nodes)
    print("4")

    with open(Config.Paths.MAP_BLOCKS_PATH, "w") as f:
        f.write(json.dumps({"nodes": nodes, "turns": turns}))


if __name__ == "__main__":
    #os.chdir("../../")
    map_name = Config.Paths.MAP_LAYOUT_PATH
    dump_map_layout_to_json(map_name)