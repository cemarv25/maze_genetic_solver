from mazelib import Maze
from mazelib.generate.Prims import Prims

if __name__ == '__main__':
    maze_width = 27
    maze_height = 11
    maze = Maze()
    maze.generator = Prims(maze_height, maze_width)
    maze.generate()
    maze.generate_entrances()
    print(maze)