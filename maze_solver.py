import random
from statistics import mean
from textwrap import wrap
from mazelib import Maze
from mazelib.generate.Prims import Prims

mock_maze_grid = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]


def sum_immediate_neighbors(maze, x, y):
    total = 0

    total += maze[x - 1][y]  # top neighbor
    total += maze[x][y - 1]  # left neighbor
    total += maze[x][y + 1]  # right neighbor
    total += maze[x + 1][y]  # bottom neighbor

    return total


def generate_maze(height, width):
    maze = Maze()
    maze.generator = Prims(height, width)
    maze.generate()
    maze.generate_entrances()

    return maze


def generate_mock_maze():
    maze = Maze()
    maze.start = (23, 0)
    maze.end = (5, 30)
    maze.grid = mock_maze_grid
    return maze


def get_maze_intersections(grid: list[list[int]]):
    intersections = []
    for row in range(1, len(grid) - 1):
        for col in range(1, len(grid[row]) - 1):
            if grid[row][col] == 0 and sum_immediate_neighbors(grid, row, col) < 2:
                intersections.append((row, col))

    return intersections


def print_maze_grid(maze: Maze):
    grid = maze.grid
    for row in range(len(grid)):
        for col in range(len(grid[row])):
            if (row, col) == maze.start:
                print('S', end=' ')
                continue

            if (row, col) == maze.end:
                print('E', end=' ')
                continue
                
            print('#', end=' ') if grid[row][col] else print(' ', end=' ')
        
        print('')
            

def print_maze_solution(maze: Maze, solution):
    grid = maze.grid
    for row in range(len(grid)):
        for col in range(len(grid[row])):
            if (row, col) == maze.start:
                print('S', end=' ')
                continue

            if (row, col) == maze.end:
                print('E', end=' ')
                continue

            if (row, col) in solution:
                print('+', end=' ')
                continue
                
            print('#', end=' ') if grid[row][col] else print(' ', end=' ')
        
        print('')


def generate_solution(maze: Maze, intersections, individual, track_intersection_count=False, start_from=None):
    if not start_from:
        start_from = maze.start

    path = []
    intersection_count = 0
    x, y = start_from

    if x == 0:
        x += 1
    elif x == len(maze.grid[0]) - 1:
        x -= 1
    elif y == 0:
        y += 1
    else:
        y -= 1

    path.append((x, y))

    while sum_immediate_neighbors(maze.grid, x, y) < 3:
        if (x, y) in intersections:
            decision_idx = intersections.index((x, y))
            x, y = move((x, y), individual[decision_idx])
            
            if track_intersection_count and (maze.grid[x][y] == 1 or  x <= 0 or x >= len(maze.grid) - 1 or y <= 0 or y >= len(maze.grid[0]) - 1):
                return (path, intersection_count)
            
            path.append((x, y))
            intersection_count += 1
        else:
            grid = maze.grid
            if grid[x - 1][y] == 0 and (x - 1, y) not in path and x - 1 > 0:
                x -= 1
            elif grid[x + 1][y] == 0 and (x + 1, y) not in path and x + 1 < len(grid) - 1:
                x += 1
            elif grid[x][y + 1] == 0 and (x, y + 1) not in path and y + 1 < len(grid[0]) - 1:
                y += 1
            elif grid[x][y - 1] == 0 and (x, y - 1) not in path and y - 1 > 0:
                y -= 1
            else:
                break
            
            path.append((x, y))
    
    return path if track_intersection_count == False else (path, intersection_count)


def distance_between(point1, point2):
    """Calculate the manhattan distance between two points."""

    return abs(point2[0] - point1[0]) + abs(point2[1] - point1[1])


def are_equal(ind1, ind2):
    for i in range(len(ind1)):
        if ind1[i] != ind2[i]:
            return False
    
    return True


def log_generation_data(maze: Maze, intersections, population: list[str], population_eval: list[int], generation_num: int):
    print(f'--- Generation #{generation_num} Statistics ---')
    print('\tBest individual: ',
          ''.join(population[population_eval.index(max(population_eval))]))
    print('\tBest individual aptitude: ', max(population_eval))
    solution_from_start = generate_solution(maze, intersections, population[population_eval.index(max(population_eval))])
    solution_from_end = generate_solution(maze, intersections, population[population_eval.index(max(population_eval))], start_from=maze.end)
    solution = solution_from_start + solution_from_end
    print('\tRoute followed: ')
    print_maze_solution(maze, solution)
    print('\tAverage aptitude: ', mean(population_eval))
    print('\n')


def move(from_coords: tuple[int, int], direction: str):
    # up
    if direction == '00':
        return (from_coords[0] - 1, from_coords[1]) 

    # down
    if direction == '01':
        return (from_coords[0] + 1, from_coords[1]) 
    
    # right
    if direction == '10':
        return (from_coords[0], from_coords[1] + 1) 
    
    # right
    if direction == '11':
        return (from_coords[0], from_coords[1] - 1) 


def select(population: list[str], population_eval: list[int]):
    total = sum(population_eval)
    weights = [population_eval[i] / total for i in range(len(population))]

    return random.choices(population, weights, k=len(population))


def reproduce(population: list[str]):
    x = 0
    y = 1
    while y < len(population):
        ind1 = ''.join(population[x])
        ind2 = ''.join(population[y])

        cross = random.randint(1, len(ind1) - 1)

        offspring1 = ind1[:cross] + ind2[cross:]
        offspring2 = ind2[:cross] + ind1[cross:]

        population[x] = wrap(offspring1, 2)
        population[y] = wrap(offspring2, 2)

        x += 2
        y += 2


def is_convergent(population: list[str]):
    model = population[0]
    individual_idx = 1

    while individual_idx < len(population):
        for char_idx in range(len(model)):
            individual = population[individual_idx]
            if model[char_idx] != individual[char_idx]:
                return False

        individual_idx += 1

    return True


def mutate(population: list[str]):
    while random.randint(0, 1):
        individual_idx = random.randint(0, len(population) - 1)
        gen_idx = random.randint(0, len(population[individual_idx]) - 1)

        new_individual = list(''.join(population[individual_idx]))
        new_individual[gen_idx] = '0' if new_individual[gen_idx] == '1' else '1'
        population[individual_idx] = wrap(''.join(new_individual), 2)


def evaluate(maze: Maze, population, intersections):
    aptitudes = []
    for individual in population:
        ind_aptitude = 0
        for decision_idx in range(len(individual)):
            x, y = move(intersections[decision_idx], individual[decision_idx])
            if maze.grid[x][y] == 0:
                ind_aptitude += 1
    
        path_from_start, intersection_count_from_start = generate_solution(maze, intersections, individual, True)
        ind_aptitude += (intersection_count_from_start) * 2

        path_from_end, intersection_count_from_end = generate_solution(maze, intersections, individual, True, maze.end)
        ind_aptitude += (intersection_count_from_end) * 3

        ind_aptitude -= (distance_between(path_from_start[-1], path_from_end[-1])) / 2

        aptitudes.append(ind_aptitude)
    
    return aptitudes


def genetic(maze: list[list[int]], intersections: list[tuple[int, int]]):
    N = len(intersections) * 2
    L = len(intersections)

    print(f'* Performing algorithm with {N} individuals of {L * 2} bits.')

    generation_num = 1
    population = []

    for _ in range(N):
        individual_list = []
        for _ in range(L):
            individual_list.append(f'{random.getrandbits(2):0{2}b}')

        population.append(individual_list)

    population_eval = evaluate(maze, population, intersections)
    best_ind = population[population_eval.index(max(population_eval))]
    best_repeat = 0

    log_generation_data(maze, intersections, population, population_eval, generation_num)

    while not is_convergent(population) or best_repeat > 15:
        population = select(population, population_eval)
        reproduce(population)
        mutate(population)
        population_eval = evaluate(maze, population, intersections)

        new_best = population[population_eval.index(max(population_eval))]
        if are_equal(''.join(best_ind), ''.join(new_best)):
            best_repeat += 1
        else:
            best_ind = new_best
            best_repeat = 0

        generation_num += 1
        log_generation_data(maze, intersections, population, population_eval, generation_num)

    return population


if __name__ == '__main__':
    # maze_width = 27
    # maze_height = 11
    # generate_maze(maze_height, maze_width)

    maze = generate_mock_maze()
    intersections = get_maze_intersections(maze.grid)

    print('--- Maze generated ---')
    print_maze_grid(maze)
    print('\n')
    
    genetic(maze, intersections)
    