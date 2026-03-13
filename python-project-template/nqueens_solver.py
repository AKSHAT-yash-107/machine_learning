import heapq
import random
import time
from copy import deepcopy
from typing import List, Tuple, Set

class NQueensSolver:
    """Solver for N-Queens problem using A* and Hill Climbing algorithms"""
    
    def __init__(self, n: int):
        self.n = n
        
    def count_conflicts(self, state: List[int]) -> int:
        """
        Count the number of conflicts (attacking queen pairs) in a state.
        State is represented as a list where state[i] = column position of queen in row i.
        """
        conflicts = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                # Same column
                if state[i] == state[j]:
                    conflicts += 1
                # Same diagonal
                elif abs(state[i] - state[j]) == abs(i - j):
                    conflicts += 1
        return conflicts
    
    def is_goal(self, state: List[int]) -> bool:
        """Check if state is a goal state (no conflicts)"""
        return self.count_conflicts(state) == 0
    
    def print_board(self, state: List[int]):
        """Print the chess board with queens"""
        print("\n" + "=" * (self.n * 4 + 1))
        for row in range(self.n):
            print("|", end="")
            for col in range(self.n):
                if state[row] == col:
                    print(" Q |", end="")
                else:
                    print(" . |", end="")
            print()
        print("=" * (self.n * 4 + 1))
        print(f"Conflicts: {self.count_conflicts(state)}\n")
    
    # ==================== A* ALGORITHM ====================
    
    def heuristic(self, state: List[int]) -> int:
        """
        Heuristic function for A*: number of conflicts (attacking pairs)
        Lower is better (0 = goal state)
        """
        return self.count_conflicts(state)
    
    def get_successors_astar(self, state: List[int]) -> List[List[int]]:
        """
        Generate successor states for A*.
        For each row, try moving the queen to each column.
        """
        successors = []
        current_row = len(state)
        
        if current_row == self.n:
            return successors
        
        # Try placing a queen in each column of the current row
        for col in range(self.n):
            new_state = state + [col]
            successors.append(new_state)
        
        return successors
    
    def solve_astar(self, verbose: bool = True) -> Tuple[List[int], dict]:
        """
        Solve N-Queens using A* search algorithm.
        Returns solution state and statistics.
        """
        start_time = time.time()
        
        # Priority queue: (f_score, g_score, state)
        # f_score = g_score + h_score
        initial_state = []
        h_score = 0  # Heuristic for empty board
        frontier = [(h_score, 0, initial_state)]
        explored = set()
        nodes_explored = 0
        max_frontier_size = 0
        
        if verbose:
            print("\n" + "="*60)
            print("A* SEARCH ALGORITHM")
            print("="*60)
        
        while frontier:
            max_frontier_size = max(max_frontier_size, len(frontier))
            f_score, g_score, current_state = heapq.heappop(frontier)
            
            nodes_explored += 1
            
            # Convert state to tuple for hashing
            state_tuple = tuple(current_state)
            if state_tuple in explored:
                continue
            explored.add(state_tuple)
            
            if verbose and nodes_explored % 1000 == 0:
                print(f"Nodes explored: {nodes_explored}, Frontier size: {len(frontier)}")
            
            # Check if goal state
            if len(current_state) == self.n and self.is_goal(current_state):
                end_time = time.time()
                stats = {
                    'nodes_explored': nodes_explored,
                    'max_frontier_size': max_frontier_size,
                    'time_taken': end_time - start_time,
                    'solution_cost': g_score
                }
                
                if verbose:
                    print(f"\nSOLUTION FOUND!")
                    print(f"Nodes explored: {nodes_explored}")
                    print(f"Max frontier size: {max_frontier_size}")
                    print(f"Time taken: {stats['time_taken']:.4f} seconds")
                    self.print_board(current_state)
                
                return current_state, stats
            
            # Generate successors
            for successor in self.get_successors_astar(current_state):
                successor_tuple = tuple(successor)
                if successor_tuple not in explored:
                    new_g_score = g_score + 1
                    h_score = self.heuristic(successor)
                    new_f_score = new_g_score + h_score
                    heapq.heappush(frontier, (new_f_score, new_g_score, successor))
        
        if verbose:
            print("No solution found!")
        return None, {'nodes_explored': nodes_explored}
    
    # ==================== HILL CLIMBING ALGORITHM ====================
    
    def get_neighbors_hillclimb(self, state: List[int]) -> List[List[int]]:
        """
        Generate all neighbor states for hill climbing.
        For each queen, try moving it to each column in its row.
        """
        neighbors = []
        for row in range(self.n):
            for col in range(self.n):
                if col != state[row]:  # Different from current position
                    neighbor = state.copy()
                    neighbor[row] = col
                    neighbors.append(neighbor)
        return neighbors
    
    def solve_hill_climbing(self, max_restarts: int = 100, verbose: bool = True) -> Tuple[List[int], dict]:
        """
        Solve N-Queens using Hill Climbing with random restarts.
        Returns solution state and statistics.
        """
        start_time = time.time()
        total_moves = 0
        restarts = 0
        
        if verbose:
            print("\n" + "="*60)
            print("HILL CLIMBING ALGORITHM (with Random Restarts)")
            print("="*60)
        
        for restart in range(max_restarts):
            # Random initial state
            current_state = [random.randint(0, self.n - 1) for _ in range(self.n)]
            current_conflicts = self.count_conflicts(current_state)
            
            if verbose:
                print(f"\nRestart {restart + 1}/{max_restarts}")
                print(f"Initial conflicts: {current_conflicts}")
            
            moves_in_this_climb = 0
            
            while True:
                # Check if goal reached
                if current_conflicts == 0:
                    end_time = time.time()
                    stats = {
                        'total_moves': total_moves,
                        'restarts': restarts,
                        'time_taken': end_time - start_time,
                        'final_conflicts': current_conflicts
                    }
                    
                    if verbose:
                        print(f"\nSOLUTION FOUND!")
                        print(f"Total restarts: {restarts}")
                        print(f"Total moves: {total_moves}")
                        print(f"Time taken: {stats['time_taken']:.4f} seconds")
                        self.print_board(current_state)
                    
                    return current_state, stats
                
                # Find best neighbor
                neighbors = self.get_neighbors_hillclimb(current_state)
                best_neighbor = None
                best_conflicts = current_conflicts
                
                for neighbor in neighbors:
                    neighbor_conflicts = self.count_conflicts(neighbor)
                    if neighbor_conflicts < best_conflicts:
                        best_neighbor = neighbor
                        best_conflicts = neighbor_conflicts
                
                # If no better neighbor found, we're at local optimum
                if best_neighbor is None:
                    if verbose:
                        print(f"Stuck at local optimum with {current_conflicts} conflicts after {moves_in_this_climb} moves")
                    break
                
                # Move to best neighbor
                current_state = best_neighbor
                current_conflicts = best_conflicts
                moves_in_this_climb += 1
                total_moves += 1
                
                if verbose and moves_in_this_climb % 10 == 0:
                    print(f"Moves: {moves_in_this_climb}, Current conflicts: {current_conflicts}")
            
            restarts += 1
        
        # Failed to find solution
        end_time = time.time()
        if verbose:
            print(f"\nFailed to find solution after {max_restarts} restarts")
            print(f"Best state found:")
            self.print_board(current_state)
        
        stats = {
            'total_moves': total_moves,
            'restarts': restarts,
            'time_taken': end_time - start_time,
            'final_conflicts': current_conflicts
        }
        
        return current_state, stats
    
    # ==================== STEEPEST ASCENT HILL CLIMBING ====================
    
    def solve_steepest_ascent(self, max_restarts: int = 100, verbose: bool = True) -> Tuple[List[int], dict]:
        """
        Solve N-Queens using Steepest Ascent Hill Climbing.
        Always chooses the best neighbor in each step.
        """
        start_time = time.time()
        total_moves = 0
        restarts = 0
        
        if verbose:
            print("\n" + "="*60)
            print("STEEPEST ASCENT HILL CLIMBING")
            print("="*60)
        
        for restart in range(max_restarts):
            current_state = [random.randint(0, self.n - 1) for _ in range(self.n)]
            current_conflicts = self.count_conflicts(current_state)
            
            if verbose:
                print(f"\nRestart {restart + 1}/{max_restarts}")
                print(f"Initial conflicts: {current_conflicts}")
            
            moves_in_this_climb = 0
            
            while True:
                if current_conflicts == 0:
                    end_time = time.time()
                    stats = {
                        'total_moves': total_moves,
                        'restarts': restarts,
                        'time_taken': end_time - start_time,
                        'final_conflicts': current_conflicts
                    }
                    
                    if verbose:
                        print(f"\nSOLUTION FOUND!")
                        print(f"Total restarts: {restarts}")
                        print(f"Total moves: {total_moves}")
                        print(f"Time taken: {stats['time_taken']:.4f} seconds")
                        self.print_board(current_state)
                    
                    return current_state, stats
                
                neighbors = self.get_neighbors_hillclimb(current_state)
                
                # Find THE best neighbor (steepest ascent)
                best_neighbor = current_state
                best_conflicts = current_conflicts
                
                for neighbor in neighbors:
                    neighbor_conflicts = self.count_conflicts(neighbor)
                    if neighbor_conflicts < best_conflicts:
                        best_neighbor = neighbor
                        best_conflicts = neighbor_conflicts
                
                # No improvement possible
                if best_conflicts >= current_conflicts:
                    if verbose:
                        print(f"Stuck at local optimum with {current_conflicts} conflicts")
                    break
                
                current_state = best_neighbor
                current_conflicts = best_conflicts
                moves_in_this_climb += 1
                total_moves += 1
            
            restarts += 1
        
        end_time = time.time()
        stats = {
            'total_moves': total_moves,
            'restarts': restarts,
            'time_taken': end_time - start_time,
            'final_conflicts': current_conflicts
        }
        
        return current_state, stats


def main():
    """Demonstrate both algorithms for N=8"""
    
    print("\n" + "#"*60)
    print("#" + " "*58 + "#")
    print("#" + " "*15 + "N-QUEENS PROBLEM SOLVER" + " "*20 + "#")
    print("#" + " "*20 + "N = 8" + " "*27 + "#")
    print("#" + " "*58 + "#")
    print("#"*60)
    
    n = 8
    solver = NQueensSolver(n)
    
    # Solve using A*
    print("\n\n" + "*"*60)
    print("*" + " "*58 + "*")
    print("*" + " "*20 + "METHOD 1: A* SEARCH" + " "*19 + "*")
    print("*" + " "*58 + "*")
    print("*"*60)
    
    astar_solution, astar_stats = solver.solve_astar(verbose=True)
    
    # Solve using Hill Climbing
    print("\n\n" + "*"*60)
    print("*" + " "*58 + "*")
    print("*" + " "*16 + "METHOD 2: HILL CLIMBING" + " "*18 + "*")
    print("*" + " "*58 + "*")
    print("*"*60)
    
    hc_solution, hc_stats = solver.solve_hill_climbing(max_restarts=100, verbose=True)
    
    # Comparison
    print("\n\n" + "="*60)
    print("COMPARISON OF ALGORITHMS")
    print("="*60)
    print(f"\n{'Metric':<30} {'A*':<15} {'Hill Climbing':<15}")
    print("-" * 60)
    print(f"{'Solution Found':<30} {'Yes' if astar_solution else 'No':<15} {'Yes' if hc_solution and hc_stats['final_conflicts'] == 0 else 'No':<15}")
    print(f"{'Time Taken (seconds)':<30} {astar_stats['time_taken']:<15.4f} {hc_stats['time_taken']:<15.4f}")
    print(f"{'Nodes/Moves Explored':<30} {astar_stats['nodes_explored']:<15} {hc_stats['total_moves']:<15}")
    if 'restarts' in hc_stats:
        print(f"{'Restarts (HC only)':<30} {'-':<15} {hc_stats['restarts']:<15}")
    print("\n" + "="*60)
    
    print("\nKEY OBSERVATIONS:")
    print("1. A* guarantees an optimal solution but explores many nodes")
    print("2. Hill Climbing is faster but may need multiple restarts")
    print("3. A* is complete and optimal; Hill Climbing can get stuck")
    print("4. For N-Queens, both methods work well for N=8")
    

if __name__ == "__main__":
    main()
