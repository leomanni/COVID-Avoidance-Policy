% 10-people map generation script.
% -------------------------------------------------------------------------
% Roberto Masocco, Edoardo Rossi, Leonardo Manni, Filippo Badalamenti,
% Emanuele Alfano
% July 21, 2021

function [grid, targets] = ten_people_map()
%TEN_PEOPLE_MAP Generates 10-people COVID gridworld map as matrix.
    
    % Map cells encoding values.
    free_val = 1;
    obstacle_val = 2;
    
    grid = ones(32, 32);
    grid = grid .* obstacle_val;
    
    % Draw room 1.
    for i = 2:15
        for j = 2:10
            grid(i, j) = free_val;
        end
    end
    
    % Draw room 2.
    for i = 2:10
        for j = 18:31
            grid(i, j) = free_val;
        end
    end
    
    % Draw room 3.
    for i = 14:20
        for j = 16:26
            grid(i, j) = free_val;
        end
    end
    
    % Draw room 4.
    for i = 23:31
        for j = 19:31
            grid(i, j) = free_val;
        end
    end
    
    % Draw room 5.
    for i = 18:30
        for j = 2:13
            grid(i, j) = free_val;
        end
    end
    
    % Draw corridor 1.
    for i = 3:5
        for j = 11:17
            grid(i, j) = free_val;
        end
    end
    
    % Draw corridor 2.
    for i = 11:13
        for j = 21:26
            grid(i, j) = free_val;
        end
    end
    
    % Draw corridor 3.
    for i = 21:22
        for j = 21:26
            grid(i, j) = free_val;
        end
    end
    
    % Draw corridor 4.
    for i = 28:30
        for j = 14:18
            grid(i, j) = free_val;
        end
    end
    
    % Draw obstacles in room 1.
    for i = 4:13
        for j = 6:7
            grid(i, j) = obstacle_val;
        end
    end
    
    % Draw obstacles in room 2.
    for i = 5:6
        for j = 22:27
            grid(i, j) = obstacle_val;
        end
    end
    
    % Draw obstacles in room 3.
    for i = 16:17
        for j = 19:20
            grid(i, j) = obstacle_val;
        end
    end
    
    % Draw obstacles in room 4.
    for i = 29:31
        grid(i, 27) = obstacle_val;
    end
    grid(25, 21) = obstacle_val;
    grid(25, 24) = obstacle_val;
    
    % Draw obstacles in room 5.
    for i = 21:25
        for j = 6:8
            grid(i, j) = obstacle_val;
        end
    end
    
    % Generate targets.
    targets = zeros(10, 1);
    targets(1) = sub2ind([32 32], 5, 2);
    targets(2) = sub2ind([32 32], 11, 10);
    targets(3) = sub2ind([32 32], 2, 18);
    targets(4) = sub2ind([32 32], 6, 28);
    targets(5) = sub2ind([32 32], 14, 16);
    targets(6) = sub2ind([32 32], 20, 26);
    targets(7) = sub2ind([32 32], 26, 30);
    targets(8) = sub2ind([32 32], 28, 27);
    targets(9) = sub2ind([32 32], 24, 9);
    targets(10) = sub2ind([32 32], 22, 5);
end