% 3-people map generation script.
% -------------------------------------------------------------------------
% Roberto Masocco, Edoardo Rossi, Leonardo Manni, Filippo Badalamenti,
% Emanuele Alfano
% April 19, 2022


function [grid, targets] = three_people_map()
% 3PEOPLE_MAP Generates 3-people COVID gridworld map as matrix.

    % Map cells encoding values.
    free_val = 1;
    obstacle_val = 2;

    grid = ones(12, 12);
    grid = grid .* free_val;

    n = size(grid);

    % Vertical edges.
    for i = 1:n(1)
        grid(i, 1) = obstacle_val;
        grid(i, n(2)) = obstacle_val;
    end

    % Horizontal edges.
    for i = 1:n(2)
      grid(1, i) = obstacle_val;
      grid(n(1), i) = obstacle_val;
    end
    
    % Highest 1x2   
    for i = 4:5
        grid(2,i) = obstacle_val;
    end
    
    % Square 2x2
    for i = 5:6
        for j = 5:6
            grid(i,j) = obstacle_val;
        end
    end
        
    % Point
    grid(9,4) = obstacle_val;
    
    % Lower 1x2
    for i = 10:11
        grid(11,i) = obstacle_val;
    end
    
    % Generate targets
    targets = zeros(3,1);
    targets(1) = sub2ind([12 12], 3, 4);
    targets(2) = sub2ind([12 12], 5, 10);
    targets(3) = sub2ind([12 12], 9, 6);


end







