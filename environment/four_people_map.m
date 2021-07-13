% 4-people map generation script.
% -------------------------------------------------------------------------
% Roberto Masocco, Edoardo Rossi, Leonardo Manni, Filippo Badalamenti,
% Emanuele Alfano
% July 13, 2021

function grid = four_people_map()
% 4PEOPLE_MAP Generates 4-people COVID gridworld map as matrix.
    % Map cells encoding values.
    free_val = 1;
    obstacle_val = 2;

    grid = ones(15, 16);
    grid .* free_val;

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

    % Lower left.
    for i = 12:14
        grid(i, 2) = obstacle_val;
    end

    % Lower right.
    for i = 11:14
        grid(i, 15) = obstacle_val;
    end

    % Higher left, vertical.
    for i = 5:7
        grid(i, 5) = obstacle_val;
    end

    % Lower horizontal.
    for i = 7:10
        grid(12, i) = obstacle_val;
    end

    % Higher right square.
    for i = 2:9
        for j = 9:15
            grid(i, j) = obstacle_val;
        end
    end
end