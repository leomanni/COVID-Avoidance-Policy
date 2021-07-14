% Multiple people gridworld implementation for COVID avoidance policy.
% -------------------------------------------------------------------------
% Roberto Masocco, Edoardo Rossi, Leonardo Manni, Filippo Badalamenti,
% Emanuele Alfano
% July 13, 2021

classdef COVIDGridworld < rl.env.MATLABEnvironment
    %COVIDGRIDWORLD: Multiple people RL gridworld.

    %% Gridworld Properties
    properties
        % Environment constants and physical characteristics.
        % People in the map.
        n_people = 0
        
        % Matrix encoding the map state in real time.
        map_mat = []
        
        % Target positions indices (depends on n_people).
        targets = []
        
        % Stall-leading actions counter.
        stall_acts_cnt = 0
        
        % Stall leading actions counter max value.
        max_stall_acts = 50;
        
        % "Defeat" state reward (depends on map size).
        defeat_rew = 0
        
        % Single step reward.
        single_step_rew = -1
        
        % "Victory" state reward.
        victory_rew = 0
        
        % People positions (depends on n_people).
        State = []
    end

    properties (Access = protected)
        % Internal flag to indicate episode termination.
        IsDone = false
    end

    %% Necessary Methods
    methods
        % Creates an instance of the environment.
        function this = COVIDGridworld(people, map, target_indices)
            % Generate a cell array that holds all possible actions.
            % Works as a car odometer.
            actions_cell = cell([5 ^ people, 1]);
            prev_cell = ones(people, 1);
            actions_cell{1} = prev_cell;
            for i = 2:(5 ^ people)
                prev_cell = actions_cell{i - 1};
                for j = 1:people
                    prev_cell(j) = prev_cell(j) + 1;
                    if prev_cell(j) == 6
                        prev_cell(j) = 1;
                        continue
                    else
                        break
                    end
                end
                actions_cell{i} = prev_cell;
            end
            
            % Initialize Observation settings.
            ObservationInfo = rlNumericSpec([people 1]);
            ObservationInfo.Name = 'COVIDGridworld Observation';
            ObservationInfo.Description = 'People positions';
            ObservationInfo.LowerLimit = 0;
            ObservationInfo.UpperLimit = size(map, 1) * size(map, 2);
            
            % Initialize Action settings.
            ActionInfo = rlFiniteSetSpec(actions_cell);
            ActionInfo.Name = 'COVIDGridworld Action';
            ActionInfo.Description = 'Set of people STOP+NSWE movements';
            
            % The following line implements built-in functions of RL env.
            % NOTE: This MUST be called before setting anything else!
            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);
            
            % Initialize other properties.
            this.n_people = people;
            this.map_mat = map;
            this.targets = target_indices;
            this.State = zeros(people, 1);
            this.defeat_rew = -1000 * size(map, 1) * size(map, 2);
        end

        % Simulates the environment with the given action for one step.
        function [Observation, Reward, IsDone, LoggedSignals] = step(this, Action)
            LoggedSignals = [];
            all_still = true;
            defeated = false;
            this.IsDone = false;
            
            % Process each person's movements.
            for i = 1:this.n_people
                curr_moved = false;
                curr_pos = this.State(i);
                curr_subs = ind2sub([size(this.map_mat, 1) size(this.map_mat, 2)], curr_pos);
                
                % Parse the action.
                switch Action(i)
                    case 1
                        % STOP
                        new_subs = curr_subs;
                        new_pos = curr_pos;
                    case 2
                        % NORTH
                        new_subs = [curr_subs(1) - 1, curr_subs(2)];
                        new_pos = sub2ind([size(this.map_mat, 1) size(this.map_mat, 2)], new_subs(1), new_subs(2));
                        curr_moved = true;
                    case 3
                        % SOUTH
                        new_subs = [curr_subs(1) + 1, curr_subs(2)];
                        new_pos = sub2ind([size(this.map_mat, 1) size(this.map_mat, 2)], new_subs(1), new_subs(2));
                        curr_moved = true;
                    case 4
                        % WEST
                        new_subs = [curr_subs(1), curr_subs(2) - 1];
                        new_pos = sub2ind([size(this.map_mat, 1) size(this.map_mat, 2)], new_subs(1), new_subs(2));
                        curr_moved = true;
                    case 5
                        % EAST
                        new_subs = [curr_subs(1), curr_subs(2) + 1];
                        new_pos = sub2ind([size(this.map_mat, 1) size(this.map_mat, 2)], new_subs(1), new_subs(2));
                        curr_moved = true;
                    otherwise
                        error('Invalid action %d for person %d.', Action(i), i);
                end
                
                % Check if the new position is feasible.
                if curr_moved == true
                    switch this.map_mat(new_subs(1), new_subs(2))
                        case 1
                            % Free cell: update map and internal state.
                            all_still = false;
                            this.map_mat(curr_subs(1), curr_subs(2)) = 1;
                            this.map_mat(new_subs(1), new_subs(2)) = 2 + i;
                            this.State(i) = new_pos;
                        case 2
                            % Obstacle detected: nothing to do.
                        otherwise
                            % Occupied cell: illegal move.
                            defeated = true;
                            break
                    end
                end
            end
            
            % Has no one moved?
            if all_still == true
                this.stall_acts_cnt = this.stall_acts_cnt + 1;
                if this.stall_acts_cnt == this.max_stall_acts
                    % Stalled for too long.
                    defeated = true;
                end
            end
            
            % "Defeat" state: set return values and get out.
            if defeated == true
                this.IsDone = true;
                IsDone = true;
                Observation = zeros(this.n_people, 1);
                Reward = this.defeat_rew;
                notifyEnvUpdated(this);
                return
            end
            
            % Check for "Victory".
            won = true;
            for i = 1:this.n_people
                if ~ismember(this.State(i), this.targets)
                    won = false;
                end
            end
            if won == true
                this.IsDone = true;
                IsDone = true;
                Observation = this.State;
                Reward = this.victory_rew;
                notifyEnvUpdated(this);
                return
            end
            
            % Just a normal execution step.
            IsDone = false;
            Observation = this.State;
            Reward = this.single_step_rew;
            notifyEnvUpdated(this);
        end

        % Resets environment and observation to initial state.
        function InitialObservation = reset(this)
            % Reset counters and other properties.
            this.stall_acts_cnt = 0;
            this.IsDone = false;
            
            % Clear the map from people (not the first time!).
            if this.State(1) ~= 0
                for i = 1:this.n_people
                    person_subs = ind2sub([size(this.map_mat, 1) size(this.map_mat, 2)], this.State(i));
                    this.map_mat(person_subs(1), person_subs(2)) = 1;
                end
            end
            
            % Generate and set a new initial state.
            InitialObservation = zeros(this.n_people, 1);
            for i = 1:this.n_people
                while true
                    new_pos = randi(size(this.map_mat, 1) * size(this.map_mat, 2));
                    % Check if this is not a target.
                    if ismember(new_pos, this.targets) == true
                        % A new random extraction is necessary.
                        continue
                    end
                    % Check if the cell is free.
                    new_subs = ind2sub([size(this.map_mat, 1) size(this.map_mat, 2)], new_pos);
                    if this.map_mat(new_subs(1), new_subs(2)) ~= 1
                        % A new random extraction is necessary.
                        continue
                    else
                        this.map_mat(new_subs(1), new_subs(2)) = 2 + i;
                        this.State(i) = new_pos;
                        InitialObservation(i) = new_pos;
                        break
                    end
                end
            end
            
            % Signal that the environment has been updated.
            notifyEnvUpdated(this);
        end
    end

    %% Auxiliary Methods
    methods
        % Visualization method.
        function plot(this)
            % Initiate the visualization.
            
            % Update the visualization.
            envUpdatedCallback(this)
        end
    end

    methods (Access = protected)
        % (optional) update visualization everytime the environment is updated 
        % (notifyEnvUpdated is called)
        function envUpdatedCallback(this)
        end
    end
end