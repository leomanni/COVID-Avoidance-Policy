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
            actions_cell = cell([1, 5 ^ people]);
            prev_cell = ones(1, people);
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
            ObservationInfo.LowerLimit = 1;
            ObservationInfo.UpperLimit = size(map, 1) * size(map, 2);
            
            % Initialize Action settings.
            ActionInfo = rlFiniteSetSpec(actions_cell);
            ActionInfo.Name = 'COVIDGridworld Action';
            ActionInfo.Description = 'Set of people NSWE+STOP movements';
            
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
        function [Observation,Reward,IsDone,LoggedSignals] = step(this, Action)
            
            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            notifyEnvUpdated(this);
        end

        % Resets environment and observation to initial state.
        function InitialObservation = reset(this)
            % Reset counters and other properties.
            this.stall_acts_cnt = 0;
            
            % Clear the map from people (not the first time!).
            if this.State(1) ~= 0
                for i = 1:this.n_people
                    person_subs = ind2sub([size(this.map_mat, 1) size(this.map_mat, 2)], this.State(i));
                    this.map_mat(person_subs(1), person_subs(2)) = 0;
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
                    if this.map_mat(new_subs(1), new_subs(2)) ~= 0
                        % A new random extraction is necessary.
                        continue
                    else
                        this.map_mat(new_subs(1), new_subs(2)) = i;
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
        % Reward function
        function Reward = getReward(this)
            if ~this.IsDone
                Reward = this.RewardForNotFalling;
            else
                Reward = this.PenaltyForFalling;
            end          
        end

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
