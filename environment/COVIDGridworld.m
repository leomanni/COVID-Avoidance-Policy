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
        
        % Stall-leading actions counter.
        stall_acts_cnt = 0
        
        % "Defeat" state reward (depends on map size).
        defeat_rew = 0
        
        % Single step reward.
        single_step_rew = -1
        
        % Victory state reward.
        victory_rew = 0
    end

    properties
        % People positions, initialized to a vector (depends on n_people).
        State = []
    end

    properties (Access = protected)
        % Internal flag to indicate episode termination.
        IsDone = false
    end

    %% Necessary Methods
    methods
        % Creates an instance of the environment.
        function this = COVIDGridworld()
            % Initialize Observation settings
            ObservationInfo = rlNumericSpec([4 1]);
            ObservationInfo.Name = 'CartPole States';
            ObservationInfo.Description = 'x, dx, theta, dtheta';
            
            % Initialize Action settings   
            ActionInfo = rlFiniteSetSpec([-1 1]);
            ActionInfo.Name = 'CartPole Action';
            
            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);
        end

        % Apply system dynamics and simulates the environment with the 
        % given action for one step.
        function [Observation,Reward,IsDone,LoggedSignals] = step(this, Action)
            
            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            notifyEnvUpdated(this);
        end

        % Reset environment to initial state and output initial observation
        function InitialObservation = reset(this)
            
            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            notifyEnvUpdated(this);
        end
    end

    %% Optional Methods
    methods               
        % Helper methods to create the environment
        % Discrete force 1 or 2
        function force = getForce(this,action)
            if ~ismember(action,this.ActionInfo.Elements)
                error('Action must be %g for going left and %g for going right.',-this.MaxForce,this.MaxForce);
            end
            force = action;           
        end
        
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
