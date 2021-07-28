% Script to generate a 4-people environment, and validate it.
% -------------------------------------------------------------------------
% Roberto Masocco, Edoardo Rossi, Leonardo Manni, Filippo Badalamenti,
% Emanuele Alfano
% July 16, 2021

clearvars
close all
clc

[map, targets] = four_people_map();
covid_four_env = COVIDGridworld(4, map, targets, {'r', 'g', 'b', 'y'});

%% Validate and reset the new environment.
validateEnvironment(covid_four_env);
covid_four_env.reset();

%% Train the agent
critic_agent = makeCriticAgent(covid_four_env);
opt = rlTrainingOptions(...
    'MaxEpisodes',1000,...
    'MaxStepsPerEpisode',1000,...
    'StopTrainingCriteria',"AverageReward",...
    'StopTrainingValue',480,...
    'UseParallel', true);

trainStats = train(critic_agent,covid_four_env,opt);
