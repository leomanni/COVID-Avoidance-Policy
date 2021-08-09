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

%% Create the training algorithm.
sarsa_agent = makeCriticAgent(covid_four_env);
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',1e5,...
    'MaxStepsPerEpisode',1000,...
    'StopTrainingCriteria',"AverageReward",...
    'StopTrainingValue',0, ...
    'Verbose',true,...
    'Plots',"training-progress");

% trainOpts.UseParallel = true;
trainOpts.ParallelizationOptions.Mode = "async";
trainOpts.ParallelizationOptions.DataToSendFromWorkers = "gradients"; %for A3C
trainOpts.ParallelizationOptions.StepsUntilDataIsSent = 20;
trainOpts.ParallelizationOptions.WorkerRandomSeeds = -1;
trainOpts.StopOnError = 'off';

%% Train the agent in the environment.
% plot(covid_four_env);
trainStats = train(sarsa_agent,covid_four_env,trainOpts);
save("sarsaTrain.mat",'trainStats','covid_four_env','trainOpts');
% close
% for i = 1:1000
%     trainStats= train(sarsa_agent,covid_four_env,trainOpts);
%     save("sarsaTrainFor.mat",'trainStats','covid_four_env','trainOpts');
%     close
% end
% Extract Weight of the network
critic = getCritic(sarsa_agent);
criticParams = getLearnableParameters(critic);
%% RESUME

% load('sarsaTrain.mat')
% trainStats = train(trainStats,covid_four_env,trainOpts);