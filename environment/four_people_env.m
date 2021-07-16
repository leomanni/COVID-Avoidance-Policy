% Script to generate a 4-people environment, and validate it.
% -------------------------------------------------------------------------
% Roberto Masocco, Edoardo Rossi, Leonardo Manni, Filippo Badalamenti,
% Emanuele Alfano
% July 16, 2021

clearvars
close all
clc

[map, targets] = four_people_map();
covid_four_env = COVIDGridworld(4, map, targets, 'rgby');

%% Validate and reset the new environment.
validateEnvironment(covid_four_env);
covid_four_env.reset();