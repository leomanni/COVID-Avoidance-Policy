% Script to generate a 10-people environment, and validate it.
% -------------------------------------------------------------------------
% Roberto Masocco, Edoardo Rossi, Leonardo Manni, Filippo Badalamenti,
% Emanuele Alfano
% July 21, 2021

clearvars
close all
clc

[map, targets] = ten_people_map();
covid_ten_env = COVIDGridworld(10, map, targets, {[1 0 0]
                                                  [0 1 0]
                                                  [0 0 1]
                                                  [0 1 1]
                                                  [1 0 1]
                                                  [1 1 0]
                                                  [0 0.4470 0.7410]
                                                  [0.85 0.325 0.098]
                                                  [0.929 0.694 0.125]
                                                  [0.494 0.184 0.556]});

%% Validate and reset the new environment.
validateEnvironment(covid_ten_env);
covid_ten_env.reset();