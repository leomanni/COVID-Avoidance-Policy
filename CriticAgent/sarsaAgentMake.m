clc
clear

%% ############## Game Parameter ##############
% L'input layer, sono n^2 nodi con 1 solo valore assegnato dall'ambiente
n = 8;          % Celle per lato
player = 3;     % 1 = libero, 2 = ostacolo, i-player=2+i => max val = #player+2


obsInfo = rlNumericSpec([(n^2) 1]); % vector of n^2 observations: for each node, the state (free, obstacle, player-i)
obsInfo.LowerLimit = 1;
obsInfo.UpperLimit = player+2; % #player+2
obsInfo.Description = "Input state Layer per l'approssimazione di Q(s,a)";
obsInfo.Name = "Q([S],a)";
% disp(obsInfo)


% Action:= 1=fermo, 2=Nord, 3=Est, 4=Sud, 5=Ovest
% Create all 5^player combination table
vectComb = zeros (5^player, player);
endCond = ones(1,player)*5;
for i = 1 : height(vectComb)
    comb = dec2base(i-1,5);
    for j = 1 : length(comb)
        vectComb(i, (player-length(comb)) + j) = str2double(comb(j));
    end
end
vectComb = vectComb + 1; % to have action from 1 to 5
% Transform the combination table in a cel list
a = {};
for act = 1:height(vectComb)
    a = [a;{vectComb(act,:)'}]; % Transpose the input because the network layer is a column
end

actInfo = rlFiniteSetSpec(a);   %Create un Insieme Finito di ingressi multi ingresso
actInfo.Description = "Input action Layer per l'approssimazione di Q(s,a)";
actInfo.Name = "Q(s,[A])";


inputNode = obsInfo.Dimension(1) + actInfo.Dimension(1);


%% ############## Network Create ##############

% Create Layer Graph
% Create the layer graph variable to contain the network layers.
lgraph = layerGraph();

% Add Layer Branches
% Add the branches of the network to the layer graph. Each branch is a linear array of layers.
% imageInputLayer([Alteza, Larghezza, NumCanali],...), dove NumCanali è sempre e comunque double
tempLayers = imageInputLayer([obsInfo.Dimension(1) obsInfo.Dimension(2) 1],"Name","StateInput","Normalization","none");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = imageInputLayer([actInfo.Dimension(1) actInfo.Dimension(2) 1],"Name","ActionInput","Normalization","none");
lgraph = addLayers(lgraph,tempLayers);

% Utility list, to define the layer, the 2 input layer arent present,
% because have to be connected to the concat layer
tempLayers = [  
    concatenationLayer(1,2,"Name","concat") % Connetto 2 layer nella dimensione 1
    fullyConnectedLayer(inputNode * 4,"Name","fc_1")
    sigmoidLayer("Name","sigmoid_1")
    fullyConnectedLayer(inputNode,"Name","fc_2")
    sigmoidLayer("Name","sigmoid_2")
    fullyConnectedLayer(1,"Name","fc_3")];
lgraph = addLayers(lgraph,tempLayers);

clear tempLayers; % clean up helper variable


% Connect Layer Branches
% Connect all the branches of the network to create the network graph.
lgraph = connectLayers(lgraph,"StateInput","concat/in1"); %in1 da man di connectLayers
lgraph = connectLayers(lgraph,"ActionInput","concat/in2"); %in2 da man di connectLayers

%% ############## agent SARSA create ##############

critic = rlQValueRepresentation(lgraph,obsInfo,actInfo,'Observation',"StateInput",'Action',"ActionInput") ;

opt = rlSARSAAgentOptions;
opt.EpsilonGreedyExploration.Epsilon = 0.05;

agent = rlSARSAAgent(critic,opt);