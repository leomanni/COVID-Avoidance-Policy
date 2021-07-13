clc
clear

% L'input layer, sono n^2 nodi con 1 solo valore assegnato dall'ambiente
n = 8;          % Celle per lato
player = 2;     % 1 = libero, 2 = ostacolo, i-player=2+i => max val = #player+2

obsInfo = rlNumericSpec([(n^2) 1]); % vector of n^2 observations: for each node, the state (free, obstacle, player-i)
obsInfo.LowerLimit = 1;
obsInfo.UpperLimit = player+2; % #player+2
obsInfo.Description = "Input state Layer per l'approssimazione di Q(s,a)";
obsInfo.Name = "Q([S],a)";
% disp(obsInfo)


% Action:= 1=fermo, 2=Nord, 3=Est, 4=Sud, 5=Ovest
actInfo = rlNumericSpec([player 1]);
actInfo.LowerLimit = 1;
actInfo.UpperLimit = 5; % #Action
actInfo.Description = "Input action Layer per l'approssimazione di Q(s,a)";
actInfo.Name = "Q(s,[A])";
% disp(actInfo)

inputNode = obsInfo.Dimension(1) + actInfo.Dimension(1);

% Create Layer Graph
% Create the layer graph variable to contain the network layers.
lgraph = layerGraph();

% Add Layer Branches
% Add the branches of the network to the layer graph. Each branch is a linear array of layers.
% imageInputLayer([Alteza, Larghezza, NumCanali],...), dove NumCanali è sempre e comunque double
tempLayers = imageInputLayer([obsInfo.Dimension(1) 1 1],"Name","StateInput","Normalization","none");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = imageInputLayer([actInfo.Dimension(1) 1 1],"Name","ActionInput","Normalization","none");
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


critic = rlQValueRepresentation(lgraph,obsInfo,actInfo,'Observation',"StateInput",'Action',"ActionInput") ;
