% Creation of a SARSA NN agent for COVIDGridworlds.
% -------------------------------------------------------------------------
% Roberto Masocco, Edoardo Rossi, Leonardo Manni, Filippo Badalamenti,
% Emanuele Alfano
% August 9, 2021

function sarsa_agent = makeCriticAgent(envCovid)
    % MAKECRITICAGENT Creates a SARSA NN agent tailored to a
    % COVIDGridworld.

    obsInfo = envCovid.getObservationInfo;
    actInfo = envCovid.getActionInfo;

    cells_num = envCovid.num_cells;
    % nella variante in cui gli input sono tutte le celle, e non solo la
    % posizione dei giocatori:
    % cells_num = obsInfo.Dimension(1) + actInfo.Dimension(1);

    % ############## Network Creation ##############

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
        fullyConnectedLayer(cells_num * 8,"Name","fc_1")
        sigmoidLayer("Name","sigmoid_1")
        fullyConnectedLayer(cells_num * 2,"Name","fc_2")
        sigmoidLayer("Name","sigmoid_2")
        fullyConnectedLayer(1,"Name","fc_3")];
    lgraph = addLayers(lgraph,tempLayers);

    clear tempLayers; % clean up helper variable

    % Connect Layer Branches
    % Connect all the branches of the network to create the network graph.
    lgraph = connectLayers(lgraph,"StateInput","concat/in1"); %in1 da man di connectLayers
    lgraph = connectLayers(lgraph,"ActionInput","concat/in2"); %in2 da man di connectLayers

    % ############## Critic Creation ##############
    device = "cpu";
    critic_opt = rlRepresentationOptions('UseDevice',device, "Optimizer","adam", 'LearnRate',0.025);
    critic = rlQValueRepresentation(lgraph,obsInfo,actInfo,'Observation',"StateInput",'Action',"ActionInput", critic_opt) ;

    % ############## Agent Creation ##############
    optSarsa = rlSARSAAgentOptions;
    optSarsa.EpsilonGreedyExploration.Epsilon = 0.6;
    optSarsa.EpsilonGreedyExploration.EpsilonDecay = 0.001;
    optSarsa.EpsilonGreedyExploration.EpsilonMin = 0.01;
    optSarsa.DiscountFactor = 1;
    sarsa_agent = rlSARSAAgent(critic,optSarsa);
    
%     optDQL = rlDQNAgentOptions('MiniBatchSize',48);
%     optDQL.EpsilonGreedyExploration.Epsilon = 0.05;
%     DQL_agent = rlDQNAgent(critic,optDQL);

%     optDQL = rlAgentInitializationOptions('NumHiddenUnit',cells_num * 4);
%     DQL_agent = rlDQNAgent(obsInfo, actInfo ,optDQL);
end