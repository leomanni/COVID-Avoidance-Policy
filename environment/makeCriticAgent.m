function critic_agent = makeCriticAgent(envCovid)
    % we expect to receive a COVIDGridworld

    obsInfo = envCovid.getObservationInfo;
    actInfo = envCovid.getActionInfo;

    cells_num = obsInfo.UpperLimit;
    % nella variante in cui gli input sono tutte le celle, e non solo la
    % posizione dei giocatori:
    % cells_num = obsInfo.Dimension(1) + actInfo.Dimension(1);

    % ############## Network Create ##############

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
        fullyConnectedLayer(cells_num * 4,"Name","fc_1")
        sigmoidLayer("Name","sigmoid_1")
        fullyConnectedLayer(cells_num,"Name","fc_2")
        sigmoidLayer("Name","sigmoid_2")
        fullyConnectedLayer(1,"Name","fc_3")];
    lgraph = addLayers(lgraph,tempLayers);

    clear tempLayers; % clean up helper variable


    % Connect Layer Branches
    % Connect all the branches of the network to create the network graph.
    lgraph = connectLayers(lgraph,"StateInput","concat/in1"); %in1 da man di connectLayers
    lgraph = connectLayers(lgraph,"ActionInput","concat/in2"); %in2 da man di connectLayers

    % ############## agent SARSA create ##############

    critic = rlQValueRepresentation(lgraph,obsInfo,actInfo,'Observation',"StateInput",'Action',"ActionInput") ;

    opt = rlSARSAAgentOptions;
    opt.EpsilonGreedyExploration.Epsilon = 0.05;

    critic_agent = rlSARSAAgent(critic,opt);

end

