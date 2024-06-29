clear; clc; clf; close all;
load('filtered_QHD_table.mat')
load('filtered_QDP_table.mat')
load('deleted_QHD_table.mat')
load('deleted_QDP_table.mat')

QH = [filtered_QHD_table.FlowRate_m3h,filtered_QHD_table.Head_m]';
D  = [filtered_QHD_table.Diameter_mm]';

QH_beps=[deleted_QHD_table.FlowRate_m3h,deleted_QHD_table.Head_m]';
D_beps=[deleted_QHD_table.Diameter_mm]';

QD = [filtered_QDP_table.FlowRate_m3h,filtered_QDP_table.Diameter_mm]';
P = [filtered_QDP_table.Power_kW]';

QD_beps=[deleted_QDP_table.FlowRate_m3h,deleted_QDP_table.Diameter_mm]';
P_beps=[deleted_QDP_table.Power_kW]';



userSeed = 4826;

% Example data 
input_data = QH;
output_data = D; 

% Number of hidden layers 
num_hidden_layers = 5;

% Bounds for the number of neurons in each hidden layer
neuron_bounds = [
    2  ,   2 ,2,2,2    ;          % Lower bounds for each hidden layer
    100,  100  ,100,100,100           % Upper bounds for each hidden layer
];


% 'PopulationSize', 50, ...
%     'MaxGenerations', 7, ...
%     'FitnessLimit',0.01,...
% Define the GA options

ga_opts = optimoptions('ga', ...
    'FitnessLimit',0.01);


% ga_opts = optimoptions('ga', 'MaxTime',3);



% [QH(1,:); D], QH(2,:)
% Call the optimization routine
[best_net, best_params,nnPerfVect] = optimize_nn_hyperparameters(QD,P, num_hidden_layers, neuron_bounds, ga_opts,userSeed);

% Display results
disp('Best Parameters:');
disp(best_params);
disp('train MSE :')
disp(nnPerfVect)






function [best_net_global, best_params, nnPerfVect] = optimize_nn_hyperparameters(input_data, output_data, num_hidden_layers, neuron_bounds, ga_opts, userSeed)
    % Set the seed for reproducibility
    rng(userSeed);

    % Validate neuron_bounds dimensions
    assert(size(neuron_bounds, 1) == 2, 'neuron_bounds should have 2 rows for lower and upper bounds');
    assert(size(neuron_bounds, 2) == num_hidden_layers, 'neuron_bounds should have columns equal to num_hidden_layers');

    % Flatten the neuron bounds for GA
    lb = reshape(neuron_bounds(1, :), 1, []);
    ub = reshape(neuron_bounds(2, :), 1, []);

    disp('lower bounds :')
    disp(lb)
    disp('upper bounds :')
    disp(ub)

    % Define the global variable to store the best net
    global best_net_global nnPerfVect;
    best_net_global = [];
    nnPerfVect = [];


    % Define the fitness function
    fitnessFcn = @(x) nn_fitness(x, input_data, output_data, num_hidden_layers, userSeed,false);

    % Run the GA
    [best_params, ~] = ga(fitnessFcn, num_hidden_layers , [], [], [], [], lb, ub, [], ga_opts);
    best_params = round(best_params);

    nn_fitness(best_params, input_data, output_data, num_hidden_layers, userSeed,true);
    plot_nn_vs_real_data(input_data,output_data,best_net_global)

    function plot_nn_vs_real_data(input_data,output_data,best_net_global)

            % Plot results if input data is 2D or 3D
    input_features_size = size(input_data);
    if input_features_size(1) == 2
                        % 2D Plot
                        figure
                        scatter(input_data(1, :), output_data, 'r')
                        hold on;
                        scatter(input_data(1, :), best_net_global(input_data), 'b')
                        xlabel('Input Feature 1');
                        ylabel('Output');
                        legend('Actual Output', 'NN Output');
                        title('2D Plot of Actual vs. NN Output');
                        xlim([min(input_data(1, :)) max(input_data(1, :))]);
                        ylim([min(transpose(output_data)) max(transpose(output_data))]);
                        grid on;
                        hold off;
                
                        % 2D Plot
                        figure
                        scatter(input_data(2, :), output_data, 'r');
                        hold on;
                        scatter(input_data(2, :), best_net_global(input_data), 'b');
                        xlabel('Input Feature 2');
                        ylabel('Output');
                        legend('Actual Output', 'NN Output');
                        title('2D Plot of Actual vs. NN Output');
                        xlim([min(input_data(2, :)) max(input_data(2, :))]);
                        ylim([min(transpose(output_data)) max(transpose(output_data))]);
                        grid on;
                        hold off;
                
                        % 3D Plot
                        figure
                        scatter3(input_data(1, :), input_data(2, :), output_data, 'r')
                        hold on;
                        scatter3(input_data(1, :), input_data(2, :), best_net_global(input_data), 'b')
                        xlabel('Input Feature 1');
                        ylabel('Input Feature 2');
                        zlabel('Output');
                        legend('Actual Output', 'NN Output');
                        title('3D Plot of Actual vs. NN Output');
                        xlim([min(input_data(1, :)) max(input_data(1, :))]);
                        ylim([min(input_data(2, :)) max(input_data(2, :))]);
                        zlim([min(output_data) max(output_data)]);
                        grid on;
                        hold off;

    end

    end

    function avgMSEs = nn_fitness(params, input_data, output_data, num_hidden_layers, userSeed,showWindow)

    
        % Set the seed for reproducibility
        rng(userSeed);
    
        % Round the relevant parameters
        neurons = round(params(1:num_hidden_layers));
        disp('current hidden layers neurons :')
        disp(neurons)
    
        % Create the neural network
        net = feedforwardnet(neurons);
    
        % Set other properties
        net.trainParam.showWindow = showWindow;
        net.input.processFcns = {'removeconstantrows', 'mapminmax'};
        net.output.processFcns = {'removeconstantrows', 'mapminmax'};
        net.divideFcn = 'dividerand';
        net.divideMode = 'sample';
        net.divideParam.trainRatio = 0.7;
        net.divideParam.valRatio = 0.15;
        net.divideParam.testRatio = 0.15;
    
        % Train the network
        [net, tr] = train(net, input_data, output_data);
    
        % calculate Training, Validation and Test Performance
        trainPerformance    = perform(net, output_data(:, tr.trainInd), net(input_data(:, tr.trainInd)));
        valPerformance      = perform(net, output_data(:, tr.valInd), net(input_data(:, tr.valInd)));
        testPerformance     = perform(net, output_data(:, tr.testInd), net(input_data(:, tr.testInd)));

        % for better performance we came up with this convention rather than
        % using the mse based on perform function only
        avgMSEs = (trainPerformance+valPerformance+testPerformance) / 3;
        best_net_global = net;
        nnPerfVect = [trainPerformance,valPerformance,testPerformance];

        % Log the results
        log_nn_details(best_net_global, neurons, [avgMSEs,nnPerfVect], userSeed);


        disp('current performance: ')
        disp([avgMSEs,nnPerfVect])


    end
    
    function log_nn_details(net, params, perfVect, seed)
        logFilePath='local_nn_logs.mat';
            % Define the log entry
            logEntry = struct();
            logEntry.DateTime = datetime('now');
            logEntry.Seed = seed;
            logEntry.Net = net;
            logEntry.NumHiddenLayers = numel(params);
            logEntry.Neurons = params;
            logEntry.AvgMSE = perfVect(1);
            logEntry.TrainMSE = perfVect(2);
            logEntry.ValMSE = perfVect(3);
            logEntry.TestMSE = perfVect(4);
    
            % Check if the log file exists
            if isfile(logFilePath)
                % Load existing log
                load(logFilePath, 'local_nn_logs');
            else
                % Initialize new log
                local_nn_logs = {};
            end
    
            % Append the new entry
            local_nn_logs{end + 1} = logEntry;
    
            % Save the updated log
            save(logFilePath, 'local_nn_logs');
        end
end




% function [best_net, best_params, train_mse ,val_mse, test_mse, avg_mse] = optimize_nn_hyperparameters(input_data, output_data, num_hidden_layers, neuron_bounds,  ga_opts,userSeed)
   
%  rng(userSeed);


% % Validate neuron_bounds dimensions
% assert(size(neuron_bounds, 1) == 2, 'neuron_bounds should have 2 rows for lower and upper bounds');
% assert(size(neuron_bounds, 2) == num_hidden_layers, 'neuron_bounds should have columns equal to num_hidden_layers');


% % Flatten the neuron bounds for GA
% lb = [reshape(neuron_bounds(1, :), 1, [])];
% ub = [reshape(neuron_bounds(2, :), 1, [])];
% disp('lower bounds :')
% disp(lb)
% disp('upper bounds :')
% disp(ub)
% % Define the fitness function
% fitnessFcn = @(x) nn_fitness(x, input_data, output_data, num_hidden_layers,userSeed);

% % Run the GA
% [best_params, ~] = ga(fitnessFcn, num_hidden_layers + 3, [], [], [], [], lb, ub, [], ga_opts);
% best_params = round(best_params);


% % Extract the best parameters
% neurons = round(best_params(1:num_hidden_layers));


% % Create and train the best neural network
% best_net = feedforwardnet(neurons);


% best_net.trainParam.showWindow = false;

% best_net.input.processFcns = {'removeconstantrows', 'mapminmax'};
% best_net.output.processFcns = {'removeconstantrows', 'mapminmax'};
% best_net.divideFcn = 'dividerand';
% best_net.divideMode = 'sample';
% best_net.divideParam.trainRatio = 0.7;
% best_net.divideParam.valRatio = 0.15;
% best_net.divideParam.testRatio = 0.15;

% [best_net, tr] = train(best_net, input_data, output_data);

% % Evaluate performance
% train_mse = perform(best_net, output_data(:, tr.trainInd), best_net(input_data(:, tr.trainInd)));
% val_mse = perform(best_net, output_data(:, tr.valInd), best_net(input_data(:, tr.valInd)));
% test_mse = perform(best_net, output_data(:, tr.testInd), best_net(input_data(:, tr.testInd)));
% avg_mse = (val_mse + test_mse) / 2;

% % Plot results if input data is 2D or 3D
% input_features_size = size(input_data);
% if input_features_size(1) == 2
% % 2D Plot
% figure;
% scatter(input_data(1, :), output_data, 'r');
% hold on;
% scatter(input_data(1, :), best_net(input_data), 'b');
% xlabel('Input Feature 1');
% ylabel('Output');
% legend('Actual Output', 'NN Output');
% title('2D Plot of Actual vs. NN Output');

% xlim([min(input_data(1, :)) max(input_data(1, :))]);
% ylim([min(transpose(output_data)) max(transpose(output_data))]);

% grid on;
% hold off;

% % 2D Plot
% figure;
% scatter(input_data(2, :), output_data, 'r');
% hold on;
% scatter(input_data(2, :), best_net(input_data), 'b');
% xlabel('Input Feature 2');
% ylabel('Output');
% legend('Actual Output', 'NN Output');
% title('2D Plot of Actual vs. NN Output');

% xlim([min(input_data(2, :)) max(input_data(2, :))]);
% ylim([min(transpose(output_data)) max(transpose(output_data))]);

% grid on;
% hold off;


% % 3D Plot
% figure;
% scatter3(input_data(1, :), input_data(2, :), output_data, 'r');
% hold on;
% scatter3(input_data(1, :), input_data(2, :), best_net(input_data), 'b');
% xlabel('Input Feature 1');
% ylabel('Input Feature 2');
% zlabel('Output');
% legend('Actual Output', 'NN Output');
% title('3D Plot of Actual vs. NN Output');

% xlim([min(input_data(1, :)) max(input_data(1, :))]);
% ylim([min(input_data(2, :)) max(input_data(2, :))]);
% zlim([min(output_data) max(output_data)]);

% grid on;
% hold off;

% end

% function mse = nn_fitness(params, input_data, output_data, num_hidden_layers,userSeed)
%     rng(userSeed);
% % Round the relevant parameters
% neurons = round(params(1:num_hidden_layers));


% % Create the neural network
% net = feedforwardnet(neurons);



% % Set other properties
% net.trainParam.showWindow = false;

% net.input.processFcns = {'removeconstantrows', 'mapminmax'};
% net.output.processFcns = {'removeconstantrows', 'mapminmax'};
% net.divideFcn = 'dividerand';
% net.divideMode = 'sample';
% net.divideParam.trainRatio = 0.7;
% net.divideParam.valRatio = 0.15;
% net.divideParam.testRatio = 0.15;

% % Train the network
% [net, tr] = train(net, input_data, output_data);

% % Evaluate performance
% val_mse = perform(net, output_data(:, tr.valInd), net(input_data(:, tr.valInd)));
% test_mse = perform(net, output_data(:, tr.testInd), net(input_data(:, tr.testInd)));

% % Fitness is the average of validation and test MSE
% mse = (val_mse + test_mse) / 2;

% end

% end 










% % Call the optimization routine
% optimize_nn_hyperparameters(input_data, output_data, num_hidden_layers, neuron_bounds, epoch_bounds, train_methods, activation_functions, optimize_layers, ga_opts,userSeed);





% function optimize_nn_hyperparameters(input_data, output_data, num_hidden_layers, ...
%     neuron_bounds, epoch_bounds, train_methods, ...
%     activation_functions, optimize_layers, ga_opts,userSeed)

% % Validate neuron_bounds dimensions
% assert(size(neuron_bounds, 1) == 2, 'neuron_bounds should have 2 rows for lower and upper bounds');
% assert(size(neuron_bounds, 2) == num_hidden_layers, 'neuron_bounds should have columns equal to num_hidden_layers');

% % Number of hidden layers bounds (1 if not optimizing, num_hidden_layers if optimizing)
% if optimize_layers
% num_layers_lb = 1;
% num_layers_ub = num_hidden_layers;
% else
% num_layers_lb = num_hidden_layers;
% num_layers_ub = num_hidden_layers;
% end

% % Flatten the neuron bounds for GA
% lb = [reshape(neuron_bounds(1, :), 1, []), epoch_bounds(1), 1, 1];
% ub = [reshape(neuron_bounds(2, :), 1, []), epoch_bounds(2), length(train_methods), length(activation_functions)];

% % Define the fitness function
% fitnessFcn = @(x) nn_fitness(x, input_data, output_data, num_hidden_layers, train_methods, activation_functions);


% rng(userSeed);
% % Run the GA
% [opt_params, opt_fitness] = ga(fitnessFcn, num_hidden_layers + 3, [], [], [], [], lb, ub, [], ga_opts);

% disp('Optimal Parameters:');
% disp(opt_params);
% disp('Optimal Fitness:');
% disp(opt_fitness);
% end

% function mse = nn_fitness(params, input_data, output_data, num_hidden_layers, train_methods, activation_functions,userSeed)
%     rng(userSeed);
% % Extract parameters
% neurons = params(1:num_hidden_layers);
% epochs = params(num_hidden_layers + 1);
% train_idx = params(num_hidden_layers + 2);
% act_idx = params(num_hidden_layers + 3);

% % Create the neural network
% net = feedforwardnet(neurons);
% net.trainFcn = train_methods{train_idx};
% net.layers{1}.transferFcn = activation_functions{act_idx};

% % Set other properties
% net.trainParam.showWindow = false;
% net.trainParam.epochs = epochs;
% net.input.processFcns = {'removeconstantrows', 'mapminmax'};
% net.output.processFcns = {'removeconstantrows', 'mapminmax'};
% net.divideFcn = 'dividerand';
% net.divideMode = 'sample';
% net.divideParam.trainRatio = 0.7;
% net.divideParam.valRatio = 0.15;
% net.divideParam.testRatio = 0.15;

% % Train the network
% [net, tr] = train(net, input_data, output_data);

% % Evaluate performance
% val_mse = perform(net, output_data(:, tr.valInd), net(input_data(:, tr.valInd)));
% test_mse = perform(net, output_data(:, tr.testInd), net(input_data(:, tr.testInd)));

% % Fitness is the average of validation and test MSE
% mse = (val_mse + test_mse) / 2;
% end
