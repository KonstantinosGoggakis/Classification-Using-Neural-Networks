%% Clean Data 

% Read the red wine quality dataset
RedData = readtable('winequality-red.csv');

% Check  missing values 
RedData_clean = rmmissing(RedData);

% Remove outliers for the "volatileAcidity" column
RedData_clean = RedData_clean(RedData_clean.volatileAcidity <= 14, :);

% Remove outliers for the "density" column
RedData_clean = RedData_clean(RedData_clean.density <= 111, :); 

% Remove duplicate rows
RedData_clean = unique(RedData_clean, 'rows'); 

% Quality Categories
categories = cell(size(RedData_clean.quality));
for i = 1:height(RedData_clean)
    if RedData_clean.quality(i) <= 4
        categories{i} = 'χαμηλής ποιότητας'; 
    elseif RedData_clean.quality(i) == 5
        categories{i} = 'κατώτερης μέτριας ποιότητας'; 
    elseif RedData_clean.quality(i) == 6
        categories{i} = 'ανώτερης μέτριας ποιότητας'; 
    else
        categories{i} = 'υψηλής ποιότητας'; 
    end
end

% Add  'Category' column
RedData_clean.Category = categories;

% Save cleaned data to CSV file
writetable(RedData_clean, 'clean_data.csv');

%% Normalize Data

% Normalize the data excluding 'quality' and 'Category'
numeric_data = RedData_clean{:, 1:(end-2)};
min_val = min(numeric_data);
max_val = max(numeric_data);
normalized_data = (numeric_data - min_val) ./ (max_val - min_val);
normalized_data_table = array2table(normalized_data, 'VariableNames', RedData_clean.Properties.VariableNames(1:(end-2)));
normalized_data_table.quality = RedData_clean.quality;
writetable(normalized_data_table, 'normalized_data.csv');

%% Preprocess Data

% Load normalized dataset
data = readtable('normalized_data.csv');
X = data{:, 1:(end-1)}; 
Y = data.quality;       % Target variable

% Convert quality into categorical classes
categories = zeros(size(Y));
categories(Y <= 4) = 1; 
categories(Y == 5) = 2; 
categories(Y == 6) = 3; 
categories(Y >= 7) = 4; 

%% Train-Test Split and Oversampling

% Randomize the dataset
rng(42); % Set random seed for reproducibility
num_samples = size(X, 1);
indices = randperm(num_samples);

% Split data 
train_ratio = 0.8;
num_train = round(train_ratio * num_samples);
X_train = X(indices(1:num_train), :);
Y_train = categories(indices(1:num_train));
X_test = X(indices(num_train+1:end), :);
Y_test = categories(indices(num_train+1:end));

% Oversample Class 1 in the training set
class_1_indices = find(Y_train == 1);
class_2_indices = find(Y_train == 2);
class_3_indices = find(Y_train == 3);
class_4_indices = find(Y_train == 4);

oversample_factor = 5; 
oversampled_class_1 = class_1_indices;
for i = 1:oversample_factor - 1
    oversampled_class_1 = [oversampled_class_1; class_1_indices];
end

% Combine oversampled Class 1 with other classes
balanced_indices = [oversampled_class_1; class_2_indices; class_3_indices; class_4_indices];
balanced_indices = balanced_indices(randperm(length(balanced_indices)));

% Create the balanced training set
X_train_balanced = X_train(balanced_indices, :);
Y_train_balanced = Y_train(balanced_indices);

% One-hot encode target variable
Y_train_balanced_categorical = full(ind2vec(Y_train_balanced'));

%% Neural Network Design

% Create a feedforward neural network
hidden_layer_sizes = [100, 80, 60, 40, 20];
net = patternnet(hidden_layer_sizes);

% Configure transfer functions
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'logsig';
net.layers{3}.transferFcn = 'softmax';

% Configure data division
net.divideParam.trainRatio = 0.6; % 60% training
net.divideParam.valRatio = 0.1;   % 10% validation
net.divideParam.testRatio = 0.3;  % 30% testing

% Configure training parameters
net.trainFcn = 'trainscg';
net.trainParam.epochs = 1500; % Max epochs
net.trainParam.lr = 0.005; % Learning rate

% Initialize the network
net = configure(net, X_train_balanced', Y_train_balanced_categorical);

%% Train the Neural Network

[net, tr] = train(net, X_train_balanced', Y_train_balanced_categorical);

%% Evaluate the Model

% Make predictions on the test set
Y_pred = net(X_test');

% Convert predicted values to class labels
[~, predicted_classes] = max(Y_pred, [], 1);
conf_matrix = confusionmat(Y_test, predicted_classes);

%% Calculate F1 scores for each class

for i = 1:4
    TP = conf_matrix(i, i);
    FP = sum(conf_matrix(:, i)) - TP;
    FN = sum(conf_matrix(i, :)) - TP;
    if (TP + FP == 0) || (TP + FN == 0)  % Avoid division by zero
        F1_score = 0;  % F1 score will be 0 if there are no predicted instances
    else
        precision = TP / (TP + FP);
        recall = TP / (TP + FN);
        F1_score = 2 * (precision * recall) / (precision + recall);
    end
   
end

% Calculate overall accuracy
correct_predictions = sum(diag(conf_matrix));  
total_predictions = sum(conf_matrix(:));       % Total number of predictions
% Accuracy in percentage
accuracy = (correct_predictions / total_predictions) * 100;  
fprintf('Total Accuracy: %.2f%%\n', accuracy);

% Save the Trained Model
save('trained_neural_net.mat', 'net');
