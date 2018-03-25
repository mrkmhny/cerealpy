%
% Analyze the nutrition facts of breakfast cereals 
% to predict customer rating
%

% Load in the data
data = csvread('cereal.csv');

% Remove header and empty row
data =  data(2:end,:);

% Shuffle all rows
data = data(randperm(size(data,1)),:);

% Remove irrelevent columns
data = data(:,[4,5,6,7,8,9,10,11,12,14,15,16]);

% Use a smaller dataset for easier workflow
% REMOVE THIS REMOVE THIS REMOVE THIS
% data = data(1:15,:);

% COLUMN NAMES for reference -- header(orig -> after modification) 
% name(1),mfr(2),type(3),calories(4->1),protein(5->2),
% fat(6->3),sodium(7->4),fiber(8->5),carbo(9->6),sugars(10->7),
% potass(11->8),vitamins(12->9),shelf(13),weight(14->10),cups(15->11),rating(16->12)

% Adjust all features to be proportionally correct based on serving size
data = [data(:,1:size(data)(2)-1) .* (1./data(:,10)), data(:,12)];

% Add a column of ones to represent y intercept
data = [ones(size(data)(1),1) data];

% Split out trainingSet and test sets by 80% and 20%
trainingSet = data(1:floor(size(data)(1)*0.8),:);
testSet = data(floor(size(data)(1)*0.8)+1:end, :);

% Initialize parameters
X = trainingSet(:,1:size(trainingSet)(2)-3);

y = trainingSet(:,size(trainingSet)(2));
m = size(X)(1);
theta = zeros(size(X)(2),1);
alpha = 0.3;
iterations = 200;

% Normalize all features
mu = mean(X(:,2:size(X)(2)));
sigma = std(X);
Xrange = range( X(:,2:size(X)(2)));
X_norm = [X(:,1) ( (X(:,2:size(X)(2)) - mu ) ./ Xrange )];

% clear J
J = [];

% Perform gradient descent
for i = 1:iterations
  theta = theta - alpha*(1/m)*(sum(((sum((X_norm .* theta'),2)-y).* X_norm)))';

  % Save Cost Function
  J(1,i) = (1/(2*m)) * sum((sum((X_norm .* theta'),2) - y).^2);
  % Add iteration number for X axis
  J(2,i) = i;
end

% Plot change in cost function over time
plot(J'(:,1), J'(:,2))

% testSet results
Xtest = testSet(:,1:size(trainingSet)(2)-3);
Xtest_norm = [Xtest(:,1) ( (Xtest(:,2:size(Xtest)(2)) - mu ) ./ Xrange )];
ytest = testSet(:,size(testSet)(2));
predicted = sum(Xtest_norm .* theta',2);

difference = sum(Xtest_norm .* theta',2) - ytest;

resultComparison = [predicted ytest difference]
averageError = mean(difference)
