%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% creating dataset for NN toolbox

clear all;
close all;
clc;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% creating input variables with the specific length and numbers

% specificy the constants of random numbers
inputNumber = 2;
lengthOfInput = 50;

outputNumber = 1;
lengthOfOutput = lengthOfInput;

% define input matrices
inputData = zeros(lengthOfInput,inputNumber+1);
% define output matrices
outputData = zeros(lengthOfOutput,outputNumber);

% define time series data
timeData = [1:1:lengthOfInput]';


%% generating inputMatrices with specific outputValues

for i = 1 : 1 : length(inputData(:,1)) % for length search
   
    for j = 1 : 1 : length(inputData(1,:)) % for different input search
        
        if i <= length(inputData(:,1))/2   % only the half of data assigned to one class
        
            if j == 1
                % bias section is added to the code
                inputData(i,j) = 1;
            
            else
            
                % these values can be changed to represent the different
                % set of input variables
                
                % max-min values
                maxVal = 0.1;
                minVal = -0.5;

                inputData(i,j) = minVal + (maxVal - minVal)*rand(1,1);
            
            end
            
        else
   
            if j == 1
                % bias section is added to the code
                inputData(i,j) = 1;
            
            else
                        
                % these values can be changed to represent the different
                % set of input variables
                
                % max-min values
                maxVal = 2;
                minVal = 1;

                inputData(i,j) = minVal + (maxVal - minVal)*rand(1,1);
            
            end
            
        end
    
    end
    
end

for i = 1 : 1 : length(outputData(:,1))  % for length search 

    for j = 1 : 1 : length(outputData(1,:))  % for different input search
    
        if i <= length(outputData(:,1))/2    % only the half of data assigned to one class
           
            % first output assigned to the class of 5!
            outputData(i,j) = 5;
        else
            % second output assigned to the class of 10!
            outputData(i,j) = 10;
            
        end
        
    end
    
end


%% plotting the results
figure
plot(timeData, inputData(:,1))
hold on
plot(timeData, inputData(:,2))
hold on
plot(timeData, inputData(:,3))
hold on
plot(timeData, outputData(:,1))
legend('Bias Term','First Input','Second Input','OutputData')
xlabel('Time Data (no units)')
ylabel('Output Data (no units)')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% generating one matrices to store the whole data

InOutMatrices = [inputData,outputData];

infoSection = [length(InOutMatrices(:,1)), length(inputData(1,:)), length(outputData(1,:)),zeros(1, length(InOutMatrices(1,:)) - 3)]; 
InOutMatrices = [infoSection;InOutMatrices];


%% creating input variables with the specific length and numbers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% writing the whole data to the text or excel files!

%% creating the proper file

% establishing writing format
writeFormat = "";

for i = 1 : 1 : length(InOutMatrices(1,:))
   
    writeFormat = writeFormat + "%f ";
    
    if i == length(InOutMatrices(1,:))
        
         writeFormat = writeFormat + "\n";
         
     end


end

% writing data to excel file

filename = 'nnInputOutputFile.csv';
writematrix(InOutMatrices,filename);


%% writing the whole data to the text or excel files!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%










%% creating dataset for NN toolbox
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
