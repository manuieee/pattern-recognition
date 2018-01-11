% Author:  Manu M
% Student ID 795410
% Final Exam 
% Recognizing the character 'A' from the set = {A,E,I,O,U}
% Training
clear all;
clc;
%Input matrices for all characters including bias
A1=[1 1 -1 1 1 1 -1 1 -1 1 -1 -1 -1 -1 -1 -1 1 1 1 -1 1];
A2=[1 1 -1 -1 1 1 -1 1 -1 1 -1 -1 -1 -1 -1 -1 1 1 1 -1 1];
A3=[1 1 -1 1 1 1 -1 -1 -1 1 -1 -1 -1 -1 -1 -1 1 1 1 -1 1];
A4=[1 1 -1 1 1 1 -1 1 -1 1 -1 -1 -1 -1 -1 -1 -1 1 1 -1 1];
A5=[1 1 -1 1 1 1 -1 1 -1 1 -1 -1 -1 -1 -1 -1 1 1 -1 -1 1];
E1=[-1 -1 -1 -1 -1 -1 1 1 1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 -1 1];
E2=[-1 -1 -1 -1 -1 -1 1 1 1 1 -1 -1 -1 1 -1 -1 -1 -1 -1 -1 1];
E3=[-1 -1 -1 -1 -1 -1 1 1 1 -1 -1 -1 -1 1 1 -1 -1 -1 -1 -1 1];
E4=[-1 -1 -1 -1 -1 -1 -1 1 1 1 -1 -1 -1 1 1 -1 -1 -1 -1 -1 1];
E5=[-1 -1 -1 -1 -1 -1 1 1 1 1 -1 -1 -1 1 1 -1 -1 -1 -1 -1 1];
I1=[1 -1 -1 -1 1 1 1 -1 1 1 1 1 -1 1 1 1 -1 -1 -1 1 1];
I2=[-1 -1 -1 -1 1 1 1 -1 1 1 1 1 -1 1 1 1 -1 -1 -1 1 1];
I3=[1 -1 -1 -1 1 1 1 -1 -1 1 1 1 -1 -1 1 1 -1 -1 -1 1 1];
I4=[1 -1 -1 -1 1 1 1 -1 1 1 1 1 -1 1 1 -1 -1 -1 -1 1 1];
I5=[1 -1 -1 -1 1 1 1 -1 1 1 1 1 -1 1 1 1 -1 1 -1 1 1];
O1=[-1 -1 -1 -1 -1 -1 1 1 1 -1 -1 1 1 -1 -1 -1 -1 -1 -1 -1 1];
O2=[-1 -1 -1 -1 -1 -1 1 1 1 -1 -1 -1 1 1 -1 -1 -1 -1 -1 -1 1];
O3=[-1 -1 -1 -1 -1 -1 1 1 -1 -1 -1 1 1 1 -1 -1 -1 -1 -1 -1 1]; 
O4=[-1 -1 -1 -1 -1 -1 -1 1 1 -1 -1 1 1 1 -1 -1 -1 -1 -1 -1 1];
O5=[-1 -1 -1 -1 -1 -1 1 1 1 -1 -1 1 1 1 -1 -1 -1 -1 -1 -1 1];
U1=[-1 1 1 1 -1 -1 1 1 1 -1 -1 1 1 1 -1 -1 -1 -1 -1 -1 1];
U2=[-1 -1 1 1 -1 -1 1 1 1 -1 -1 1 1 1 -1 -1 -1 -1 -1 -1 1];
U3=[-1 1 1 -1 -1 -1 1 1 1 -1 -1 1 1 1 -1 -1 -1 -1 -1 -1 1];
U4=[-1 1 1 1 -1 -1 1 1 1 -1 -1 -1 1 1 -1 -1 -1 -1 -1 -1 1];
U5=[-1 1 1 1 -1 -1 1 1 1 -1 -1 1 1 -1 -1 -1 -1 -1 -1 -1 1];
% i = input matrix
i =[A1;A2;A3;A4;A5;E1;E2;E3;E4;E5;I1;I2;I3;I4;I5;O1;O2;O3;O4;O5;U1;U2;U3;U4;U5]; %size
%weight matrix
w=[0.02 0.02 -0.01 0.01 -0.03 -0.02 0.02 0.01 0.03 -0.01 -0.04 0.01 0.02 -0.03 -0.02 0.05 0.01 0.01 -0.01 -0.04 0.04]; %size
%target matrix
t=[1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]; % size
yin=[]; % net input matrix
alpha=0.001;
tol=0.0001; % toleralnce 
n=100;
Wt=[];%size
wc=[]; % change in weight
cnt=0;LW=1;
% matrices for calculating errors in each epoch
error = [];max_error=[]; mse =[];mse_track=[];maxwtc=[];
while(LW < tol | cnt < n ) % stopping condition

yin = i *w'; % size
        wnew =w + alpha*((t' - yin)'*i); %size
        Wt =[Wt;wnew]; %size
        wc=abs(wnew-w);%size
        LW = max(wc);
        w=wnew;       % assigning new weights to old weight matrix 'w'        
        maxwtc=[maxwtc;LW];  % maximum weight change             

cnt=cnt+1; %count ~ epoch


%max_error/epoch
mxerror=0;
error= t'-yin; % size
mxerror=max(error);
max_error=[max_error;mxerror];


% mean squared error/epoch
mse=sum(((error).^2))/21;
mse_track=[mse_track;mse];

end

figure(1)
%plot 1
hold on
plot(Wt(:,1),'r');
plot(Wt(:,2),'b');
plot(Wt(:,3),'y');
plot(Wt(:,4),'k');
plot(Wt(:,5),'g');
hold off
xlabel('epochs');
ylabel('weights');
title('Plot of the 1st five weights/ephoch');

% plot 2
figure(2)
plot(max_error);
xlabel('epochs');
ylabel('max_error');
title('Plot of the max error/ephoch');

%plot 3
figure(3)
plot(mse_track);

xlabel('epochs');
ylabel('mean sq error');
title('Plot of the mean squared error/epoch');

%plot 4

figure(4)
plot(maxwtc);
xlabel('epochs');
ylabel('maximum weight change');
title('Plot of the maximum weight change in each epoch');

%plot 5

figure(5)
stem(w);
xlabel('epochs');
ylabel('weights');
title('Plot of the final weights');


    


        