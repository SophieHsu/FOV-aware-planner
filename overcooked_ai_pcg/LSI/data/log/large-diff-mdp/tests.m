clear all
close all

Lab = [];
Data = [];
bounds = {0:12,0:4,0:4};
eps = 50;
BCindx = 2;
for x= bounds{BCindx+1}
  D{x+1} = load (['BCs_' num2str(eps) '/' num2str(BCindx) '/' num2str(x) '.dat']);
  Data = [Data;D{x+1}];
  Lab = [Lab;(x+1)*ones(length(D{x+1}),1)];

end
[rho, pval] = corr(Lab,Data, 'Type','Spearman')